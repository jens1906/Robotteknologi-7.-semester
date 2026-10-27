"""Who is the leader: enrolment, the stored views, finding them again, lost / last seen. Plain python, no ROS.

To include:
- States: IDLE, ENROLLING, FOLLOWING, SEARCHING, LOST (the values of leader_state in PerceptionDebugFrame).
- The leader is described by stored views in five groups: front, back, side left, side right and partly
  hidden (body_view.py decides the group from the keypoints). At most viewsPerGroup views per group.
- Enrolment (calibrate service): the leader turns around once; up to enrollViewsPerDirection clean views
  per facing direction, enrollGap apart, nobody overlapping. Done when all four directions are filled,
  or after enrollDuration (or when the person leaves) when enrollMinimumDirections are filled. These are the anchors and are never
  dropped. Enrolment survives a tracker id change at the same place.
- While FOLLOWING: locked to the tracker id; the person is compared with the stored views every
  verifyPeriod (an OSNet check, so a swap of tracker ids is noticed). A view is learned only when we are
  sure who it is: a strong match (below reidAccept) at most trustTime ago on the same unbroken track,
  a clean photo, no other box touching the leader. A new view replaces the most similar learned view of
  its group, so the views stay varied. The lock is dropped when the person looks different (reidDrop),
  when somebody overlaps them (crossingIou, OSNet cannot separate overlapping people) or when they are
  missing longer than graceTime (flicker).
- While SEARCHING: compare searchCandidates clean people every searchPeriod (promising ones first, then
  the ones compared longest ago); each candidate's last averageViews views are averaged before comparing;
  accept only below reidAccept, with reidMargin over every other visible person, confirmed reidConfirm
  times in the last reidConfirmWindow comparisons.
- Precision first: "not identified" is fine, the wrong person is the worst failure.
- Keeps the time the leader was last identified; SEARCHING becomes LOST after lostTime.
- visibleLeaderTrackId is the leader's track id only in frames where they are identified, else -1.
- Save the anchors with their group to a file (numpy, no pickle, tagged with the reid model) and load
  them at start, so the leader survives a restart.
- Embeddings come in through a callable, so this file needs no model.
"""

import math
import os
from collections import deque
from types import SimpleNamespace

import numpy as np

from perception.body_view import (BODY_VIEW_SETTING_DEFAULTS, FACING_GROUPS, UNKNOWN_GROUP, VIEW_GROUPS,
                                  FindViewGroup, IsCleanView)

# Parameter defaults (set in config/perception_params.yaml, the origin of each value is written there)
LEADER_SETTING_DEFAULTS = {
    'reidAccept': 0.45,
    'reidDrop': 0.82,
    'reidMargin': 0.05,
    'reidConfirm': 3,
    'reidConfirmWindow': 6,
    'verifyPeriod': 0.2,
    'searchPeriod': 0.3,
    'searchCandidates': 4,
    'averageViews': 5,
    'enrollViewsPerDirection': 3,
    'enrollMinimumDirections': 1,
    'enrollGap': 0.4,
    'enrollDuration': 8.0,
    'enrollTimeout': 20.0,
    'enrollGraceTime': 2.0,
    'enrollRebindIou': 0.6,
    'viewsPerGroup': 6,
    'noveltyDistance': 0.25,
    'trustTime': 3.0,
    'graceTime': 1.0,
    'lostTime': 10.0,
    'crossingIou': 0.15,
    'competitorAge': 3.0,
    'loseAfterChecks': 2,
    'recentDistanceTime': 2.0,
    **BODY_VIEW_SETTING_DEFAULTS,
}

# Leader states (the same numbers as leader_state in PerceptionDebugFrame)
IDLE_STATE = 0
ENROLLING_STATE = 1
FOLLOWING_STATE = 2
SEARCHING_STATE = 3
LOST_STATE = 4

# Events and ids
NO_EVENT = ''
ENROLLED_EVENT = 'enrolled'
ENROLL_FAILED_EVENT = 'enroll_failed'
LOST_EVENT = 'lost'
FOUND_EVENT = 'found'
NO_TRACK_ID = -1


class LeaderManager:
    def __init__(self, leaderSettings=None):
        self.settings = SimpleNamespace(**{**LEADER_SETTING_DEFAULTS, **(leaderSettings or {})})
        self.distanceByTrackId = {}
        self.distanceSecondsByTrackId = {}
        self.Reset()

    def Reset(self):
        self.state = IDLE_STATE
        self.visibleLeaderTrackId = NO_TRACK_ID
        self.eventReason = ''
        self.followedTrackId = NO_TRACK_ID
        self.viewEmbeddings = {viewGroup: [] for viewGroup in VIEW_GROUPS}
        self.anchorCounts = {viewGroup: 0 for viewGroup in VIEW_GROUPS}
        self.enrollTrackId = NO_TRACK_ID
        self.enrollLastPerson = None
        self.enrollmentRebound = False
        self.enrollStartSeconds = 0.0
        self.enrollLastSeenSeconds = 0.0
        self.lastSampleSeconds = 0.0
        self.lastVerifySeconds = 0.0
        self.lastSearchSeconds = 0.0
        self.lastStrongMatchSeconds = -math.inf
        self.searchStartSeconds = None
        self.absentSinceSeconds = None
        self.lastSeenSeconds = 0.0
        self.badCheckCount = 0
        self.winHistories = {}
        self.recentEmbeddingsByTrackId = {}
        self.distanceByTrackId.clear()
        self.distanceSecondsByTrackId.clear()

    def Enroll(self, trackId, nowSeconds):
        self.Reset()
        self.state = ENROLLING_STATE
        self.enrollTrackId = trackId
        self.enrollStartSeconds = nowSeconds
        self.enrollLastSeenSeconds = nowSeconds

    def Update(self, detectedPeople, embedPeople, nowSeconds, imageWidth):
        self.eventReason = ''
        self.visibleLeaderTrackId = NO_TRACK_ID
        if self.state == ENROLLING_STATE:
            return self.UpdateEnrolling(detectedPeople, embedPeople, nowSeconds, imageWidth)
        if self.state == FOLLOWING_STATE:
            return self.UpdateFollowing(detectedPeople, embedPeople, nowSeconds, imageWidth)
        if self.state in (SEARCHING_STATE, LOST_STATE):
            return self.UpdateSearching(detectedPeople, embedPeople, nowSeconds, imageWidth)
        return NO_EVENT

    def CountStoredViews(self):
        return {viewGroup: len(embeddings) for viewGroup, embeddings in self.viewEmbeddings.items()}

    def SaveLeader(self, leaderFilePath, modelTag):
        anchorEmbeddings = [embedding for viewGroup in VIEW_GROUPS
                            for embedding in self.viewEmbeddings[viewGroup][:self.anchorCounts[viewGroup]]]
        anchorGroups = [viewGroup for viewGroup in VIEW_GROUPS for _ in range(self.anchorCounts[viewGroup])]
        os.makedirs(os.path.dirname(leaderFilePath) or '.', exist_ok=True)
        np.savez(leaderFilePath, embeddings=np.stack(anchorEmbeddings), viewGroups=np.array(anchorGroups),
                 modelTag=modelTag)

    def LoadLeader(self, leaderFilePath, modelTag):
        try:
            savedLeader = np.load(leaderFilePath, allow_pickle=False)
            if str(savedLeader['modelTag']) != modelTag:
                return False
            embeddings = savedLeader['embeddings'].astype(np.float32)
            viewGroups = savedLeader['viewGroups'].astype(int)
        except (OSError, KeyError, ValueError):
            return False
        if embeddings.ndim != 2 or len(embeddings) != len(viewGroups):
            return False
        self.Reset()
        for embedding, viewGroup in zip(embeddings, viewGroups):
            self.viewEmbeddings[int(viewGroup)].append(embedding)
            self.anchorCounts[int(viewGroup)] += 1
        self.state = SEARCHING_STATE
        return True

    def GetRecentDistance(self, trackId, nowSeconds):
        distanceSeconds = self.distanceSecondsByTrackId.get(trackId)
        if distanceSeconds is None or nowSeconds - distanceSeconds > self.settings.recentDistanceTime:
            return math.nan
        return self.distanceByTrackId[trackId]

    def UpdateEnrolling(self, detectedPeople, embedPeople, nowSeconds, imageWidth):
        person = self.FindPerson(detectedPeople, self.enrollTrackId)
        if person is None and self.enrollLastPerson is not None:
            person = self.RebindEnrollment(detectedPeople)
        if person is not None:
            self.enrollLastSeenSeconds = nowSeconds
            self.enrollLastPerson = person
        enrolledSeconds = nowSeconds - self.enrollStartSeconds
        enoughDirections = self.CountFilledDirections() >= self.settings.enrollMinimumDirections
        personGone = nowSeconds - self.enrollLastSeenSeconds > self.settings.enrollGraceTime
        if (personGone or enrolledSeconds > self.settings.enrollDuration) and enoughDirections:
            return self.FinishEnrollment(person, nowSeconds)
        if personGone:
            self.Reset()
            self.eventReason = 'person gone'
            return ENROLL_FAILED_EVENT
        if enrolledSeconds > self.settings.enrollTimeout:
            filledDirections = self.CountFilledDirections()
            self.Reset()
            self.eventReason = f'timeout, only {filledDirections} directions'
            return ENROLL_FAILED_EVENT
        if person is None or nowSeconds - self.lastSampleSeconds < self.settings.enrollGap:
            return NO_EVENT
        if self.IsOverlapped(person, detectedPeople) or not IsCleanView(person, imageWidth, self.settings):
            return NO_EVENT
        viewGroup = FindViewGroup(person, self.settings)
        if viewGroup not in FACING_GROUPS or self.anchorCounts[viewGroup] >= self.settings.enrollViewsPerDirection:
            return NO_EVENT
        embedding = embedPeople([person.trackId])[0]
        self.lastSampleSeconds = nowSeconds
        if self.enrollmentRebound and self.viewEmbeddings[viewGroup]:
            if CalculateDistances(embedding, np.stack(self.viewEmbeddings[viewGroup])).min() > self.settings.reidDrop:
                self.Reset()
                self.eventReason = 'person changed'
                return ENROLL_FAILED_EVENT
            self.enrollmentRebound = False
        self.viewEmbeddings[viewGroup].append(embedding)
        self.anchorCounts[viewGroup] += 1
        if self.CountFilledDirections() < len(FACING_GROUPS):
            return NO_EVENT
        return self.FinishEnrollment(person, nowSeconds)

    def CountFilledDirections(self):
        return sum(self.anchorCounts[viewGroup] >= self.settings.enrollViewsPerDirection for viewGroup in FACING_GROUPS)

    def FinishEnrollment(self, person, nowSeconds):
        self.state = FOLLOWING_STATE
        self.followedTrackId = self.enrollTrackId
        self.visibleLeaderTrackId = self.enrollTrackId if person is not None else NO_TRACK_ID
        self.lastVerifySeconds = nowSeconds
        self.lastSeenSeconds = nowSeconds
        self.lastStrongMatchSeconds = nowSeconds
        self.absentSinceSeconds = None if person is not None else nowSeconds
        return ENROLLED_EVENT

    def UpdateFollowing(self, detectedPeople, embedPeople, nowSeconds, imageWidth):
        person = self.FindPerson(detectedPeople, self.followedTrackId)
        if person is None:
            if self.absentSinceSeconds is None:
                self.absentSinceSeconds = nowSeconds
            if nowSeconds - self.absentSinceSeconds > self.settings.graceTime:
                return self.LoseLeader('track gone', nowSeconds)
            return NO_EVENT
        returned = self.absentSinceSeconds is not None
        self.absentSinceSeconds = None
        self.lastSeenSeconds = nowSeconds
        if self.IsOverlapped(person, detectedPeople):
            return self.LoseLeader('crossing', nowSeconds)
        if returned:
            self.lastStrongMatchSeconds = -math.inf
        if returned or nowSeconds - self.lastVerifySeconds >= self.settings.verifyPeriod:
            embedding = embedPeople([person.trackId])[0]
            self.lastVerifySeconds = nowSeconds
            distance = self.CalculateDistanceToLeader(embedding)
            self.distanceByTrackId[person.trackId] = distance
            self.distanceSecondsByTrackId[person.trackId] = nowSeconds
            if returned and not distance < self.settings.reidAccept:
                return self.LoseLeader('not sure after a gap', nowSeconds)
            if distance > self.settings.reidDrop:
                self.badCheckCount += 1
                if self.badCheckCount >= self.settings.loseAfterChecks:
                    return self.LoseLeader('looks different', nowSeconds)
            else:
                self.badCheckCount = 0
                self.LearnIfSureWhoItIs(person, detectedPeople, embedding, distance, nowSeconds, imageWidth)
        self.visibleLeaderTrackId = person.trackId
        return NO_EVENT

    def LearnIfSureWhoItIs(self, person, detectedPeople, embedding, distance, nowSeconds, imageWidth):
        recentlyMatched = nowSeconds - self.lastStrongMatchSeconds <= self.settings.trustTime
        if distance < self.settings.reidAccept:
            self.lastStrongMatchSeconds = nowSeconds
        if not recentlyMatched or self.IsTouched(person, detectedPeople) or not IsCleanView(person, imageWidth, self.settings):
            return
        viewGroup = FindViewGroup(person, self.settings)
        if viewGroup != UNKNOWN_GROUP:
            self.LearnView(embedding, viewGroup)

    def LearnView(self, embedding, viewGroup):
        groupEmbeddings = self.viewEmbeddings[viewGroup]
        if groupEmbeddings and CalculateDistances(embedding, np.stack(groupEmbeddings)).min() < self.settings.noveltyDistance:
            return
        if len(groupEmbeddings) < self.settings.viewsPerGroup:
            groupEmbeddings.append(embedding)
            return
        anchorCount = self.anchorCounts[viewGroup]
        learnedEmbeddings = groupEmbeddings[anchorCount:]
        if not learnedEmbeddings:
            return
        mostSimilarIndex = anchorCount + int(CalculateDistances(embedding, np.stack(learnedEmbeddings)).argmin())
        groupEmbeddings[mostSimilarIndex] = embedding

    def UpdateSearching(self, detectedPeople, embedPeople, nowSeconds, imageWidth):
        if self.searchStartSeconds is None:
            self.searchStartSeconds = nowSeconds
        searchedSeconds = nowSeconds - self.searchStartSeconds
        self.state = LOST_STATE if searchedSeconds > self.settings.lostTime else SEARCHING_STATE
        if nowSeconds - self.lastSearchSeconds < self.settings.searchPeriod:
            return NO_EVENT
        candidates = [person for person in detectedPeople
                      if not self.IsOverlapped(person, detectedPeople) and IsCleanView(person, imageWidth, self.settings)]
        if not candidates:
            return NO_EVENT
        rankedCandidates = sorted(candidates, key=lambda person: self.RankCandidate(person, nowSeconds))
        comparedPeople = rankedCandidates[:self.settings.searchCandidates]
        self.lastSearchSeconds = nowSeconds
        embeddings = embedPeople([person.trackId for person in comparedPeople])
        visibleTrackIds = {person.trackId for person in detectedPeople}
        self.recentEmbeddingsByTrackId = {trackId: recentEmbeddings for trackId, recentEmbeddings
                                          in self.recentEmbeddingsByTrackId.items() if trackId in visibleTrackIds}
        distances = {}
        for person, embedding in zip(comparedPeople, embeddings):
            recentEmbeddings = self.recentEmbeddingsByTrackId.setdefault(person.trackId,
                                                                         deque(maxlen=self.settings.averageViews))
            recentEmbeddings.append(embedding)
            distances[person.trackId] = self.CalculateDistanceToLeader(AverageEmbeddings(recentEmbeddings))
            self.distanceByTrackId[person.trackId] = distances[person.trackId]
            self.distanceSecondsByTrackId[person.trackId] = nowSeconds
        self.RecordSearchWins(distances, visibleTrackIds, nowSeconds)
        confirmedDistances = {trackId: distances.get(trackId, math.inf)
                              for trackId, winHistory in self.winHistories.items()
                              if sum(winHistory) >= self.settings.reidConfirm}
        if not confirmedDistances:
            return NO_EVENT
        winnerTrackId = min(confirmedDistances, key=confirmedDistances.get)
        self.state = FOLLOWING_STATE
        self.followedTrackId = winnerTrackId
        self.visibleLeaderTrackId = winnerTrackId
        self.lastVerifySeconds = nowSeconds
        self.lastSeenSeconds = nowSeconds
        self.lastStrongMatchSeconds = nowSeconds
        self.badCheckCount = 0
        self.absentSinceSeconds = None
        self.searchStartSeconds = None
        self.winHistories.clear()
        self.recentEmbeddingsByTrackId.clear()
        return FOUND_EVENT

    def RankCandidate(self, person, nowSeconds):
        distanceSeconds = self.distanceSecondsByTrackId.get(person.trackId)
        promising = (distanceSeconds is not None and self.distanceByTrackId[person.trackId] < self.settings.reidAccept
                     and nowSeconds - distanceSeconds < self.settings.competitorAge)
        comparedSeconds = -math.inf if distanceSeconds is None else distanceSeconds
        return (not promising, comparedSeconds, -CalculateArea(person))

    def RecordSearchWins(self, distances, visibleTrackIds, nowSeconds):
        self.winHistories = {trackId: winHistory for trackId, winHistory in self.winHistories.items()
                             if trackId in visibleTrackIds}
        for trackId, distance in distances.items():
            competitorDistances = [self.distanceByTrackId[otherTrackId] for otherTrackId in self.distanceByTrackId
                                   if otherTrackId != trackId and otherTrackId in visibleTrackIds
                                   and nowSeconds - self.distanceSecondsByTrackId[otherTrackId]
                                   < self.settings.competitorAge]
            leadOverCompetitors = min(competitorDistances, default=math.inf) - distance
            isWin = distance < self.settings.reidAccept and leadOverCompetitors >= self.settings.reidMargin
            self.winHistories.setdefault(trackId, deque(maxlen=self.settings.reidConfirmWindow)).append(isWin)

    def LoseLeader(self, reason, nowSeconds):
        self.state = SEARCHING_STATE
        self.followedTrackId = NO_TRACK_ID
        self.badCheckCount = 0
        self.absentSinceSeconds = None
        self.searchStartSeconds = nowSeconds
        self.lastStrongMatchSeconds = -math.inf
        self.winHistories.clear()
        self.recentEmbeddingsByTrackId.clear()
        self.eventReason = reason
        return LOST_EVENT

    def CalculateDistanceToLeader(self, embedding):
        storedEmbeddings = [embedding for groupEmbeddings in self.viewEmbeddings.values() for embedding in groupEmbeddings]
        if not storedEmbeddings:
            return math.inf
        return float(CalculateDistances(embedding, np.stack(storedEmbeddings)).min())

    def IsOverlapped(self, person, detectedPeople):
        return any(other.trackId != person.trackId and CalculateIou(person, other) > self.settings.crossingIou
                   for other in detectedPeople)

    def IsTouched(self, person, detectedPeople):
        return any(other.trackId != person.trackId and CalculateIou(person, other) > 0.0 for other in detectedPeople)

    def FindPerson(self, detectedPeople, trackId):
        return next((person for person in detectedPeople if person.trackId == trackId), None)

    def RebindEnrollment(self, detectedPeople):
        nearestPerson = max(detectedPeople, key=lambda person: CalculateIou(person, self.enrollLastPerson),
                            default=None)
        if nearestPerson is None or CalculateIou(nearestPerson, self.enrollLastPerson) < self.settings.enrollRebindIou:
            return None
        self.enrollTrackId = nearestPerson.trackId
        self.enrollmentRebound = True
        return nearestPerson


def CalculateDistances(embedding, storedEmbeddings):
    return np.linalg.norm(storedEmbeddings - embedding, axis=1)


def AverageEmbeddings(embeddings):
    averageEmbedding = np.mean(np.stack(embeddings), axis=0)
    return averageEmbedding / max(float(np.linalg.norm(averageEmbedding)), 1e-9)


def CalculateArea(person):
    return max(0.0, person.right - person.left) * max(0.0, person.bottom - person.top)


def CalculateIou(personA, personB):
    overlapWidth = min(personA.right, personB.right) - max(personA.left, personB.left)
    overlapHeight = min(personA.bottom, personB.bottom) - max(personA.top, personB.top)
    if overlapWidth <= 0 or overlapHeight <= 0:
        return 0.0
    overlapArea = overlapWidth * overlapHeight
    return overlapArea / (CalculateArea(personA) + CalculateArea(personB) - overlapArea)
