"""Who is the leader: enrolment, the stored views, finding them again, lost / last seen. Plain python, no ROS.

To include:
- States: IDLE, ENROLLING, FOLLOWING, SEARCHING, LOST (the values of leader_state in PerceptionDebugFrame).
- The leader is described by stored views in five groups: front, back, side left, side right and partly
  hidden (body_view.py decides the group from the keypoints). At most viewsPerGroup views per group.
  A view is a PersonDescription (reid_model.py): OSNet's 16x8 cells + which cells show the person.
- Comparing: only over the cells visible in both photos (shared area). Below minimumSharedArea the result is
  unknown (NaN): no evidence either way, never a low score. The distance to the leader is the smallest over
  all stored views.
- Enrolment (calibrate service): the leader turns around once; up to enrollViewsPerDirection clean views
  per facing direction, enrollGap apart, nobody overlapping. Done when all four directions are filled,
  or after enrollDuration (or when the person leaves) when enrollMinimumDirections are filled. These are the anchors and are never
  dropped. Enrolment survives a tracker id change at the same place.
- While FOLLOWING: locked to the tracker id; the person is compared with the stored views every
  verifyPeriod (an OSNet check, so a swap of tracker ids is noticed). A view is learned only when we are
  sure who it is: a strong match (below reidAccept) at most trustTime ago on the same unbroken track,
  a clean photo, no other box touching the leader. A new view replaces the most similar learned view of
  its group, so the views stay varied. The lock is dropped when:
  - the person looks different (reidDrop, loseAfterChecks checks in a row),
  - the followed track is farther than swapGateSigma from where the leader is predicted to be (the tracker
    handed the leader's id to somebody else: never followed, not even for one frame),
  - somebody overlaps them (crossingIou) and they are too hidden (shared area unknown), or not clearly them (not
    below the accept distance) loseAfterChecks checks in a row; while overlapped the leader is checked every frame,
  - they are missing longer than graceTime (flicker).
- Negative bank: while FOLLOWING with a recent strong match, every negativePeriod the other clean people not
  touching the leader are described and remembered (they are surely not the leader), up to negativeBankSize,
  oldest dropped, near-duplicates (noveltyDistance) skipped. Not saved: the people change between sessions.
- While SEARCHING: candidates are clean people nobody overlaps (measured on the shuffle bag: an overlapped crop is
  no evidence, the leader's median distance was 0.56 while overlapped, other people's lowest 0.53).
  - Motion gate (SEARCHING only, the leader's Kalman prediction from the node, in sigmas): people farther than
    motionGateSigma are not candidates; people inside it are compared every frame and accepted below gatedAccept
    (the place already speaks for them). People without a gate distance (no position, or no prediction yet) are
    compared every searchPeriod, searchCandidates at a time (promising ones first, then the ones compared longest
    ago) and accepted below reidAccept.
  - A candidate's distance is the mean of its last averageViews known distances.
  - Accept only below the accept distance, with reidMargin over every other visible person and over the negative
    bank, confirmed reidConfirm times in the last reidConfirmWindow comparisons. Unknown comparisons are not counted.
  - The track that was followed before the loss gets no free pass: it has to win like anybody else.
- LOST (searched longer than lostTime): the motion gate is ignored, everybody is searched the slow way.
- Precision first: "not identified" is fine, the wrong person is the worst failure.
- Keeps the time the leader was last identified.
- visibleLeaderTrackId is the leader's track id only in frames where they are identified, else -1.
- reidMargin (debug): the identified or best candidate's lead over the nearest other person or negative, NaN when
  nothing was compared this frame.
- Save the anchors with their group to a file (numpy, no pickle, tagged with the reid model) and load
  them at start, so the leader survives a restart.
- Descriptions and comparisons come in through callables, so this file needs no model.
"""

import math
import os
from collections import deque
from types import SimpleNamespace

import numpy as np

from perception.body_view import (BODY_VIEW_SETTING_DEFAULTS, FACING_GROUPS, UNKNOWN_GROUP, VIEW_GROUPS,
                                  FindViewGroup, IsCleanView)
from perception.reid_model import PersonDescription, StackDescriptions

# Parameter defaults (set in config/perception_params.yaml, the origin of each value is written there)
LEADER_SETTING_DEFAULTS = {
    'reidAccept': 0.45,
    'gatedAccept': 0.7,
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
    'minimumSharedArea': 0.4,
    'motionGateSigma': 3.0,
    'swapGateSigma': 4.0,
    'negativeBankSize': 40,
    'negativePeriod': 1.0,
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
    def __init__(self, compareDescriptions, leaderSettings=None):
        self.settings = SimpleNamespace(**{**LEADER_SETTING_DEFAULTS, **(leaderSettings or {})})
        self.compareDescriptions = compareDescriptions
        self.distanceByTrackId = {}
        self.sharedAreaByTrackId = {}
        self.distanceSecondsByTrackId = {}
        self.Reset()

    def Reset(self):
        self.state = IDLE_STATE
        self.visibleLeaderTrackId = NO_TRACK_ID
        self.eventReason = ''
        self.reidMargin = math.nan
        self.followedTrackId = NO_TRACK_ID
        self.viewDescriptions = {viewGroup: [] for viewGroup in VIEW_GROUPS}
        self.anchorCounts = {viewGroup: 0 for viewGroup in VIEW_GROUPS}
        self.viewStack = StackDescriptions([])
        self.negativeDescriptions = deque(maxlen=self.settings.negativeBankSize)
        self.negativeStack = StackDescriptions([])
        self.enrollTrackId = NO_TRACK_ID
        self.enrollLastPerson = None
        self.enrollmentRebound = False
        self.enrollStartSeconds = 0.0
        self.enrollLastSeenSeconds = 0.0
        self.lastSampleSeconds = 0.0
        self.lastVerifySeconds = 0.0
        self.lastSearchSeconds = 0.0
        self.lastNegativeSeconds = -math.inf
        self.lastStrongMatchSeconds = -math.inf
        self.searchStartSeconds = None
        self.absentSinceSeconds = None
        self.lastSeenSeconds = 0.0
        self.badCheckCount = 0
        self.winHistories = {}
        self.recentDistancesByTrackId = {}
        self.distanceByTrackId.clear()
        self.sharedAreaByTrackId.clear()
        self.distanceSecondsByTrackId.clear()

    def Enroll(self, trackId, nowSeconds):
        self.Reset()
        self.state = ENROLLING_STATE
        self.enrollTrackId = trackId
        self.enrollStartSeconds = nowSeconds
        self.enrollLastSeenSeconds = nowSeconds

    def Update(self, detectedPeople, describePeople, gateDistances, nowSeconds, imageWidth):
        self.eventReason = ''
        self.reidMargin = math.nan
        self.visibleLeaderTrackId = NO_TRACK_ID
        if self.state == ENROLLING_STATE:
            return self.UpdateEnrolling(detectedPeople, describePeople, nowSeconds, imageWidth)
        if self.state == FOLLOWING_STATE:
            return self.UpdateFollowing(detectedPeople, describePeople, gateDistances, nowSeconds, imageWidth)
        if self.state in (SEARCHING_STATE, LOST_STATE):
            return self.UpdateSearching(detectedPeople, describePeople, gateDistances, nowSeconds, imageWidth)
        return NO_EVENT

    def CountStoredViews(self):
        return {viewGroup: len(descriptions) for viewGroup, descriptions in self.viewDescriptions.items()}

    def SaveLeader(self, leaderFilePath, modelTag):
        anchorDescriptions = [description for viewGroup in VIEW_GROUPS
                              for description in self.viewDescriptions[viewGroup][:self.anchorCounts[viewGroup]]]
        anchorGroups = [viewGroup for viewGroup in VIEW_GROUPS for _ in range(self.anchorCounts[viewGroup])]
        anchorStack = StackDescriptions(anchorDescriptions)
        os.makedirs(os.path.dirname(leaderFilePath) or '.', exist_ok=True)
        np.savez(leaderFilePath, cellFeatures=anchorStack.cellFeatures, visibleCells=anchorStack.visibleCells,
                 viewGroups=np.array(anchorGroups), modelTag=modelTag)

    def LoadLeader(self, leaderFilePath, modelTag):
        try:
            savedLeader = np.load(leaderFilePath, allow_pickle=False)
            if str(savedLeader['modelTag']) != modelTag:
                return False
            cellFeatures = savedLeader['cellFeatures'].astype(np.float32)
            visibleCells = savedLeader['visibleCells'].astype(np.float32)
            viewGroups = savedLeader['viewGroups'].astype(int)
        except (OSError, KeyError, ValueError):
            return False
        if cellFeatures.ndim != 3 or not len(cellFeatures) == len(visibleCells) == len(viewGroups):
            return False
        self.Reset()
        for personCells, personVisibleCells, viewGroup in zip(cellFeatures, visibleCells, viewGroups):
            self.viewDescriptions[int(viewGroup)].append(PersonDescription(personCells, personVisibleCells))
            self.anchorCounts[int(viewGroup)] += 1
        self.RestackViews()
        self.state = SEARCHING_STATE
        return True

    def GetRecentDistance(self, trackId, nowSeconds):
        distanceSeconds = self.distanceSecondsByTrackId.get(trackId)
        if distanceSeconds is None or nowSeconds - distanceSeconds > self.settings.recentDistanceTime:
            return math.nan
        return self.distanceByTrackId[trackId]

    def GetRecentSharedArea(self, trackId, nowSeconds):
        distanceSeconds = self.distanceSecondsByTrackId.get(trackId)
        if distanceSeconds is None or nowSeconds - distanceSeconds > self.settings.recentDistanceTime:
            return math.nan
        return self.sharedAreaByTrackId[trackId]

    def UpdateEnrolling(self, detectedPeople, describePeople, nowSeconds, imageWidth):
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
        description = describePeople([person.trackId])[0]
        self.lastSampleSeconds = nowSeconds
        if self.enrollmentRebound and self.viewDescriptions[viewGroup]:
            groupDistance, _ = self.FindNearestDistance(description, StackDescriptions(self.viewDescriptions[viewGroup]))
            if groupDistance > self.settings.reidDrop:
                self.Reset()
                self.eventReason = 'person changed'
                return ENROLL_FAILED_EVENT
            self.enrollmentRebound = False
        self.viewDescriptions[viewGroup].append(description)
        self.anchorCounts[viewGroup] += 1
        self.RestackViews()
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

    def UpdateFollowing(self, detectedPeople, describePeople, gateDistances, nowSeconds, imageWidth):
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
        if gateDistances.get(person.trackId, math.nan) > self.settings.swapGateSigma:
            return self.LoseLeader('jump', nowSeconds)
        overlapped = self.IsOverlapped(person, detectedPeople)
        if returned:
            self.lastStrongMatchSeconds = -math.inf
        if not (returned or overlapped or nowSeconds - self.lastVerifySeconds >= self.settings.verifyPeriod):
            self.visibleLeaderTrackId = person.trackId
            return NO_EVENT
        recentlyMatched = nowSeconds - self.lastStrongMatchSeconds <= self.settings.trustTime
        negativePeople = self.PickNegativePeople(person, detectedPeople, nowSeconds, imageWidth) if recentlyMatched else []
        descriptions = describePeople([person.trackId] + [negativePerson.trackId for negativePerson in negativePeople])
        self.lastVerifySeconds = nowSeconds
        distance, sharedArea = self.FindNearestDistance(descriptions[0], self.viewStack)
        self.RememberDistance(person.trackId, distance, sharedArea, nowSeconds)
        if math.isnan(distance):
            if returned or overlapped:
                return self.LoseLeader('hidden', nowSeconds)
            self.visibleLeaderTrackId = person.trackId
            return NO_EVENT
        acceptDistance = self.FindAcceptDistance(gateDistances.get(person.trackId, math.nan))
        if returned and not distance < acceptDistance:
            return self.LoseLeader('not sure after a gap', nowSeconds)
        if distance > self.settings.reidDrop or (overlapped and not distance < acceptDistance):
            self.badCheckCount += 1
            if self.badCheckCount >= self.settings.loseAfterChecks:
                return self.LoseLeader('not sure while overlapped' if overlapped else 'looks different', nowSeconds)
        else:
            self.badCheckCount = 0
            self.AddNegatives(descriptions[1:], nowSeconds)
            self.LearnIfSureWhoItIs(person, detectedPeople, descriptions[0], distance, nowSeconds, imageWidth)
        self.reidMargin = self.FindLeadOverOthers(person.trackId, distance, descriptions[0], detectedPeople,
                                                  nowSeconds)
        self.visibleLeaderTrackId = person.trackId
        return NO_EVENT

    def PickNegativePeople(self, leaderPerson, detectedPeople, nowSeconds, imageWidth):
        if nowSeconds - self.lastNegativeSeconds < self.settings.negativePeriod:
            return []
        return [person for person in detectedPeople
                if person.trackId != leaderPerson.trackId and CalculateIou(person, leaderPerson) == 0.0
                and IsCleanView(person, imageWidth, self.settings)]

    def AddNegatives(self, negativeDescriptions, nowSeconds):
        if not negativeDescriptions:
            return
        self.lastNegativeSeconds = nowSeconds
        for description in negativeDescriptions:
            nearestDistance, _ = self.FindNearestDistance(description, self.negativeStack)
            if not nearestDistance < self.settings.noveltyDistance:
                self.negativeDescriptions.append(description)
                self.negativeStack = StackDescriptions(list(self.negativeDescriptions))

    def LearnIfSureWhoItIs(self, person, detectedPeople, description, distance, nowSeconds, imageWidth):
        recentlyMatched = nowSeconds - self.lastStrongMatchSeconds <= self.settings.trustTime
        if distance < self.settings.reidAccept:
            self.lastStrongMatchSeconds = nowSeconds
        if not recentlyMatched or self.IsTouched(person, detectedPeople) or not IsCleanView(person, imageWidth, self.settings):
            return
        viewGroup = FindViewGroup(person, self.settings)
        if viewGroup != UNKNOWN_GROUP:
            self.LearnView(description, viewGroup)

    def LearnView(self, description, viewGroup):
        groupDescriptions = self.viewDescriptions[viewGroup]
        if groupDescriptions:
            nearestDistance, _ = self.FindNearestDistance(description, StackDescriptions(groupDescriptions))
            if not nearestDistance >= self.settings.noveltyDistance:
                return
        if len(groupDescriptions) < self.settings.viewsPerGroup:
            groupDescriptions.append(description)
            self.RestackViews()
            return
        anchorCount = self.anchorCounts[viewGroup]
        learnedDescriptions = groupDescriptions[anchorCount:]
        if not learnedDescriptions:
            return
        learnedDistances, _ = self.compareDescriptions(description, StackDescriptions(learnedDescriptions),
                                                       self.settings.minimumSharedArea)
        if np.isnan(learnedDistances).all():
            return
        groupDescriptions[anchorCount + int(np.nanargmin(learnedDistances))] = description
        self.RestackViews()

    def UpdateSearching(self, detectedPeople, describePeople, gateDistances, nowSeconds, imageWidth):
        if self.searchStartSeconds is None:
            self.searchStartSeconds = nowSeconds
        searchedSeconds = nowSeconds - self.searchStartSeconds
        self.state = LOST_STATE if searchedSeconds > self.settings.lostTime else SEARCHING_STATE
        gatedPeople = []
        ungatedPeople = []
        for person in detectedPeople:
            if self.IsOverlapped(person, detectedPeople) or not IsCleanView(person, imageWidth, self.settings):
                continue
            gateDistance = gateDistances.get(person.trackId, math.nan) if self.state == SEARCHING_STATE else math.nan
            if gateDistance <= self.settings.motionGateSigma:
                gatedPeople.append(person)
            elif math.isnan(gateDistance):
                ungatedPeople.append(person)
        searchDue = nowSeconds - self.lastSearchSeconds >= self.settings.searchPeriod
        rankedPeople = sorted(ungatedPeople, key=lambda person: self.RankCandidate(person, nowSeconds))
        comparedPeople = gatedPeople + (rankedPeople[:self.settings.searchCandidates] if searchDue else [])
        if not comparedPeople:
            return NO_EVENT
        if searchDue:
            self.lastSearchSeconds = nowSeconds
        descriptions = describePeople([person.trackId for person in comparedPeople])
        visibleTrackIds = {person.trackId for person in detectedPeople}
        self.recentDistancesByTrackId = {trackId: recentDistances for trackId, recentDistances
                                         in self.recentDistancesByTrackId.items() if trackId in visibleTrackIds}
        distances = {}
        negativeDistances = {}
        acceptDistances = {person.trackId: self.FindAcceptDistance(gateDistances.get(person.trackId, math.nan))
                           for person in comparedPeople}
        for person, description in zip(comparedPeople, descriptions):
            distance, sharedArea = self.FindNearestDistance(description, self.viewStack)
            if math.isnan(distance):
                self.RememberDistance(person.trackId, distance, sharedArea, nowSeconds)
                continue
            recentDistances = self.recentDistancesByTrackId.setdefault(person.trackId,
                                                                       deque(maxlen=self.settings.averageViews))
            recentDistances.append(distance)
            distances[person.trackId] = float(np.mean(recentDistances))
            negativeDistances[person.trackId] = self.FindNegativeDistance(description)
            self.RememberDistance(person.trackId, distances[person.trackId], sharedArea, nowSeconds)
        self.RecordSearchWins(distances, negativeDistances, acceptDistances, visibleTrackIds, nowSeconds)
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
        self.recentDistancesByTrackId.clear()
        return FOUND_EVENT

    def RankCandidate(self, person, nowSeconds):
        distanceSeconds = self.distanceSecondsByTrackId.get(person.trackId)
        promising = (distanceSeconds is not None and self.distanceByTrackId[person.trackId] < self.settings.reidAccept
                     and nowSeconds - distanceSeconds < self.settings.competitorAge)
        comparedSeconds = -math.inf if distanceSeconds is None else distanceSeconds
        return (not promising, comparedSeconds, -CalculateArea(person))

    def RecordSearchWins(self, distances, negativeDistances, acceptDistances, visibleTrackIds, nowSeconds):
        self.winHistories = {trackId: winHistory for trackId, winHistory in self.winHistories.items()
                             if trackId in visibleTrackIds}
        bestLead = math.nan
        for trackId, distance in distances.items():
            competitorDistances = [self.distanceByTrackId[otherTrackId] for otherTrackId in self.distanceByTrackId
                                   if otherTrackId != trackId and otherTrackId in visibleTrackIds
                                   and nowSeconds - self.distanceSecondsByTrackId[otherTrackId]
                                   < self.settings.competitorAge]
            nearestOther = min([competitorDistance for competitorDistance in competitorDistances
                                if not math.isnan(competitorDistance)] + [negativeDistances[trackId]], default=math.inf)
            leadOverOthers = nearestOther - distance
            isWin = distance < acceptDistances[trackId] and leadOverOthers >= self.settings.reidMargin
            self.winHistories.setdefault(trackId, deque(maxlen=self.settings.reidConfirmWindow)).append(isWin)
            if distance == min(distances.values()) and math.isfinite(leadOverOthers):
                bestLead = leadOverOthers
        self.reidMargin = bestLead

    def FindAcceptDistance(self, gateDistance):
        if self.state != LOST_STATE and gateDistance <= self.settings.motionGateSigma:
            return self.settings.gatedAccept
        return self.settings.reidAccept

    def FindLeadOverOthers(self, leaderTrackId, distance, description, detectedPeople, nowSeconds):
        otherDistances = [self.GetRecentDistance(person.trackId, nowSeconds) for person in detectedPeople
                          if person.trackId != leaderTrackId]
        nearestOther = min([otherDistance for otherDistance in otherDistances if not math.isnan(otherDistance)]
                           + [self.FindNegativeDistance(description)], default=math.inf)
        return nearestOther - distance if math.isfinite(nearestOther) else math.nan

    def LoseLeader(self, reason, nowSeconds):
        self.state = SEARCHING_STATE
        self.followedTrackId = NO_TRACK_ID
        self.badCheckCount = 0
        self.absentSinceSeconds = None
        self.searchStartSeconds = nowSeconds
        self.lastStrongMatchSeconds = -math.inf
        self.winHistories.clear()
        self.recentDistancesByTrackId.clear()
        self.eventReason = reason
        return LOST_EVENT

    def FindNearestDistance(self, description, storedStack):
        if not len(storedStack.cellFeatures):
            return math.inf, math.nan
        distances, sharedAreas = self.compareDescriptions(description, storedStack, self.settings.minimumSharedArea)
        if np.isnan(distances).all():
            return math.nan, float(sharedAreas.max())
        nearestIndex = int(np.nanargmin(distances))
        return float(distances[nearestIndex]), float(sharedAreas[nearestIndex])

    def FindNegativeDistance(self, description):
        negativeDistance, _ = self.FindNearestDistance(description, self.negativeStack)
        return math.inf if math.isnan(negativeDistance) else negativeDistance

    def RememberDistance(self, trackId, distance, sharedArea, nowSeconds):
        self.distanceByTrackId[trackId] = distance
        self.sharedAreaByTrackId[trackId] = sharedArea
        self.distanceSecondsByTrackId[trackId] = nowSeconds

    def RestackViews(self):
        self.viewStack = StackDescriptions([description for descriptions in self.viewDescriptions.values()
                                            for description in descriptions])

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


def CalculateArea(person):
    return max(0.0, person.right - person.left) * max(0.0, person.bottom - person.top)


def CalculateIou(personA, personB):
    overlapWidth = min(personA.right, personB.right) - max(personA.left, personB.left)
    overlapHeight = min(personA.bottom, personB.bottom) - max(personA.top, personB.top)
    if overlapWidth <= 0 or overlapHeight <= 0:
        return 0.0
    overlapArea = overlapWidth * overlapHeight
    return overlapArea / (CalculateArea(personA) + CalculateArea(personB) - overlapArea)
