import math
import statistics
from collections import deque
from dataclasses import dataclass, field

from builtin_interfaces.msg import Time
from follow_anywhere_interfaces.msg import EvaluationMetrics
from std_msgs.msg import Header

# Parameter defaults (set in the evaluation_node section of config/perception_params.yaml)
EVALUATION_SETTING_DEFAULTS = {
    'desiredDistance': 1.5,
    'nearCollisionDistance': 0.3,
    'switchJumpDistance': 1.5,
    'rateWindow': 1.0,
}

# Perception leader states (PerceptionLeader.status, PerceptionDebugFrame.leader_state)
LEADER_VISIBLE_STATUS = 1
FOLLOWING_STATE = 2
LOSS_STATES = (3, 4)

# Manual answers
ANSWER_UNANSWERED = -1
ANSWER_YES = 1

# Event types
COLLISION_EVENT = 'collision'
WRONG_LEADER_EVENT = 'wrong_leader'
MANUAL_EVENT_TYPES = (COLLISION_EVENT, WRONG_LEADER_EVENT)

# Metrics message
BASE_FRAME = 'base_link'
NAN = float('nan')


@dataclass
class RunEvent:
    runTime: float
    eventType: str
    description: str


@dataclass
class DistanceStatistics:
    averageDistance: float = NAN
    averageAbsoluteError: float = NAN
    rmse: float = NAN
    minimumDistance: float = NAN
    maximumDistance: float = NAN


@dataclass
class RateCounter:
    rateWindow: float
    onlyWhileActive: bool
    receiveTimes: deque = field(default_factory=deque)
    windowRates: list = field(default_factory=list)
    activeSinceSeconds: float = NAN
    lastReceiveSeconds: float = NAN

    def Count(self, receiveSeconds):
        gapEndsActivity = self.onlyWhileActive and receiveSeconds - self.lastReceiveSeconds > self.rateWindow
        if math.isnan(self.activeSinceSeconds) or gapEndsActivity:
            self.activeSinceSeconds = receiveSeconds
        self.lastReceiveSeconds = receiveSeconds
        self.receiveTimes.append(receiveSeconds)
        if receiveSeconds - self.activeSinceSeconds >= self.rateWindow:
            self.windowRates.append(self.CalculateRate(receiveSeconds))

    def CalculateRate(self, nowSeconds):
        while self.receiveTimes and self.receiveTimes[0] <= nowSeconds - self.rateWindow:
            self.receiveTimes.popleft()
        return len(self.receiveTimes) / self.rateWindow


class RunMetrics:
    def __init__(self, settings):
        self.settings = settings
        self.Reset()

    def Reset(self):
        self.runStartSeconds = NAN
        self.latestStampSeconds = NAN
        self.leaderVisible = False
        self.leaderDistance = NAN
        self.distances = []
        self.previousLeaderVisible = None
        self.previousLastKnownPoint = None
        self.leaderState = None
        self.previousStateStampSeconds = NAN
        self.followingSeconds = 0.0
        self.lossSeconds = 0.0
        self.lossStartSeconds = NAN
        self.lossCount = 0
        self.recoveryTimes = []
        self.suspectedSwitchCount = 0
        self.obstacleClearance = NAN
        self.minimumObstacleClearance = NAN
        self.nearCollisionCount = 0
        self.obstacleWasNear = False
        self.markCounts = {eventType: 0 for eventType in MANUAL_EVENT_TYPES}
        self.frameTimes = []
        self.imageAges = []
        self.cameraAgesAtFusion = []
        self.systemSamples = []
        self.perceptionRate = RateCounter(self.settings.rateWindow, onlyWhileActive=False)
        self.navigationRate = RateCounter(self.settings.rateWindow, onlyWhileActive=True)
        self.events = []

    def UpdateLeader(self, stampSeconds, status, closestPoint, lastKnownPoint, receiveSeconds):
        self.AdvanceRunClock(stampSeconds)
        self.perceptionRate.Count(receiveSeconds)
        self.leaderVisible = status == LEADER_VISIBLE_STATUS
        self.leaderDistance = math.hypot(closestPoint.x, closestPoint.y) if self.leaderVisible else NAN
        if math.isfinite(self.leaderDistance):
            self.distances.append(self.leaderDistance)
        if self.leaderVisible and self.previousLeaderVisible is False:
            self.CheckRecoveryJump(closestPoint)
        self.previousLeaderVisible = self.leaderVisible
        self.previousLastKnownPoint = lastKnownPoint

    def CheckRecoveryJump(self, closestPoint):
        jumpDistance = math.hypot(closestPoint.x - self.previousLastKnownPoint.x,
                                  closestPoint.y - self.previousLastKnownPoint.y)
        if jumpDistance > self.settings.switchJumpDistance:
            self.suspectedSwitchCount += 1
            self.AddEvent('suspected_switch', f'leader found again {jumpDistance:.2f} m from where it was last seen')

    def UpdateDebugFrame(self, stampSeconds, leaderState, secondsSinceSeen, frameTimeMs, imageAgeMs):
        self.AdvanceRunClock(stampSeconds)
        if math.isfinite(frameTimeMs):
            self.frameTimes.append(frameTimeMs)
        if math.isfinite(imageAgeMs):
            self.imageAges.append(imageAgeMs)
        self.AccumulateStateTime(stampSeconds)
        if self.leaderState == FOLLOWING_STATE and leaderState in LOSS_STATES:
            self.StartLoss(stampSeconds, secondsSinceSeen)
        if self.leaderState in LOSS_STATES and leaderState == FOLLOWING_STATE:
            self.FinishLoss(stampSeconds)
        self.leaderState = leaderState
        self.previousStateStampSeconds = stampSeconds

    def AccumulateStateTime(self, stampSeconds):
        elapsedSeconds = stampSeconds - self.previousStateStampSeconds
        if not math.isfinite(elapsedSeconds) or elapsedSeconds < 0.0:
            return
        if self.leaderState == FOLLOWING_STATE:
            self.followingSeconds += elapsedSeconds
        elif self.leaderState in LOSS_STATES and self.followingSeconds > 0.0:
            self.lossSeconds += elapsedSeconds

    def StartLoss(self, stampSeconds, secondsSinceSeen):
        graceSeconds = min(secondsSinceSeen, self.followingSeconds) if math.isfinite(secondsSinceSeen) else 0.0
        self.lossStartSeconds = stampSeconds - graceSeconds
        self.followingSeconds -= graceSeconds
        self.lossSeconds += graceSeconds
        self.lossCount += 1
        self.AddEvent('loss', f'leader lost (loss {self.lossCount}), last seen {graceSeconds:.2f} s earlier')

    def FinishLoss(self, stampSeconds):
        recoveryTime = stampSeconds - self.lossStartSeconds
        self.recoveryTimes.append(recoveryTime)
        self.lossStartSeconds = NAN
        self.AddEvent('recovery', f'leader found again after {recoveryTime:.2f} s')

    def UpdateObstacles(self, obstacleDistances):
        self.obstacleClearance = min((distance for distance in obstacleDistances if math.isfinite(distance)),
                                     default=NAN)
        if math.isnan(self.obstacleClearance):
            return
        if not self.obstacleClearance >= self.minimumObstacleClearance:
            self.minimumObstacleClearance = self.obstacleClearance
        obstacleIsNear = self.obstacleClearance < self.settings.nearCollisionDistance
        if obstacleIsNear and not self.obstacleWasNear:
            self.nearCollisionCount += 1
            self.AddEvent('near_collision', f'obstacle {self.obstacleClearance:.2f} m from the robot')
        self.obstacleWasNear = obstacleIsNear

    def CountNavigationCommand(self, receiveSeconds):
        self.navigationRate.Count(receiveSeconds)

    def UpdateCameraAgeAtFusion(self, cameraAgeMs):
        if math.isfinite(cameraAgeMs) and cameraAgeMs >= 0.0:
            self.cameraAgesAtFusion.append(cameraAgeMs)

    def UpdateSystemUsage(self, systemSample):
        self.systemSamples.append(systemSample)

    def SummariseSystemUsage(self):
        summary = {}
        for valueName in ('cpuPercent', 'gpuPercent', 'powerTotalMw', 'networkReceiveMbps', 'networkSendMbps',
                          'loopbackMbps', 'diskWriteMbps'):
            values = [getattr(sample, valueName) for sample in self.systemSamples]
            summary[valueName] = DescribeValues(values)
        summary['ramUsedMbMax'] = max((sample.ramUsedMb for sample in self.systemSamples), default=NAN)
        summary['junctionTemperatureMax'] = MaximumOrNan(sample.junctionTemperature for sample in self.systemSamples)
        summary['nodes'] = {}
        for nodeName in (self.systemSamples[0].nodes if self.systemSamples else {}):
            nodeSamples = [sample.nodes[nodeName] for sample in self.systemSamples]
            summary['nodes'][nodeName] = {
                'cpuPercent': DescribeValues([nodeSample.cpuPercent for nodeSample in nodeSamples]),
                'memoryMbMax': MaximumOrNan(nodeSample.memoryMb for nodeSample in nodeSamples),
            }
        return summary

    def MarkEvent(self, eventType, note):
        if eventType not in MANUAL_EVENT_TYPES:
            return False
        self.markCounts[eventType] += 1
        self.AddEvent(eventType, note or 'marked in the web viewer')
        return True

    def AdvanceRunClock(self, stampSeconds):
        if math.isnan(self.runStartSeconds):
            self.runStartSeconds = stampSeconds
        if not stampSeconds <= self.latestStampSeconds:
            self.latestStampSeconds = stampSeconds

    def CalculateRunTime(self):
        runTime = self.latestStampSeconds - self.runStartSeconds
        return runTime if math.isfinite(runTime) else 0.0

    def AddEvent(self, eventType, description):
        self.events.append(RunEvent(runTime=round(self.CalculateRunTime(), 3), eventType=eventType,
                                    description=description))

    def CalculateDistanceStatistics(self):
        if not self.distances:
            return DistanceStatistics()
        errors = [distance - self.settings.desiredDistance for distance in self.distances]
        return DistanceStatistics(
            averageDistance=statistics.fmean(self.distances),
            averageAbsoluteError=statistics.fmean(abs(error) for error in errors),
            rmse=math.sqrt(statistics.fmean(error * error for error in errors)),
            minimumDistance=min(self.distances),
            maximumDistance=max(self.distances))

    def CalculateLossTimeRatio(self):
        trackedSeconds = self.followingSeconds + self.lossSeconds
        return self.lossSeconds / trackedSeconds if trackedSeconds > 0.0 else NAN

    def BuildMetrics(self, nowReceiveSeconds):
        distanceStatistics = self.CalculateDistanceStatistics()
        currentLossDuration = 0.0
        if math.isfinite(self.lossStartSeconds):
            currentLossDuration = self.latestStampSeconds - self.lossStartSeconds
        return EvaluationMetrics(
            header=Header(stamp=ConvertSecondsToStamp(self.latestStampSeconds), frame_id=BASE_FRAME),
            run_time=self.CalculateRunTime(),
            leader_visible=self.leaderVisible,
            leader_distance=self.leaderDistance,
            distance_error=self.leaderDistance - self.settings.desiredDistance,
            average_distance=distanceStatistics.averageDistance,
            average_absolute_error=distanceStatistics.averageAbsoluteError,
            distance_rmse=distanceStatistics.rmse,
            minimum_distance=distanceStatistics.minimumDistance,
            maximum_distance=distanceStatistics.maximumDistance,
            loss_count=self.lossCount,
            loss_time_ratio=self.CalculateLossTimeRatio(),
            current_loss_duration=currentLossDuration,
            recovery_count=len(self.recoveryTimes),
            average_recovery_time=MeanOrNan(self.recoveryTimes),
            last_recovery_time=self.recoveryTimes[-1] if self.recoveryTimes else NAN,
            suspected_switch_count=self.suspectedSwitchCount,
            obstacle_clearance=self.obstacleClearance,
            minimum_obstacle_clearance=self.minimumObstacleClearance,
            near_collision_count=self.nearCollisionCount,
            collision_mark_count=self.markCounts[COLLISION_EVENT],
            wrong_leader_mark_count=self.markCounts[WRONG_LEADER_EVENT],
            perception_rate=self.perceptionRate.CalculateRate(nowReceiveSeconds),
            navigation_rate=self.navigationRate.CalculateRate(nowReceiveSeconds),
            frame_time_ms=self.frameTimes[-1] if self.frameTimes else NAN,
            image_age_ms=self.imageAges[-1] if self.imageAges else NAN,
            camera_age_at_fusion_ms=self.cameraAgesAtFusion[-1] if self.cameraAgesAtFusion else NAN)

    def BuildReport(self, finishRequest):
        distanceStatistics = self.CalculateDistanceStatistics()
        leaderInSightAtEnd = ResolveAnswer(finishRequest.leader_in_sight_at_end, self.leaderState == FOLLOWING_STATE)
        collision = ResolveAnswer(finishRequest.collision, self.markCounts[COLLISION_EVENT] > 0 or None)
        wrongLeader = ResolveAnswer(finishRequest.wrong_leader, self.markCounts[WRONG_LEADER_EVENT] > 0 or None)
        return {
            'runDuration': self.CalculateRunTime(),
            'parameters': vars(self.settings).copy(),
            'manual': {
                'collision': finishRequest.collision,
                'wrongLeader': finishRequest.wrong_leader,
                'leaderInSightAtEnd': finishRequest.leader_in_sight_at_end,
                'notes': finishRequest.notes,
                'collisionMarks': self.markCounts[COLLISION_EVENT],
                'wrongLeaderMarks': self.markCounts[WRONG_LEADER_EVENT],
            },
            'followSuccess': {
                'success': None if collision is None else leaderInSightAtEnd and not collision,
                'leaderInSightAtEnd': leaderInSightAtEnd,
                'collision': collision,
            },
            'leaderLossTimeRatio': self.CalculateLossTimeRatio(),
            'collision': {
                'collision': collision,
                'minimumObstacleClearance': self.minimumObstacleClearance,
                'nearCollisionCount': self.nearCollisionCount,
            },
            'distance': {
                'desiredDistance': self.settings.desiredDistance,
                'averageDistance': distanceStatistics.averageDistance,
                'averageError': distanceStatistics.averageDistance - self.settings.desiredDistance,
                'averageAbsoluteError': distanceStatistics.averageAbsoluteError,
                'rmse': distanceStatistics.rmse,
                'minimumDistance': distanceStatistics.minimumDistance,
                'maximumDistance': distanceStatistics.maximumDistance,
            },
            'recovery': {
                'lossCount': self.lossCount,
                'recoveryCount': len(self.recoveryTimes),
                'recoverySuccessRate': len(self.recoveryTimes) / self.lossCount if self.lossCount else NAN,
                'averageRecoveryTime': MeanOrNan(self.recoveryTimes),
                'medianRecoveryTime': statistics.median(self.recoveryTimes) if self.recoveryTimes else NAN,
                'maximumRecoveryTime': max(self.recoveryTimes, default=NAN),
            },
            'wrongLeader': {
                'wrongLeader': wrongLeader,
                'suspectedSwitchCount': self.suspectedSwitchCount,
            },
            'runtime': {
                'perceptionRateMean': MeanOrNan(self.perceptionRate.windowRates),
                'perceptionRateWorst': min(self.perceptionRate.windowRates, default=NAN),
                'navigationRateMean': MeanOrNan(self.navigationRate.windowRates),
                'navigationRateWorst': min(self.navigationRate.windowRates, default=NAN),
                'frameTimeMsMean': MeanOrNan(self.frameTimes),
                'frameTimeMsP95': CalculatePercentile(self.frameTimes, 95),
                'imageAgeMs': DescribeValues(self.imageAges),
                'cameraAgeAtFusionMs': DescribeValues(self.cameraAgesAtFusion),
            },
            'systemUsage': self.SummariseSystemUsage(),
            'events': [vars(event) for event in self.events],
        }


def ConvertStampToSeconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def ConvertSecondsToStamp(seconds):
    wholeSeconds = math.floor(seconds)
    return Time(sec=int(wholeSeconds), nanosec=int((seconds - wholeSeconds) * 1e9))


def ResolveAnswer(manualAnswer, automaticValue):
    if manualAnswer == ANSWER_UNANSWERED:
        return automaticValue
    return manualAnswer == ANSWER_YES


def MeanOrNan(values):
    return statistics.fmean(values) if values else NAN


def CalculatePercentile(values, percentile):
    if not values:
        return NAN
    sortedValues = sorted(values)
    return sortedValues[min(len(sortedValues) - 1, int(len(sortedValues) * percentile / 100))]


def DescribeValues(values):
    measuredValues = [value for value in values if math.isfinite(value)]
    return {'mean': MeanOrNan(measuredValues), 'p50': CalculatePercentile(measuredValues, 50),
            'p95': CalculatePercentile(measuredValues, 95), 'max': max(measuredValues, default=NAN)}


def MaximumOrNan(values):
    return max((value for value in values if math.isfinite(value)), default=NAN)
