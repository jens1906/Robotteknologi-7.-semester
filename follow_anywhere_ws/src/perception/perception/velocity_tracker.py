"""Positions, velocities and their uncertainty of tracked people and the leader over time. Plain python, no ROS.

- People are filtered in the odom frame (fixed to the floor), so a person standing still has velocity 0 even while
  the Go2 walks or turns. The robot's pose in odom comes from /utlidar/robot_odom (perception_node); before any
  odometry arrived the robot counts as standing still (pose 0, no uncertainty).
- The robot's pose gets an uncertainty when odometry goes silent: it may have moved robotMaxSpeed and turned
  robotMaxYawRate per silent second. That uncertainty is added to every measurement (sideways it grows with the
  distance, turning moves far people more), so the filters report larger uncertainties by themselves.
- One kalman_filter per tracker id. A new id gets a new filter (velocity unknown, unknownVelocitySigma); a filter
  without a measurement for trackTimeout is dropped. A measurement farther than jumpDistance from the prediction
  restarts the filter (swapped track ids, wrong re-identification) instead of making a huge speed.
- One leader filter kept across track id changes (re-identification says it is the same person); restarted after
  the leader was gone longer than graceTime. While the leader is not measured it is only predicted: the velocity
  of the last measurement is held, the uncertainty grows.
- Standing still: a speed below minimumSpeed or below STANDING_SIGMAS * its uncertainty is reported as 0, with the
  uncertainty kept (a new person is "0 +- unknownVelocitySigma", never "certainly standing still").
- ClockOffsetEstimator: the Go2 stamps odometry with its own clock (measured ~38 s off from the PC, constant within
  +-20 ms; the data itself is late < 0.06 s). Offset = smallest (arrival - stamp) over clockOffsetWindow, so the Go2's
  evenly spaced stamps are kept without network jitter. Forgotten when odometry restarts.
"""

import math
from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from perception.kalman_filter import CreateFilter, PredictFilter, ReadPositionSigma, ReadVelocitySigma, UpdateFilter

# Parameter defaults (set in config/perception_params.yaml)
VELOCITY_SETTING_DEFAULTS = {
    'kalmanProcessNoise': 1.0,
    'kalmanMeasurementNoise': 0.1,
    'trackTimeout': 1.0,
    'unknownVelocitySigma': 1.5,
    'minimumSpeed': 0.15,
    'jumpDistance': 1.0,
    'minimumPositionUncertainty': 0.05,
    'odometryTimeout': 0.5,
    'clockOffsetWindow': 5.0,
    'robotMaxSpeed': 0.6,
    'robotMaxYawRate': 1.6,
}

# Standing still: speeds below this many sigmas are reported as 0
STANDING_SIGMAS = 2.0


@dataclass
class RobotPose:
    positionX: float
    positionY: float
    heading: float
    positionSigma: float
    headingSigma: float


@dataclass
class PersonMeasurement:
    positionX: float
    positionY: float
    covariance: np.ndarray


@dataclass
class MotionEstimate:
    positionX: float
    positionY: float
    velocityX: float
    velocityY: float
    positionSigma: float
    velocitySigma: float


@dataclass
class ClockSample:
    arrivalSeconds: float
    offsetSeconds: float


class VelocityTracker:
    def __init__(self, settings):
        self.settings = settings
        self.filtersByTrackId = {}
        self.leaderFilter = None
        self.leaderMeasuredSeconds = 0.0
        self.lastLeaderEstimate = None

    def UpdatePeople(self, measurementsByTrackId, imageSeconds):
        for trackId, measurement in measurementsByTrackId.items():
            self.filtersByTrackId[trackId] = self.FeedFilter(self.filtersByTrackId.get(trackId), measurement,
                                                             imageSeconds)
        expiredTrackIds = [trackId for trackId, personFilter in self.filtersByTrackId.items()
                           if imageSeconds - personFilter.lastSeconds > self.settings.trackTimeout]
        for trackId in expiredTrackIds:
            del self.filtersByTrackId[trackId]
        return {trackId: self.ReadEstimate(self.filtersByTrackId[trackId]) for trackId in measurementsByTrackId}

    def UpdateLeader(self, leaderMeasurement, imageSeconds, graceTime):
        if leaderMeasurement is None:
            if self.leaderFilter is None:
                return None
            PredictFilter(self.leaderFilter, imageSeconds, self.settings.kalmanProcessNoise)
            return replace(self.ReadEstimate(self.leaderFilter), velocityX=self.lastLeaderEstimate.velocityX,
                           velocityY=self.lastLeaderEstimate.velocityY)
        if imageSeconds - self.leaderMeasuredSeconds > graceTime:
            self.leaderFilter = None
        self.leaderFilter = self.FeedFilter(self.leaderFilter, leaderMeasurement, imageSeconds)
        self.leaderMeasuredSeconds = imageSeconds
        self.lastLeaderEstimate = self.ReadEstimate(self.leaderFilter)
        return self.lastLeaderEstimate

    def FeedFilter(self, personFilter, measurement, imageSeconds):
        if personFilter is not None:
            PredictFilter(personFilter, imageSeconds, self.settings.kalmanProcessNoise)
            jumpDistance = math.hypot(measurement.positionX - personFilter.state[0],
                                      measurement.positionY - personFilter.state[1])
            if jumpDistance <= self.settings.jumpDistance:
                UpdateFilter(personFilter, measurement.positionX, measurement.positionY, measurement.covariance)
                return personFilter
        return CreateFilter(measurement.positionX, measurement.positionY, measurement.covariance,
                            self.settings.unknownVelocitySigma, imageSeconds)

    def ReadEstimate(self, personFilter):
        velocityX = float(personFilter.state[2])
        velocityY = float(personFilter.state[3])
        velocitySigma = ReadVelocitySigma(personFilter)
        if math.hypot(velocityX, velocityY) < max(self.settings.minimumSpeed, STANDING_SIGMAS * velocitySigma):
            velocityX = 0.0
            velocityY = 0.0
        return MotionEstimate(float(personFilter.state[0]), float(personFilter.state[1]), velocityX, velocityY,
                              max(ReadPositionSigma(personFilter), self.settings.minimumPositionUncertainty),
                              velocitySigma)

    def ResetLeader(self):
        self.leaderFilter = None
        self.lastLeaderEstimate = None

    def ResetAll(self):
        self.filtersByTrackId = {}
        self.ResetLeader()


class ClockOffsetEstimator:
    def __init__(self, windowSeconds):
        self.windowSeconds = windowSeconds
        self.clockSamples = deque()
        self.offsetSeconds = math.nan

    def CorrectStamp(self, stampSeconds, arrivalSeconds):
        offsetSeconds = arrivalSeconds - stampSeconds
        while self.clockSamples and self.clockSamples[-1].offsetSeconds >= offsetSeconds:
            self.clockSamples.pop()
        self.clockSamples.append(ClockSample(arrivalSeconds, offsetSeconds))
        while self.clockSamples[0].arrivalSeconds < arrivalSeconds - self.windowSeconds:
            self.clockSamples.popleft()
        self.offsetSeconds = self.clockSamples[0].offsetSeconds
        return stampSeconds + self.offsetSeconds

    def Reset(self):
        self.clockSamples.clear()
        self.offsetSeconds = math.nan


def MakeRobotPose(positionX, positionY, heading, silentSeconds, settings):
    silentSeconds = max(silentSeconds, 0.0)
    return RobotPose(positionX, positionY, heading, settings.robotMaxSpeed * silentSeconds,
                     settings.robotMaxYawRate * silentSeconds)


def MakeStillRobotPose():
    return RobotPose(0.0, 0.0, 0.0, 0.0, 0.0)


def BuildMeasurement(personCircle, depthSigma, robotPose, settings):
    centerX, centerY = ConvertPointToOdom(robotPose, personCircle.centerX, personCircle.centerY)
    measurementSigma = max(depthSigma, settings.kalmanMeasurementNoise)
    covariance = (measurementSigma ** 2 + robotPose.positionSigma ** 2) * np.eye(2)
    distance = math.hypot(personCircle.centerX, personCircle.centerY)
    if distance > 0.0:
        sideways = np.array([-(centerY - robotPose.positionY), centerX - robotPose.positionX]) / distance
        covariance += (distance * robotPose.headingSigma) ** 2 * np.outer(sideways, sideways)
    return PersonMeasurement(centerX, centerY, covariance)


def ConvertPointToOdom(robotPose, pointX, pointY):
    cosHeading = math.cos(robotPose.heading)
    sinHeading = math.sin(robotPose.heading)
    return (robotPose.positionX + cosHeading * pointX - sinHeading * pointY,
            robotPose.positionY + sinHeading * pointX + cosHeading * pointY)


def ConvertPointToBase(robotPose, pointX, pointY):
    offsetX = pointX - robotPose.positionX
    offsetY = pointY - robotPose.positionY
    return RotateToBase(robotPose, offsetX, offsetY)


def RotateToBase(robotPose, vectorX, vectorY):
    cosHeading = math.cos(robotPose.heading)
    sinHeading = math.sin(robotPose.heading)
    return cosHeading * vectorX + sinHeading * vectorY, -sinHeading * vectorX + cosHeading * vectorY


def ConvertEstimateToBase(robotPose, motionEstimate):
    positionX, positionY = ConvertPointToBase(robotPose, motionEstimate.positionX, motionEstimate.positionY)
    velocityX, velocityY = RotateToBase(robotPose, motionEstimate.velocityX, motionEstimate.velocityY)
    return replace(motionEstimate, positionX=positionX, positionY=positionY, velocityX=velocityX, velocityY=velocityY)
