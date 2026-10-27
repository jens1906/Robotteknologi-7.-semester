"""The Kalman filter maths, separate so it can be tuned and tested on its own. Plain python (numpy), no ROS.

- Constant-velocity model: state [x, y, vx, vy] on the floor, in the odom frame (velocity_tracker converts).
  Between two frames the velocity is assumed to stay the same; any change (speeding up, stopping, turning) is
  random acceleration with accelerationSigma (m/s^2) and is corrected by the next measurement.
- CreateFilter: starts at the measured position, velocity 0 with velocitySigma (unknown until measured twice).
- PredictFilter: moves the state ahead to a time; the uncertainty grows.
- UpdateFilter: takes a measured position with its 2x2 covariance (depth noise + uncertainty of the robot's pose).
- ReadPositionSigma / ReadVelocitySigma: 1 sigma along the worst axis.
"""

import math
from dataclasses import dataclass

import numpy as np

# Filter shape
STATE_SIZE = 4
MEASUREMENT_MATRIX = np.hstack((np.eye(2), np.zeros((2, 2))))


@dataclass
class PersonFilter:
    state: np.ndarray
    covariance: np.ndarray
    lastSeconds: float


def CreateFilter(positionX, positionY, measurementCovariance, velocitySigma, seconds):
    covariance = np.zeros((STATE_SIZE, STATE_SIZE))
    covariance[:2, :2] = measurementCovariance
    covariance[2, 2] = velocitySigma ** 2
    covariance[3, 3] = velocitySigma ** 2
    return PersonFilter(np.array([positionX, positionY, 0.0, 0.0]), covariance, seconds)


def PredictFilter(personFilter, seconds, accelerationSigma):
    stepSeconds = seconds - personFilter.lastSeconds
    if stepSeconds <= 0.0:
        return
    transition = np.eye(STATE_SIZE)
    transition[0, 2] = stepSeconds
    transition[1, 3] = stepSeconds
    accelerationEffect = np.array([stepSeconds ** 2 / 2, stepSeconds])
    axisNoise = accelerationSigma ** 2 * np.outer(accelerationEffect, accelerationEffect)
    processNoise = np.zeros((STATE_SIZE, STATE_SIZE))
    processNoise[np.ix_([0, 2], [0, 2])] = axisNoise
    processNoise[np.ix_([1, 3], [1, 3])] = axisNoise
    personFilter.state = transition @ personFilter.state
    personFilter.covariance = transition @ personFilter.covariance @ transition.T + processNoise
    personFilter.lastSeconds = seconds


def UpdateFilter(personFilter, positionX, positionY, measurementCovariance):
    innovation = np.array([positionX, positionY]) - personFilter.state[:2]
    innovationCovariance = personFilter.covariance[:2, :2] + measurementCovariance
    gain = personFilter.covariance[:, :2] @ np.linalg.inv(innovationCovariance)
    personFilter.state = personFilter.state + gain @ innovation
    updatedCovariance = (np.eye(STATE_SIZE) - gain @ MEASUREMENT_MATRIX) @ personFilter.covariance
    personFilter.covariance = (updatedCovariance + updatedCovariance.T) / 2


def ReadPositionSigma(personFilter):
    return math.sqrt(max(np.linalg.eigvalsh(personFilter.covariance[:2, :2]).max(), 0.0))


def ReadVelocitySigma(personFilter):
    return math.sqrt(max(np.linalg.eigvalsh(personFilter.covariance[2:, 2:]).max(), 0.0))
