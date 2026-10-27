"""The points perception publishes for obstacles and the leader: people as circles on the floor. Creates no topics.

To include:
- A person is a circle on the floor. The depth measures the front of the torso, so the centre is that point plus
  the radius along the camera's view direction. Radius: half the box width, kept in personRadiusMin..personRadiusMax.
- closest point: the circle point nearest the robot (base_link origin).
- line of sight point: the circle point nearest the line camera -> leader (the line point itself when the circle
  crosses the line).
- position uncertainty: spread of the depth samples, at least minimumPositionUncertainty.
- Everything 2D in base_link (z = 0).
- Later: static obstacles from the lidar (/utlidar/cloud_base, ground removed, obstacleMaxRange), the depth image as
  fallback; points of a tracked person are not also a static obstacle.
"""

import math
from dataclasses import dataclass

from geometry_msgs.msg import Point

# Parameter defaults (set in config/perception_params.yaml)
OBSTACLE_SETTING_DEFAULTS = {
    'personRadiusMin': 0.15,
    'personRadiusMax': 0.4,
    'minimumPositionUncertainty': 0.05,
}

# Geometry
SMALLEST_LENGTH = 1e-6


@dataclass
class PersonCircle:
    centerX: float
    centerY: float
    radius: float


def BuildPersonCircle(personPosition, cameraPosition, settings):
    radius = min(max(personPosition.width / 2, settings.personRadiusMin), settings.personRadiusMax)
    viewX = personPosition.position.x - cameraPosition.x
    viewY = personPosition.position.y - cameraPosition.y
    viewLength = max(math.hypot(viewX, viewY), SMALLEST_LENGTH)
    return PersonCircle(personPosition.position.x + radius * viewX / viewLength,
                        personPosition.position.y + radius * viewY / viewLength, radius)


def FindClosestPointToRobot(personCircle):
    return FindCirclePointTowards(personCircle, 0.0, 0.0)


def FindClosestPointToLine(personCircle, lineStart, lineEnd):
    lineX = lineEnd.x - lineStart.x
    lineY = lineEnd.y - lineStart.y
    lineLengthSquared = lineX * lineX + lineY * lineY
    lineShare = 0.0
    if lineLengthSquared > SMALLEST_LENGTH:
        projectedShare = ((personCircle.centerX - lineStart.x) * lineX
                          + (personCircle.centerY - lineStart.y) * lineY) / lineLengthSquared
        lineShare = min(max(projectedShare, 0.0), 1.0)
    return FindCirclePointTowards(personCircle, lineStart.x + lineShare * lineX, lineStart.y + lineShare * lineY)


def FindCirclePointTowards(personCircle, targetX, targetY):
    offsetX = targetX - personCircle.centerX
    offsetY = targetY - personCircle.centerY
    targetDistance = math.hypot(offsetX, offsetY)
    if targetDistance <= personCircle.radius:
        return MakeFloorPoint(targetX, targetY)
    return MakeFloorPoint(personCircle.centerX + personCircle.radius * offsetX / targetDistance,
                          personCircle.centerY + personCircle.radius * offsetY / targetDistance)


def EstimatePositionUncertainty(personPosition, settings):
    return max(personPosition.uncertainty, settings.minimumPositionUncertainty)


def MakeFloorPoint(pointX, pointY):
    return Point(x=float(pointX), y=float(pointY), z=0.0)


def MakeUnknownPoint():
    return Point(x=math.nan, y=math.nan, z=math.nan)
