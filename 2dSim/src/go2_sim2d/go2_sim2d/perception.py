"""Fake perception: what the perception stack would report, made from ground truth.

No ROS in here, so it can be tested on its own; fake_perception.py is the node.

Each object is checked against two sensor fields around base_link: the lidar
(wide, geometry only) and the camera (narrow, classifies). An object is seen
by a sensor if its centre or the closest point of its outline lies in that
sensor's field and nothing else blocks the straight line to that point. Seen
objects are reported in polar coordinates (range, bearing) with Gaussian noise.
Their velocity over the ground is estimated with noise and a first-order filter.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from go2_sim2d.geometry import line_of_sight, nearest_point, ray_cast, wrap

CLASS_UNKNOWN, CLASS_OBJECT, CLASS_PERSON, CLASS_LEADER = 0, 1, 2, 3
SENSOR_LIDAR, SENSOR_CAMERA = 1, 2

# A velocity filter that has not been updated for this long starts over.
FILTER_RESET_S = 1.0


@dataclass
class Params:
    """Every parameter of the fake perception (also the node's ROS parameters)."""

    lidar_fov_deg: float = 360.0
    lidar_range_min: float = 0.1
    lidar_range_max: float = 15.0
    # Range std = a + b * range (m); bearing std in degrees.
    lidar_range_std: list = field(default_factory=lambda: [0.02, 0.01])
    lidar_bearing_std_deg: float = 1.0
    camera_fov_deg: float = 90.0
    camera_range_min: float = 0.3
    camera_range_max: float = 8.0
    camera_range_std: list = field(default_factory=lambda: [0.05, 0.05])
    camera_bearing_std_deg: float = 0.5
    camera_confidence: float = 0.9
    # Std of the velocity estimate = a + b * speed (m/s), per axis.
    velocity_std: list = field(default_factory=lambda: [0.05, 0.05])
    # Weight of a new velocity measurement in the filter (1 = no filtering).
    velocity_alpha: float = 0.5
    moving_threshold: float = 0.1
    radius_std: float = 0.05
    # Chance per object per frame that it is not reported.
    dropout_prob: float = 0.03
    # Chance per frame that the leader is reported as an ordinary person.
    leader_confusion_prob: float = 0.0
    occlusion: bool = True
    # Scales every noise std; 0 reports exact ground truth.
    noise_scale: float = 1.0
    scan_rays: int = 720


@dataclass
class Entity:
    """One thing in the world, as ground truth."""

    id: int
    kind: str                 # 'static', 'mover' or 'person'
    shape: object             # geometry.Circle or geometry.Box, in odom
    leader: bool = False
    velocity: tuple = (0.0, 0.0)  # over the ground, odom frame


@dataclass
class Detection:
    id: int
    class_id: int
    confidence: float
    sensors: int
    range: float
    bearing: float
    range_std: float
    bearing_std: float
    nearest_range: float
    nearest_bearing: float
    radius: float
    velocity: tuple           # (vx, vy) in base_link
    velocity_std: float
    moving: bool


class VelocityTracker:
    """Velocity of each entity by finite differences of its ground-truth position."""

    def __init__(self):
        self._last = {}
        self._velocity = {}

    def update(self, key, x, y, t):
        last = self._last.get(key)
        if last is not None and t > last[2]:
            dt = t - last[2]
            self._velocity[key] = ((x - last[0]) / dt, (y - last[1]) / dt)
        self._last[key] = (x, y, t)
        return self.get(key)

    def get(self, key):
        return self._velocity.get(key, (0.0, 0.0))


def _in_field(r, bearing, fov_deg, r_min, r_max):
    return r_min <= r <= r_max and abs(wrap(bearing)) <= math.radians(fov_deg) / 2 + 1e-9


class FakePerception:

    def __init__(self, params=None, seed=None):
        self.p = params or Params()
        self.rng = np.random.default_rng(seed)
        # id -> (vx, vy in odom, time of last update)
        self._filters = {}

    def _noise(self, std):
        std *= self.p.noise_scale
        return float(self.rng.normal(0.0, std)) if std > 0 else 0.0

    def _polar(self, robot, x, y):
        rx, ry, ryaw = robot
        return math.hypot(x - rx, y - ry), wrap(math.atan2(y - ry, x - rx) - ryaw)

    def _sees(self, robot, shapes, index, points, fov_deg, r_min, r_max):
        """True if one of points (centre, nearest) of shapes[index] is in the field and in sight."""
        for x, y, r, b in points:
            if not _in_field(r, b, fov_deg, r_min, r_max):
                continue
            if not self.p.occlusion or line_of_sight(robot[0], robot[1], x, y, shapes, {index}):
                return True
        return False

    def detect(self, robot, entities, t):
        """Detections of entities from robot pose (x, y, yaw) in odom at time t (s)."""
        p = self.p
        shapes = [e.shape for e in entities]
        detections = []
        for index, e in enumerate(entities):
            cx, cy = e.shape.cx, e.shape.cy
            nx, ny = nearest_point(e.shape, robot[0], robot[1])
            r_c, b_c = self._polar(robot, cx, cy)
            r_n, b_n = self._polar(robot, nx, ny)
            points = ((cx, cy, r_c, b_c), (nx, ny, r_n, b_n))

            lidar = self._sees(robot, shapes, index, points,
                               p.lidar_fov_deg, p.lidar_range_min, p.lidar_range_max)
            camera = self._sees(robot, shapes, index, points,
                                p.camera_fov_deg, p.camera_range_min, p.camera_range_max)
            if not (lidar or camera):
                continue
            if self.rng.random() < p.dropout_prob:
                continue

            # Lidar measures range better, the camera bearing.
            ra, rb = p.lidar_range_std if lidar else p.camera_range_std
            bearing_std = math.radians(
                p.camera_bearing_std_deg if camera else p.lidar_bearing_std_deg)

            if camera and e.kind == 'person':
                class_id, confidence = CLASS_PERSON, p.camera_confidence
                if e.leader and self.rng.random() >= p.leader_confusion_prob:
                    class_id = CLASS_LEADER
            elif camera:
                class_id, confidence = CLASS_OBJECT, p.camera_confidence
            else:
                class_id, confidence = CLASS_UNKNOWN, 0.0

            (vx, vy), v_std = self._velocity(e, robot[2], t)
            detections.append(Detection(
                id=e.id,
                class_id=class_id,
                confidence=confidence,
                sensors=(SENSOR_LIDAR if lidar else 0) | (SENSOR_CAMERA if camera else 0),
                range=max(0.0, r_c + self._noise(ra + rb * r_c)),
                bearing=wrap(b_c + self._noise(bearing_std)),
                range_std=(ra + rb * r_c) * p.noise_scale,
                bearing_std=bearing_std * p.noise_scale,
                nearest_range=max(0.0, r_n + self._noise(ra + rb * r_n)),
                nearest_bearing=wrap(b_n + self._noise(bearing_std)),
                radius=max(0.05, e.shape.radius + self._noise(p.radius_std)),
                velocity=(vx, vy),
                velocity_std=v_std,
                moving=math.hypot(vx, vy) > p.moving_threshold,
            ))
        return detections

    def _velocity(self, e, robot_yaw, t):
        """Filtered noisy ground velocity of e, rotated into base_link, and its std."""
        p = self.p
        vx, vy = e.velocity
        a, b = p.velocity_std
        std = a + b * math.hypot(vx, vy)
        mx, my = vx + self._noise(std), vy + self._noise(std)
        last = self._filters.get(e.id)
        if last is None or t - last[2] > FILTER_RESET_S:
            fx, fy = mx, my
        else:
            fx = last[0] + p.velocity_alpha * (mx - last[0])
            fy = last[1] + p.velocity_alpha * (my - last[1])
        self._filters[e.id] = (fx, fy, t)
        c, s = math.cos(robot_yaw), math.sin(robot_yaw)
        return (c * fx + s * fy, -s * fx + c * fy), std * p.noise_scale

    def scan(self, robot, shapes):
        """Lidar ranges over 360 deg from base_link, starting at -pi, with noise."""
        p = self.p
        n = p.scan_rays
        bearings = -math.pi + np.arange(n) * (2 * math.pi / n)
        ranges, _ = ray_cast(robot[0], robot[1], bearings + robot[2], shapes, p.lidar_range_max)
        hit = np.isfinite(ranges)
        a, b = p.lidar_range_std
        std = (a + b * ranges[hit]) * p.noise_scale
        ranges[hit] += self.rng.normal(0.0, 1.0, hit.sum()) * std
        ranges[hit & (ranges < p.lidar_range_min)] = math.inf
        return bearings, ranges
