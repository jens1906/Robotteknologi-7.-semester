"""Fake perception: what the perception stack would report, made from ground truth.

No ROS in here, so it can be tested on its own; fake_perception.py is the node.

Each object is checked against two sensor fields: the lidar (wide, geometry
only) and the camera (narrow, classifies, around base_link). An object is seen
by a sensor if its centre or the closest point of its outline lies in that
sensor's field and nothing else blocks the straight line to that point. Seen
objects are reported in polar coordinates (range, bearing) with Gaussian noise.
Their velocity over the ground is estimated with noise and a first-order filter.

The lidar is the Go2's Unitree L1, measured from a rosbag of the real robot: it
sits lidar_x in front of base_link, upside down and pitched lidar_tilt_deg
forward, so its 90 deg dome looks down. Behind the robot it only sees below the
horizon, which limits the range there to floor-standing objects within
lidar_height / tan(tilt). Outside lidar_clear_fov_deg (seen from the sensor)
the legs and body block parts of the scan now and then.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from go2_sim2d.geometry import line_of_sight, nearest_point, ray_cast, wrap

CLASS_UNKNOWN, CLASS_OBJECT, CLASS_PERSON, CLASS_LEADER = 0, 1, 2, 3
SENSOR_LIDAR, SENSOR_CAMERA = 1, 2

# A velocity filter that has not been updated for this long starts over.
FILTER_RESET_S = 1.0
# Width of the scan sectors the legs block as a whole.
OCCLUSION_SECTOR_DEG = 10.0


@dataclass
class Params:
    """Every parameter of the fake perception (also the node's ROS parameters)."""

    lidar_fov_deg: float = 360.0
    lidar_range_min: float = 0.35
    lidar_range_max: float = 15.0
    # Range std = a + b * range (m); bearing std in degrees.
    lidar_range_std: list = field(default_factory=lambda: [0.02, 0.005])
    lidar_bearing_std_deg: float = 1.0
    # Mount: ahead of base_link (m), above the floor (m), dome pitched forward.
    lidar_x: float = 0.283
    lidar_height: float = 0.42
    lidar_tilt_deg: float = 15.0
    # Sector around the front, seen from the sensor, that the body never blocks;
    # outside it each sector is blocked with this chance per scan or detection.
    lidar_clear_fov_deg: float = 150.0
    lidar_self_occlusion_prob: float = 0.25
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


def lidar_limit(p, bearing):
    """Max lidar range (m) from the sensor at sensor bearing(s), for floor-standing objects.

    The tilted dome reaches tan(tilt) * cos(bearing) above the horizon, so behind
    the robot a ray only gets back down to the floor within height / (tan(tilt) * -cos).
    """
    c = -np.cos(bearing) * math.tan(math.radians(p.lidar_tilt_deg))
    with np.errstate(divide='ignore'):
        limit = np.where(c > 0, p.lidar_height / np.where(c > 0, c, 1.0), np.inf)
    return np.minimum(limit, p.lidar_range_max)


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

    def _sensor(self, robot):
        """Pose (x, y, yaw) of the lidar in odom."""
        rx, ry, ryaw = robot
        lx = self.p.lidar_x
        return rx + lx * math.cos(ryaw), ry + lx * math.sin(ryaw), ryaw

    def _lidar_in_field(self, sensor, x, y, r):
        """Mask of points (x, y) at base_link range r that the lidar field covers, sight aside."""
        p = self.p
        sx, sy, syaw = sensor
        rs = np.hypot(x - sx, y - sy)
        bs = np.arctan2(y - sy, x - sx) - syaw
        bs = np.arctan2(np.sin(bs), np.cos(bs))
        ok = (r >= p.lidar_range_min) & (rs <= lidar_limit(p, bs))
        ok &= np.abs(bs) <= math.radians(p.lidar_fov_deg) / 2 + 1e-9
        # Legs and body: whole sectors outside the clear one drop out at random.
        outside = np.abs(bs) > math.radians(p.lidar_clear_fov_deg) / 2
        if p.lidar_self_occlusion_prob > 0 and outside.any():
            width = math.radians(OCCLUSION_SECTOR_DEG)
            sectors = np.floor((bs + math.pi) / width).astype(int)
            blocked = self.rng.random(int(math.ceil(2 * math.pi / width)) + 1) \
                < p.lidar_self_occlusion_prob
            ok &= ~(outside & blocked[sectors])
        return ok

    def _lidar_sees(self, robot, shapes, index, points):
        """Like _sees for the lidar: its own field, sight lines from the sensor."""
        sensor = self._sensor(robot)
        for x, y, r, _ in points:
            if not self._lidar_in_field(sensor, np.array([x]), np.array([y]), np.array([r]))[0]:
                continue
            if not self.p.occlusion or line_of_sight(sensor[0], sensor[1], x, y, shapes, {index}):
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

            lidar = self._lidar_sees(robot, shapes, index, points)
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
        """Lidar ranges over 360 deg from base_link, starting at -pi, with noise.

        Rays are cast from base_link; a hit is kept if the lidar's field covers it.
        """
        p = self.p
        n = p.scan_rays
        bearings = -math.pi + np.arange(n) * (2 * math.pi / n)
        ranges, _ = ray_cast(robot[0], robot[1], bearings + robot[2], shapes, p.lidar_range_max)
        hit = np.isfinite(ranges)
        a = bearings[hit] + robot[2]
        x = robot[0] + ranges[hit] * np.cos(a)
        y = robot[1] + ranges[hit] * np.sin(a)
        covered = self._lidar_in_field(self._sensor(robot), x, y, ranges[hit])
        ranges[np.flatnonzero(hit)[~covered]] = math.inf
        hit = np.isfinite(ranges)
        a, b = p.lidar_range_std
        std = (a + b * ranges[hit]) * p.noise_scale
        ranges[hit] += self.rng.normal(0.0, 1.0, hit.sum()) * std
        ranges[hit & (ranges < p.lidar_range_min)] = math.inf
        return bearings, ranges


# Leader status, as in follow_anywhere_interfaces/PerceptionLeader.
NO_LEADER, VISIBLE, LOST = 0, 1, 2


@dataclass
class Leader:
    """What /perception/leader reports. Points (x, y) in base_link; NaN when unknown."""

    status: int = NO_LEADER
    closest_point: tuple = (math.nan, math.nan)
    velocity: tuple = (0.0, 0.0)
    position_uncertainty: float = math.nan
    velocity_uncertainty: float = math.nan
    lost_duration: float = math.nan
    last_known_point: tuple = (math.nan, math.nan)


@dataclass
class Obstacle:
    """One entry of /perception/obstacles. Points (x, y) in base_link."""

    closest_point: tuple
    line_of_sight_point: tuple
    velocity: tuple
    position_uncertainty: float
    velocity_uncertainty: float


def _cartesian(r, b):
    return r * math.cos(b), r * math.sin(b)


def line_of_sight_point(cx, cy, radius, ex, ey):
    """Point of the circle (cx, cy, radius) nearest the segment from the origin to (ex, ey)."""
    length2 = ex * ex + ey * ey
    s = 0.0 if length2 == 0.0 else min(1.0, max(0.0, (cx * ex + cy * ey) / length2))
    px, py = s * ex, s * ey
    d = math.hypot(cx - px, cy - py)
    if d <= radius:
        return px, py               # the segment crosses the circle
    k = radius / d
    return cx + (px - cx) * k, cy + (py - cy) * k


class LeaderTracker:
    """Turns detections into the leader / obstacles split the perception stack publishes.

    The leader is VISIBLE while the camera identifies it (CLASS_LEADER), LOST after
    that, NO_LEADER until it is first seen. An unidentified leader is an ordinary
    obstacle, as in the real stack. last_known_point is kept in odom and projected
    into the current base_link, so it stays put in the world while the robot moves.
    """

    def __init__(self):
        self._last_seen_t = None
        self._last_known = None       # (x, y) in odom
        self._last_velocity = (0.0, 0.0)
        self._last_position_std = math.nan
        self._last_velocity_std = math.nan

    def update(self, detections, leader_id, robot, t):
        """(Leader, [Obstacle]) from detections at robot pose (x, y, yaw) in odom, time t (s)."""
        rx, ry, ryaw = robot
        c, s = math.cos(ryaw), math.sin(ryaw)
        leader = Leader()
        seen = None
        for d in detections:
            if d.id == leader_id and d.class_id == CLASS_LEADER:
                seen = d
                break

        if seen is not None:
            px, py = _cartesian(seen.nearest_range, seen.nearest_bearing)
            velocity = seen.velocity if seen.moving else (0.0, 0.0)
            self._last_seen_t = t
            self._last_known = (rx + c * px - s * py, ry + s * px + c * py)
            self._last_velocity = velocity
            self._last_position_std = seen.range_std
            self._last_velocity_std = seen.velocity_std
            leader.status = VISIBLE
            leader.closest_point = (px, py)
            leader.lost_duration = 0.0
        elif self._last_seen_t is not None:
            leader.status = LOST
            leader.lost_duration = max(0.0, t - self._last_seen_t)

        if self._last_known is not None:
            dx, dy = self._last_known[0] - rx, self._last_known[1] - ry
            leader.last_known_point = (c * dx + s * dy, -s * dx + c * dy)
            leader.velocity = self._last_velocity
            leader.velocity_uncertainty = self._last_velocity_std
            leader.position_uncertainty = (self._last_position_std
                                           + self._last_velocity_std * leader.lost_duration)

        ex, ey = leader.last_known_point
        has_end = math.isfinite(ex)
        obstacles = []
        for d in detections:
            if d is seen:
                continue
            cx, cy = _cartesian(d.range, d.bearing)
            obstacles.append(Obstacle(
                closest_point=_cartesian(d.nearest_range, d.nearest_bearing),
                line_of_sight_point=(line_of_sight_point(cx, cy, d.radius, ex, ey)
                                     if has_end else (math.nan, math.nan)),
                velocity=d.velocity if d.moving else (0.0, 0.0),
                position_uncertainty=d.range_std,
                velocity_uncertainty=d.velocity_std,
            ))
        return leader, obstacles
