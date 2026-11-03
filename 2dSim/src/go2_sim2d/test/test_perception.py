import math

import numpy as np
import pytest

from go2_sim2d.geometry import Box, Circle
from go2_sim2d.perception import (
    CLASS_LEADER, CLASS_OBJECT, CLASS_UNKNOWN, LOST, NO_LEADER, SENSOR_CAMERA, SENSOR_LIDAR,
    VISIBLE, Entity, FakePerception, LeaderTracker, Params, VelocityTracker, lidar_limit,
    line_of_sight_point)


def exact(**overrides):
    """Perception without noise or dropouts: reports ground truth."""
    params = Params(noise_scale=0.0, dropout_prob=0.0, velocity_alpha=1.0,
                    lidar_self_occlusion_prob=0.0)
    for k, v in overrides.items():
        setattr(params, k, v)
    return FakePerception(params, seed=0)


def by_id(detections):
    return {d.id: d for d in detections}


def test_leader_ahead_is_seen_by_both_sensors():
    leader = Entity(1, 'person', Circle(2.0, 0.0, 0.25), leader=True)
    (d,) = exact().detect((0.0, 0.0, 0.0), [leader], 0.0)
    assert d.class_id == CLASS_LEADER
    assert d.sensors == SENSOR_LIDAR | SENSOR_CAMERA
    assert d.range == pytest.approx(2.0)
    assert d.bearing == pytest.approx(0.0)
    assert d.nearest_range == pytest.approx(1.75)


def test_behind_is_lidar_only_and_unclassified():
    person = Entity(1, 'person', Circle(-1.2, 0.0, 0.25))
    (d,) = exact().detect((0.0, 0.0, 0.0), [person], 0.0)
    assert d.sensors == SENSOR_LIDAR
    assert d.class_id == CLASS_UNKNOWN
    assert abs(d.bearing) == pytest.approx(math.pi)


def test_polar_is_relative_to_robot_pose():
    # Robot at (1, 1) facing +y; object at (1, 4) is straight ahead, 3 m.
    obj = Entity(1, 'static', Circle(1.0, 4.0, 0.3))
    (d,) = exact().detect((1.0, 1.0, math.pi / 2), [obj], 0.0)
    assert d.class_id == CLASS_OBJECT
    assert d.range == pytest.approx(3.0)
    assert d.bearing == pytest.approx(0.0)


def test_occlusion():
    wall = Entity(1, 'static', Box(2.0, 0.0, 0.0, 0.2, 2.0))
    hidden = Entity(2, 'person', Circle(4.0, 0.0, 0.25))
    assert set(by_id(exact().detect((0.0, 0.0, 0.0), [wall, hidden], 0.0))) == {1}
    seen = exact(occlusion=False).detect((0.0, 0.0, 0.0), [wall, hidden], 0.0)
    assert set(by_id(seen)) == {1, 2}


def test_out_of_range():
    far = Entity(1, 'static', Circle(30.0, 0.0, 0.3))
    assert exact().detect((0.0, 0.0, 0.0), [far], 0.0) == []


def test_lidar_limit_from_tilt():
    p = Params()
    behind = p.lidar_height / math.tan(math.radians(p.lidar_tilt_deg))
    assert lidar_limit(p, math.pi) == pytest.approx(behind)          # ~1.57 m
    assert lidar_limit(p, 0.0) == pytest.approx(p.lidar_range_max)
    assert lidar_limit(p, math.pi / 2) == pytest.approx(p.lidar_range_max)


def test_lidar_is_short_sighted_behind():
    far_behind = Entity(1, 'static', Circle(-3.0, 0.0, 0.25))
    beside = Entity(2, 'static', Circle(0.283, 3.0, 0.25))
    seen = by_id(exact().detect((0.0, 0.0, 0.0), [far_behind, beside], 0.0))
    assert set(seen) == {2}
    assert seen[2].sensors == SENSOR_LIDAR


def test_legs_never_block_the_clear_sector():
    p = exact(lidar_self_occlusion_prob=1.0, scan_rays=360)
    ring = [Circle(0.283 + 2.0 * math.cos(a), 2.0 * math.sin(a), 0.3)
            for a in [math.radians(d) for d in range(0, 360, 10)]]
    bearings, ranges = p.scan((0.0, 0.0, 0.0), ring)
    hit = np.isfinite(ranges)
    # Rays whose hit is within +-70 deg of the sensor's front all return.
    hx = ranges * np.cos(bearings) - 0.283
    hy = ranges * np.sin(bearings)
    front = np.abs(np.arctan2(hy, hx)) < math.radians(70)
    assert hit[front & np.isfinite(hx)].all() and (front & hit).any()
    # Everything outside +-75 deg is blocked when the legs always are.
    back = np.abs(np.arctan2(hy, hx)) > math.radians(75)
    assert not (back & hit).any()


def test_wall_nearest_point():
    # Long wall 3 m to the left; its centre is far ahead but the nearest point is beside us.
    wall = Entity(1, 'static', Box(10.0, 3.0, 0.0, 20.0, 0.2))
    (d,) = exact().detect((0.0, 0.0, 0.0), [wall], 0.0)
    assert d.nearest_range == pytest.approx(2.9)
    assert d.nearest_bearing == pytest.approx(math.pi / 2)


def test_velocity_in_base_link():
    # Robot faces +y; a person walking +x over the ground moves to the robot's right (-y).
    person = Entity(1, 'person', Circle(0.0, 2.0, 0.25), velocity=(1.0, 0.0))
    (d,) = exact().detect((0.0, 0.0, math.pi / 2), [person], 0.0)
    assert d.velocity == pytest.approx((0.0, -1.0))
    assert d.moving


def test_velocity_tracker():
    tracker = VelocityTracker()
    tracker.update('a', 0.0, 0.0, 0.0)
    assert tracker.update('a', 0.5, -0.1, 0.5) == pytest.approx((1.0, -0.2))


def test_noise_is_bounded():
    person = Entity(1, 'person', Circle(5.0, 0.0, 0.25))
    perception = FakePerception(Params(dropout_prob=0.0), seed=1)
    ranges = [perception.detect((0.0, 0.0, 0.0), [person], i * 0.1)[0].range
              for i in range(500)]
    mean = sum(ranges) / len(ranges)
    std = math.sqrt(sum((r - mean) ** 2 for r in ranges) / len(ranges))
    assert mean == pytest.approx(5.0, abs=0.02)
    a, b = Params().lidar_range_std
    assert std == pytest.approx(a + b * 5.0, rel=0.2)


def test_scan():
    perception = exact(scan_rays=4)
    bearings, ranges = perception.scan((0.0, 0.0, 0.0), [Box(5.0, 0.0, 0.0, 1.0, 1.0)])
    assert list(bearings) == pytest.approx([-math.pi, -math.pi / 2, 0.0, math.pi / 2])
    assert ranges[2] == pytest.approx(4.5)
    assert math.isinf(ranges[0])


def test_leader_tracker_status():
    leader = Entity(1, 'person', Circle(2.0, 0.0, 0.25), leader=True)
    box = Entity(2, 'static', Box(0.0, 3.0, 0.0, 1.0, 1.0))
    p, tracker = exact(), LeaderTracker()

    state, obstacles = tracker.update([], 1, (0.0, 0.0, 0.0), 0.0)
    assert state.status == NO_LEADER and math.isnan(state.last_known_point[0])
    assert obstacles == []

    detections = p.detect((0.0, 0.0, 0.0), [leader, box], 0.0)
    state, obstacles = tracker.update(detections, 1, (0.0, 0.0, 0.0), 0.0)
    assert state.status == VISIBLE and state.lost_duration == 0.0
    assert state.closest_point == pytest.approx((1.75, 0.0))
    assert len(obstacles) == 1                       # the box, not the leader
    assert obstacles[0].closest_point == pytest.approx((0.0, 2.5))

    # Leader out of sight, robot moved 1 m forward: last known point stays in place.
    state, obstacles = tracker.update([], 1, (1.0, 0.0, 0.0), 2.0)
    assert state.status == LOST and state.lost_duration == pytest.approx(2.0)
    assert math.isnan(state.closest_point[0])
    assert state.last_known_point == pytest.approx((0.75, 0.0))


def test_unidentified_leader_is_an_obstacle():
    leader = Entity(1, 'person', Circle(-1.2, 0.0, 0.25), leader=True)   # behind: lidar only
    detections = exact().detect((0.0, 0.0, 0.0), [leader], 0.0)
    state, obstacles = LeaderTracker().update(detections, 1, (0.0, 0.0, 0.0), 0.0)
    assert state.status == NO_LEADER
    assert len(obstacles) == 1


def test_line_of_sight_point():
    # Circle beside the line to (4, 0): nearest point is on the circle, towards the line.
    assert line_of_sight_point(2.0, 1.0, 0.25, 4.0, 0.0) == pytest.approx((2.0, 0.75))
    # Circle across the line: the point is on the line.
    assert line_of_sight_point(2.0, 0.1, 0.25, 4.0, 0.0) == pytest.approx((2.0, 0.0))
    # Past the end of the line: towards the end point.
    assert line_of_sight_point(6.0, 0.0, 0.5, 4.0, 0.0) == pytest.approx((5.5, 0.0))
