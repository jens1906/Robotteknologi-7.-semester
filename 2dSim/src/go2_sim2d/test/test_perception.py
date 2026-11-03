import math

import pytest

from go2_sim2d.geometry import Box, Circle
from go2_sim2d.perception import (
    CLASS_LEADER, CLASS_OBJECT, CLASS_UNKNOWN, SENSOR_CAMERA, SENSOR_LIDAR,
    Entity, FakePerception, Params, VelocityTracker)


def exact(**overrides):
    """Perception without noise or dropouts: reports ground truth."""
    params = Params(noise_scale=0.0, dropout_prob=0.0, velocity_alpha=1.0)
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
    person = Entity(1, 'person', Circle(-2.0, 0.0, 0.25))
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
    assert std == pytest.approx(0.02 + 0.01 * 5.0, rel=0.2)


def test_scan():
    perception = exact(scan_rays=4)
    bearings, ranges = perception.scan((0.0, 0.0, 0.0), [Box(5.0, 0.0, 0.0, 1.0, 1.0)])
    assert list(bearings) == pytest.approx([-math.pi, -math.pi / 2, 0.0, math.pi / 2])
    assert ranges[2] == pytest.approx(4.5)
    assert math.isinf(ranges[0])
