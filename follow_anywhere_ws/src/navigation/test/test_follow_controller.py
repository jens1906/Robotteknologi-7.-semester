import math
from types import SimpleNamespace

from navigation.follow_controller import CLASS_LEADER, FollowController


def det(id, range, bearing, class_id=CLASS_LEADER, radius=0.25):
    return SimpleNamespace(id=id, class_id=class_id, range=range, bearing=bearing,
                           radius=radius)


def test_no_leader_no_motion():
    c = FollowController()
    c.update_detections([det(1, 3.0, 0.0, class_id=2)], 0.0)
    assert c.compute([]).is_zero()


def test_drives_towards_far_leader_and_turns():
    c = FollowController()
    c.update_detections([det(7, 3.0, 0.3)], 0.0)
    cmd = c.compute([])
    assert cmd.vx > 0.0 and cmd.wz > 0.0


def test_keeps_id_when_class_drops():
    c = FollowController()
    c.update_detections([det(7, 3.0, 0.0)], 0.0)
    c.update_detections([det(7, 3.0, 0.0, class_id=0), det(8, 2.0, 0.0, class_id=2)], 0.1)
    assert c.leader_id == 7


def test_unlocks_after_timeout():
    c = FollowController()
    c.update_detections([det(7, 3.0, 0.0)], 0.0)
    c.update_detections([], 1.0)
    assert c.has_leader()
    c.update_detections([], 2.0)
    assert not c.has_leader()


def test_stops_for_obstacle_ahead_but_not_for_leader():
    c = FollowController()
    c.update_detections([det(7, 3.0, 0.0)], 0.0)
    assert c.compute([(3.0, 0.0)]).vx > 0.0          # scan hit on the leader
    assert c.compute([(0.5, 0.0)]).vx == 0.0         # box in the way


def test_obstacle_on_the_left_pushes_right():
    c = FollowController()
    c.update_detections([det(7, 3.0, 0.0)], 0.0)
    cmd = c.compute([(0.0, 0.6)])
    assert cmd.vy < 0.0
    assert math.isfinite(cmd.vx)
