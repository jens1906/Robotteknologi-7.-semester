import math

import pytest

from go2_sim2d.geometry import Box, Circle, distance, line_of_sight, nearest_point, ray_cast


def test_ray_hits_box_face():
    r, owner = ray_cast(0.0, 0.0, [0.0], [Box(5.0, 0.0, 0.0, 1.0, 1.0)])
    assert r[0] == pytest.approx(4.5)
    assert owner[0] == 0


def test_ray_hits_rotated_box_corner_first():
    # Rotated 45 deg: the corner points at the origin, sqrt(0.5) from the centre.
    r, _ = ray_cast(0.0, 0.0, [0.0], [Box(5.0, 0.0, math.pi / 4, 1.0, 1.0)])
    assert r[0] == pytest.approx(5.0 - math.sqrt(0.5))


def test_ray_nearest_of_several_and_misses():
    shapes = [Circle(3.0, 0.0, 0.5), Box(6.0, 0.0, 0.0, 1.0, 1.0), Circle(0.0, -4.0, 1.0)]
    r, owner = ray_cast(0.0, 0.0, [0.0, math.pi / 2, -math.pi / 2], shapes)
    assert r[0] == pytest.approx(2.5) and owner[0] == 0
    assert math.isinf(r[1]) and owner[1] == -1
    assert r[2] == pytest.approx(3.0) and owner[2] == 2


def test_ray_max_range():
    r, owner = ray_cast(0.0, 0.0, [0.0], [Circle(10.0, 0.0, 0.5)], max_range=5.0)
    assert math.isinf(r[0]) and owner[0] == -1


def test_line_of_sight():
    wall = Box(2.0, 0.0, 0.0, 0.2, 2.0)
    assert not line_of_sight(0.0, 0.0, 4.0, 0.0, [wall])
    assert line_of_sight(0.0, 0.0, 4.0, 3.0, [wall])
    assert line_of_sight(0.0, 0.0, 4.0, 0.0, [wall], ignore={0})


def test_distance_and_nearest_point():
    box = Box(0.0, 5.0, 0.0, 4.0, 1.0)
    assert distance(box, 0.0, 0.0) == pytest.approx(4.5)
    assert nearest_point(box, 3.0, 0.0) == pytest.approx((2.0, 4.5))
    circle = Circle(3.0, 4.0, 1.0)
    assert distance(circle, 0.0, 0.0) == pytest.approx(4.0)
    assert nearest_point(circle, 0.0, 0.0) == pytest.approx((2.4, 3.2))
