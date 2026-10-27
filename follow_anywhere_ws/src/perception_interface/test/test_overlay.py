from types import SimpleNamespace

import numpy as np

from perception_interface.overlay import (LEADER_COLOR, DrawTrackedPeople, FindPersonAtPoint)


def MakeTrackedPerson(trackId, xMin, yMin, xMax, yMax, isLeader=False, reidDistance=float('nan')):
    return SimpleNamespace(id=trackId, x_min=xMin, y_min=yMin, x_max=xMax, y_max=yMax, is_leader=isLeader,
                           reid_distance=reidDistance, position_valid=False,
                           position=SimpleNamespace(x=0.0, y=0.0))


def test_click_picks_the_smallest_box_under_the_point():
    bigPerson = MakeTrackedPerson(1, 0, 0, 200, 200)
    smallPerson = MakeTrackedPerson(2, 50, 50, 100, 100)
    assert FindPersonAtPoint([bigPerson, smallPerson], 60, 60) is smallPerson
    assert FindPersonAtPoint([bigPerson, smallPerson], 150, 150) is bigPerson


def test_click_outside_every_box_finds_nobody():
    assert FindPersonAtPoint([MakeTrackedPerson(1, 0, 0, 50, 50)], 300, 300) is None
    assert FindPersonAtPoint([], 10, 10) is None


def test_drawing_changes_the_image_and_uses_the_leader_color():
    image = np.zeros((200, 200, 3), np.uint8)
    DrawTrackedPeople(image, [MakeTrackedPerson(7, 40, 60, 160, 180, isLeader=True)], True)
    assert image.any()
    assert tuple(image[120, 40]) == LEADER_COLOR


def test_drawing_with_no_people_leaves_the_image_unchanged():
    image = np.zeros((50, 50, 3), np.uint8)
    DrawTrackedPeople(image, [], True)
    assert not image.any()
