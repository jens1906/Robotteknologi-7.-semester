import numpy as np

from vision.appearance import colour_hist, colour_sim, inner_crop
from vision.motion import LeaderMotion


def person(shirt, trousers, h=200, w=100):
    """Synthetic BGR crop: shirt colour in the upper half, trousers in the lower half."""
    img = np.zeros((h, w, 3), np.uint8)
    img[: h // 2] = shirt
    img[h // 2:] = trousers
    return img


RED, BLUE, BLACK, WHITE = (0, 0, 220), (220, 0, 0), (10, 10, 10), (240, 240, 240)


def test_same_clothes_match_and_different_clothes_do_not():
    a = colour_hist(person(RED, BLACK))
    same = colour_hist(person(RED, BLACK))
    other = colour_hist(person(BLUE, WHITE))
    assert colour_sim(a, same[None])[0] > 0.99
    assert colour_sim(a, other[None])[0] < 0.2


def test_colour_sim_handles_many_views_at_once():
    a = colour_hist(person(RED, BLACK))
    views = np.stack([colour_hist(person(RED, BLACK)), colour_hist(person(BLUE, WHITE))])
    sims = colour_sim(a, views)
    assert sims.shape == (2,) and sims[0] > sims[1]


def test_tiny_crop_gives_no_histogram():
    assert colour_hist(np.zeros((8, 4, 3), np.uint8)) is None


def test_inner_crop_trims_the_box_and_stays_inside_the_frame():
    frame = np.zeros((480, 640, 3), np.uint8)
    crop = inner_crop(frame, (100, 100, 200, 300), side=0.1, vert=0.05)
    assert crop.shape[:2] == (180, 80)
    edge = inner_crop(frame, (-20, -20, 50, 60))        # box sticking out of the image
    assert edge.size > 0


def test_motion_allows_near_and_rejects_far_positions():
    m = LeaderMotion(max_speed=1.5, base_radius=0.6)
    m.update((0.0, 0.0, 2.0), now=0.0)
    assert m.allowed((0.3, 0.0, 2.2), now=0.1)
    assert not m.allowed((3.0, 0.0, 5.0), now=0.1)


def test_motion_gate_widens_with_time():
    m = LeaderMotion(max_speed=1.5, base_radius=0.6)
    m.update((0.0, 0.0, 2.0), now=0.0)
    far = (3.0, 0.0, 5.0)                                # about 4.2 m away
    assert not m.allowed(far, now=1.0)
    assert m.allowed(far, now=5.0)


def test_motion_unknown_position_is_never_excluded():
    m = LeaderMotion()
    assert m.allowed((9.0, 0.0, 9.0), now=0.0)           # no leader position known yet
    m.update((0.0, 0.0, 2.0), now=0.0)
    assert m.allowed(None, now=0.1)                      # candidate without depth


def test_motion_predicts_a_walking_leader():
    m = LeaderMotion()
    for i in range(10):
        m.update((i * 0.1, 0.0, 3.0), now=i * 0.1)       # walking 1 m/s sideways
    px, pz = m.predicted(1.0)
    assert 0.95 <= px <= 1.15 and abs(pz - 3.0) < 0.05   # last seen at x=0.9, 0.1 s ago
