import numpy as np

from vision.relink import Relinker

FRAME = (640, 480)


def unit(*v):
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v)


A = unit(1, 0, 0, 0)
B = unit(0, 1, 0, 0)
A2 = unit(1, 0.1, 0, 0)         # the same person seen a bit differently
A3 = unit(1, 0.15, 0, 0)
BOX_A = (100.0, 100.0, 160.0, 300.0)
BOX_NEAR = (130.0, 100.0, 190.0, 300.0)
BOX_FAR = (500.0, 100.0, 560.0, 300.0)


class World:
    """Drives a Relinker with a fake embedder and keeps count of embeddings."""

    def __init__(self, **kw):
        self.r = Relinker(**kw)
        self.vec = {}
        self.calls = 0
        self.t = 0.0

    def embed(self, box):
        self.calls += 1
        return self.vec.get(box)

    def step(self, visible, hidden=(), dt=0.1):
        self.t += dt
        return self.r.update(self.t, visible, list(hidden), self.embed, FRAME)

    def see(self, tid, box, vec, seconds=3.0):
        self.vec[box] = vec
        for _ in range(int(seconds / 0.1)):
            self.step([(tid, box)])

    def hide_then_return(self, hidden, new_tid, new_box, new_vec, secs=1.0, frames=6):
        self.vec[new_box] = new_vec
        out = None
        for _ in range(frames):
            out = self.step([(new_tid, new_box)], [(t, b, secs) for t, b in hidden])
        return out


def test_returning_person_gets_the_old_id_back():
    w = World()
    w.see(1, BOX_A, A)
    out = w.hide_then_return([(1, BOX_A)], 2, BOX_NEAR, A2)
    assert out[2] == 1
    assert w.r.relinks == 1


def test_a_different_looking_person_keeps_a_new_id():
    w = World()
    w.see(1, BOX_A, A)
    out = w.hide_then_return([(1, BOX_A)], 2, BOX_NEAR, B)
    assert out[2] == 2


def test_somebody_who_could_not_have_walked_there_is_not_linked():
    w = World()
    w.see(1, BOX_A, A)
    out = w.hide_then_return([(1, BOX_A)], 2, BOX_FAR, A2, secs=0.5)
    assert out[2] == 2


def test_the_right_hidden_person_is_chosen():
    w = World()
    w.see(1, BOX_A, A)
    w.see(5, BOX_NEAR, B)
    out = w.hide_then_return([(1, BOX_A), (5, BOX_NEAR)], 9, (110.0, 100.0, 170.0, 300.0), B * 0.99
                             + 0.01 * A)
    assert out[9] == 5


def test_two_lookalike_hidden_people_are_ambiguous_so_nothing_is_linked():
    w = World()
    w.see(1, BOX_A, A)
    w.see(5, BOX_NEAR, A3)
    out = w.hide_then_return([(1, BOX_A), (5, BOX_NEAR)], 9, (110.0, 100.0, 170.0, 300.0), A2)
    assert out[9] == 9


def test_one_hidden_person_is_given_to_one_new_id_only():
    w = World()
    w.see(1, BOX_A, A)
    box2 = (200.0, 100.0, 260.0, 300.0)
    w.vec[BOX_NEAR], w.vec[box2] = A2, A3
    for _ in range(6):
        out = w.step([(2, BOX_NEAR), (3, box2)], [(1, BOX_A, 1.0)])
    assert 1 in (out[2], out[3])                  # one of them is the returning person ...
    assert out[2] != out[3]                       # ... and they do not both get the old id


def test_hidden_for_too_long_is_not_a_candidate():
    w = World(keep_s=10.0)
    w.see(1, BOX_A, A)
    out = w.hide_then_return([(1, BOX_A)], 2, BOX_NEAR, A2, secs=11.0)
    assert out[2] == 2


def test_people_who_overlap_are_not_described():
    w = World()
    w.vec[BOX_A] = A
    w.vec[BOX_NEAR] = B
    for _ in range(10):
        w.step([(1, BOX_A), (2, BOX_NEAR)])         # the boxes overlap
    assert w.calls == 0


def test_visible_people_are_described_rarely():
    w = World(refresh_s=1.5)
    w.see(1, BOX_A, A, seconds=6.0)
    assert w.calls <= 6                               # settle + one per 1.5 s, not one per frame
