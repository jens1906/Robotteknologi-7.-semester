"""Leader logic on synthetic people: no models, no ROS."""
from dataclasses import dataclass

import numpy as np

from vision_pre.leader import ENROLL_VIEWS, Leader, State

P, D = 9, 16


@dataclass
class Person:
    id: int
    box: tuple

    @property
    def area(self):
        return (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])


def unit(rng):
    v = rng.normal(size=(P, D))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class World:
    """Who looks like what. look[track id] is a (P, D) appearance; vis[id] says which parts show."""

    def __init__(self):
        rng = np.random.default_rng(1)
        self.rng, self.leader, self.other = rng, unit(rng), unit(rng)
        self.look, self.vis = {}, {}

    def embed(self, ids):
        emb = np.stack([self.look[i] + self.rng.normal(scale=0.02, size=(P, D)) for i in ids])
        emb /= np.linalg.norm(emb, axis=2, keepdims=True)
        return emb, np.stack([self.vis.get(i, np.ones(P, bool)) for i in ids])


def box(x, w=100):
    return (x, 0, x + w, 300)


def run(leader, world, people, t0, seconds, dt=0.1):
    """Step the leader with a fixed set of people. Returns the last result and the end time."""
    t, res = t0, None
    while t < t0 + seconds:
        res = leader.update(people, world.embed, t)
        t += dt
    return res, t


def following(world, leader_id=1):
    leader = Leader()
    world.look[leader_id], world.look[2] = world.leader, world.other
    people = [Person(leader_id, box(0)), Person(2, box(400))]
    leader.enroll(leader_id, 0.0)
    res, t = run(leader, world, people, 0.0, ENROLL_VIEWS * 0.5 + 1)
    assert res.state == State.FOLLOWING and res.leader_id == leader_id
    return leader, people, t


def test_enrolls_the_chosen_person_after_enough_views():
    world = World()
    leader, people, _ = following(world)
    assert leader.n_views == ENROLL_VIEWS


def test_enrolment_waits_while_somebody_overlaps():
    world = World()
    world.look[1], world.look[2] = world.leader, world.other
    leader = Leader()
    leader.enroll(1, 0.0)
    res, t = run(leader, world, [Person(1, box(0)), Person(2, box(50))], 0.0, 8)
    assert res.state == State.ENROLLING and leader.n_views == 0
    res, _ = run(leader, world, [Person(1, box(0)), Person(2, box(400))], t, 8)
    assert res.state == State.FOLLOWING


def test_crossing_removes_the_leader_and_the_same_person_is_found_again():
    world = World()
    leader, people, t = following(world)
    res = leader.update([Person(1, box(0)), Person(2, box(50))], world.embed, t)
    assert res.state == State.SEARCHING and res.leader_id is None and res.reason == 'crossing'
    res, _ = run(leader, world, people, t + 0.1, 4)
    assert res.state == State.FOLLOWING and res.leader_id == 1 and leader.track_id == 1


def test_a_missing_frame_does_not_lose_the_leader():
    world = World()
    leader, people, t = following(world)
    res = leader.update([Person(2, box(400))], world.embed, t)          # flicker: nobody is flagged
    assert res.state == State.FOLLOWING and res.leader_id is None
    res = leader.update(people, world.embed, t + 0.2)                   # back, and the check passes
    assert res.state == State.FOLLOWING and res.leader_id == 1


def test_lost_track_returns_under_a_new_id():
    world = World()
    leader, people, t = following(world)
    world.look[7] = world.leader
    res, t = run(leader, world, [Person(2, box(400))], t, 1.3)           # gone for longer than the grace time
    assert res.state == State.SEARCHING and res.leader_id is None
    res, _ = run(leader, world, [Person(7, box(10)), Person(2, box(400))], t, 4)
    assert res.state == State.FOLLOWING and res.leader_id == 7


def test_the_person_being_enrolled_may_flicker():
    world = World()
    world.look[1] = world.leader
    leader = Leader()
    leader.enroll(1, 0.0)
    assert leader.update([], world.embed, 0.5).state == State.ENROLLING
    assert leader.update([], world.embed, 2.6).event == 'enroll_failed'


def test_two_identical_looking_people_are_never_picked():
    world = World()
    leader, people, t = following(world)
    run(leader, world, [Person(2, box(400))], t, 1.3)               # track 1 gone -> searching
    world.look[7] = world.look[8] = world.leader
    res, _ = run(leader, world, [Person(7, box(0)), Person(8, box(400))], t + 1.4, 6)
    assert res.state == State.SEARCHING and res.leader_id is None


def test_somebody_else_is_not_picked():
    world = World()
    leader, people, t = following(world)
    run(leader, world, [Person(2, box(400))], t, 1.3)
    res, _ = run(leader, world, [Person(2, box(400))], t + 1.4, 6)
    assert res.state == State.SEARCHING


def test_unknown_view_is_not_accepted():
    world = World()
    leader, people, t = following(world)
    run(leader, world, [Person(2, box(400))], t, 1.3)
    world.look[7] = world.leader
    world.vis[7] = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], bool)     # only the head shows
    res, _ = run(leader, world, [Person(7, box(0))], t + 1.4, 6)
    assert res.state == State.SEARCHING


def test_locked_track_that_starts_to_look_different_is_dropped():
    world = World()
    leader, people, t = following(world)
    world.look[1] = world.other                                  # the track now shows somebody else
    res, _ = run(leader, world, people, t, 4)
    assert res.state == State.SEARCHING and res.leader_id is None


def test_a_saved_leader_is_found_again_in_a_new_session(tmp_path):
    world = World()
    leader, people, t = following(world)
    path = str(tmp_path / 'sub' / 'leader.npz')
    leader.save(path, 'kpr:test')
    fresh = Leader()
    assert not fresh.load(path, 'osnet') and not fresh.load(str(tmp_path / 'missing.npz'), 'kpr:test')
    assert fresh.state == State.IDLE                           # a wrong model or no file changes nothing
    assert fresh.load(path, 'kpr:test') and fresh.state == State.SEARCHING and fresh.n_views == ENROLL_VIEWS
    world.look[9] = world.leader                               # new session, new track ids
    res, _ = run(fresh, world, [Person(9, box(0)), Person(2, box(400))], 0.0, 4)
    assert res.state == State.FOLLOWING and res.leader_id == 9


def test_a_new_scene_keeps_the_leader_but_forgets_the_tracks():
    world = World()
    leader, people, t = following(world)
    leader.new_scene()
    assert leader.state == State.SEARCHING and leader.track_id is None and leader.n_views == ENROLL_VIEWS
    world.look[11] = world.leader
    res, _ = run(leader, world, [Person(11, box(0)), Person(2, box(400))], t, 4)
    assert res.state == State.FOLLOWING and res.leader_id == 11
    unfinished = Leader()
    unfinished.enroll(1, 0.0)
    unfinished.new_scene()
    assert unfinished.state == State.IDLE                      # nothing to keep yet


def test_the_leader_is_found_among_many_people_even_if_small():
    world = World()
    leader, people, t = following(world)
    run(leader, world, [Person(2, box(400))], t, 1.3)               # track 1 gone -> searching
    crowd = []
    for i in range(20, 32):                                         # 12 big strangers, the leader is the smallest
        world.look[i] = unit(world.rng)
        crowd.append(Person(i, (i * 50, 0, i * 50 + 100, 300)))
    world.look[40] = world.leader
    crowd.append(Person(40, (2000, 100, 2040, 190)))
    res, _ = run(leader, world, crowd, t + 1.4, 15)
    assert res.state == State.FOLLOWING and res.leader_id == 40


def test_enrolment_survives_a_track_id_change_but_not_a_different_person():
    world = World()
    world.look[1], world.look[5], world.look[6] = world.leader, world.leader, world.other
    leader = Leader()
    leader.enroll(1, 0.0)
    run(leader, world, [Person(1, box(0))], 0.0, 1.2)                # a few views of track 1
    res, _ = run(leader, world, [Person(5, box(5))], 1.3, 5)         # the tracker renamed them 5, same place
    assert res.state == State.FOLLOWING and res.leader_id == 5
    other = Leader()
    other.enroll(1, 0.0)
    run(other, world, [Person(1, box(0))], 0.0, 1.2)
    res, _ = run(other, world, [Person(6, box(5))], 1.3, 5)          # a different-looking person took the place
    assert res.state == State.IDLE
