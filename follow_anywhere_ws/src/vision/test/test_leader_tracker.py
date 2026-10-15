import numpy as np

from vision.leader_tracker import Candidate, LeaderTracker, State

FRAME = (640, 480)


def unit(*v):
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v)


LEADER = unit(1, 0, 0)
OTHER = unit(0, 1, 0)
LOOKALIKE = unit(1, 0.05, 0)     # almost the same as LEADER


def cand(track_id, x=320, cls=0, size=100, pos=None):
    return Candidate(track_id, cls, (x - size / 2, 140, x + size / 2, 340), pos=pos)


def make(vectors, **kw):
    """Tracker plus an embed_fn that looks the vector up by track id."""
    tracker = LeaderTracker(calib_frames=5, min_calib_frames=3, confirm_frames=2,
                            reid_period=0.0, verify_period=1.0, **kw)
    return tracker, lambda c: vectors[c.track_id]


def calibrate(tracker, embed, cands, t=0.0):
    tracker.start_calibration(t)
    res = None
    for i in range(5):
        res = tracker.update(cands, embed, FRAME, t + i * 0.1)
    return res


def test_calibration_picks_candidate_nearest_centre():
    tracker, embed = make({1: LEADER, 2: OTHER})
    res = calibrate(tracker, embed, [cand(2, x=100), cand(1, x=330)])
    assert res.event == 'calibrated'
    assert res.state == State.TRACKING
    assert tracker.locked_id == 1
    assert tracker.gallery.shape == (5, 3)


def test_calibration_fails_without_candidates():
    tracker, embed = make({})
    tracker.start_calibration(0.0)
    tracker.update([], embed, FRAME, 1.0)
    res = tracker.update([], embed, FRAME, 99.0)
    assert res.event == 'calibration_failed'
    assert res.state == State.IDLE


def test_target_class_filters_calibration():
    tracker, embed = make({1: LEADER, 2: OTHER}, target_class=0)
    res = calibrate(tracker, embed, [cand(1, x=330, cls=56), cand(2, x=100, cls=0)])
    assert res.leader.track_id == 2


def test_leader_followed_by_track_id():
    tracker, embed = make({1: LEADER, 2: OTHER})
    calibrate(tracker, embed, [cand(1), cand(2, x=100)])
    res = tracker.update([cand(2, x=100), cand(1, x=400)], embed, FRAME, 1.0)
    assert res.state == State.TRACKING
    assert res.leader.track_id == 1


def test_lost_then_reacquired_under_new_id():
    tracker, embed = make({1: LEADER, 2: OTHER, 7: LEADER})
    calibrate(tracker, embed, [cand(1)])
    res = tracker.update([cand(2, x=100)], embed, FRAME, 1.0)
    assert res.event == 'lost'
    # leader comes back with a different track id; needs confirm_frames sightings
    res = tracker.update([cand(2, x=100), cand(7)], embed, FRAME, 2.0)
    assert res.state == State.LOST
    res = tracker.update([cand(2, x=100), cand(7)], embed, FRAME, 2.1)
    assert res.event == 'reacquired'
    assert res.leader.track_id == 7


def test_bystander_is_not_accepted():
    tracker, embed = make({1: LEADER, 2: OTHER})
    calibrate(tracker, embed, [cand(1)])
    tracker.update([], embed, FRAME, 1.0)
    for t in (2.0, 2.1, 2.2, 2.3):
        res = tracker.update([cand(2)], embed, FRAME, t)
        assert res.state == State.LOST


def test_margin_rejects_two_lookalikes():
    tracker, embed = make({1: LEADER, 3: LEADER, 4: LOOKALIKE})
    calibrate(tracker, embed, [cand(1)])
    tracker.update([], embed, FRAME, 1.0)
    for t in (2.0, 2.1, 2.2, 2.3):
        res = tracker.update([cand(3, x=200), cand(4, x=450)], embed, FRAME, t)
        assert res.state == State.LOST


def test_id_swap_detected_by_verification():
    vectors = {1: LEADER}
    tracker, embed = make(vectors, lose_after=2)
    calibrate(tracker, embed, [cand(1)])
    vectors[1] = OTHER           # the detector now gives id 1 to someone else
    res = tracker.update([cand(1)], embed, FRAME, 1.5)
    assert res.state == State.TRACKING            # first bad check is tolerated
    res = tracker.update([cand(1)], embed, FRAME, 3.0)
    assert res.event == 'uncertain'               # robot should stop, not follow blindly
    assert res.reason == 'low_score'


def test_bad_checks_tolerated_up_to_lose_after():
    vectors = {1: LEADER}
    tracker, embed = make(vectors)                # default lose_after = 3
    calibrate(tracker, embed, [cand(1)])
    vectors[1] = OTHER
    for t in (1.5, 3.0):
        assert tracker.update([cand(1)], embed, FRAME, t).state == State.TRACKING
    assert tracker.update([cand(1)], embed, FRAME, 4.5).event == 'uncertain'


VARIANT = unit(1, 0.5, 0)         # same leader, other light: sim to LEADER about 0.89


def test_new_view_is_learned_while_tracking():
    vectors = {1: LEADER}
    tracker, embed = make(vectors)
    calibrate(tracker, embed, [cand(1)])
    vectors[1] = VARIANT
    tracker.update([cand(1)], embed, FRAME, 1.5)
    assert tracker.rolling_count == 1
    assert len(tracker.anchors) == 5              # anchors untouched
    assert tracker.similarity(VARIANT) > 0.99     # now recognised directly


def test_identical_view_is_not_stored_again():
    tracker, embed = make({1: LEADER})
    calibrate(tracker, embed, [cand(1)])
    tracker.update([cand(1)], embed, FRAME, 1.5)
    assert tracker.rolling_count == 0


def test_occluded_or_cut_off_view_is_not_learned():
    vectors = {1: VARIANT}
    tracker, embed = make(vectors)
    calibrate(tracker, embed, [cand(1)])
    # another person's box overlaps the leader
    both = {1: VARIANT, 2: OTHER}
    tracker.update([cand(1), cand(2, x=350)], lambda c: both[c.track_id], FRAME, 1.5)
    assert tracker.rolling_count == 0
    # leader touches the left border of the image
    vectors[1] = VARIANT
    tracker.update([cand(1, x=40)], embed, FRAME, 3.0)
    assert tracker.rolling_count == 0


def test_low_similarity_view_is_not_learned():
    vectors = {1: LEADER}
    tracker, embed = make(vectors)
    calibrate(tracker, embed, [cand(1)])
    vectors[1] = unit(1, 1.5, 0)                   # sim ~0.55, below threshold
    tracker.update([cand(1)], embed, FRAME, 1.5)
    assert tracker.rolling_count == 0


def test_rolling_memory_is_capped_and_cleared_on_recalibration():
    vectors = {1: LEADER}
    tracker, embed = make(vectors, rolling_size=3)
    calibrate(tracker, embed, [cand(1)])
    views = [unit(1, a, b) for a in (0.5, -0.5) for b in (0, 0.5, -0.5)]   # all distinct
    for i, view in enumerate(views):
        vectors[1] = view
        tracker.update([cand(1)], embed, FRAME, 1.5 + i * 1.1)
    assert tracker.rolling_count == 3
    assert len(tracker.anchors) == 5
    calibrate(tracker, embed, [cand(1)], t=100.0)
    assert tracker.rolling_count == 0


def test_load_starts_searching():
    tracker, embed = make({5: LEADER})
    tracker.load(np.stack([LEADER, LEADER]), 0)
    assert tracker.state == State.LOST
    tracker.update([cand(5)], embed, FRAME, 0.0)
    res = tracker.update([cand(5)], embed, FRAME, 0.1)
    assert res.event == 'reacquired'


# --- sticky lock, crossings, motion gate, bystander memory ------------------------------

def uncertain_after_crossing(tracker, embed, t=1.0):
    """Leader (id 1) and bystander (id 2) overlap in the image at the same distance."""
    res = tracker.update([cand(1), cand(2, x=360)], embed, FRAME, t)
    assert res.event == 'uncertain' and res.reason == 'crossing'
    assert res.state == State.UNCERTAIN


def test_moderate_score_does_not_drop_a_continuous_track():
    vectors = {1: LEADER}
    tracker, embed = make(vectors)
    calibrate(tracker, embed, [cand(1)])
    vectors[1] = unit(1, 1.2, 0)                   # sim 0.64: below accept bar, above drop floor
    for t in (1.5, 3.0, 4.5, 6.0):
        res = tracker.update([cand(1)], embed, FRAME, t)
        assert res.state == State.TRACKING


def test_crossing_makes_uncertain_then_leader_is_verified():
    vectors = {1: LEADER, 2: OTHER}
    tracker, embed = make(vectors)
    calibrate(tracker, embed, [cand(1)])
    uncertain_after_crossing(tracker, embed)
    # still crossed: no decision possible
    assert tracker.update([cand(1), cand(2, x=360)], embed, FRAME, 1.1).state == State.UNCERTAIN
    # they separate: the leader is recognised again on the same track
    tracker.update([cand(1, x=150), cand(2, x=500)], embed, FRAME, 2.0)
    res = tracker.update([cand(1, x=150), cand(2, x=500)], embed, FRAME, 2.1)
    assert res.event == 'verified'
    assert res.leader.track_id == 1
    assert tracker.lock_changes == 0


def test_id_swap_in_a_crossing_is_followed_to_the_real_leader():
    vectors = {1: LEADER, 2: OTHER}
    tracker, embed = make(vectors)
    calibrate(tracker, embed, [cand(1)])
    uncertain_after_crossing(tracker, embed)
    vectors[1], vectors[2] = OTHER, LEADER         # the detector swapped the two ids
    tracker.update([cand(1, x=150), cand(2, x=500)], embed, FRAME, 2.0)
    res = tracker.update([cand(1, x=150), cand(2, x=500)], embed, FRAME, 2.1)
    assert res.event == 'switched'
    assert res.leader.track_id == 2
    assert tracker.lock_changes == 1


def test_crossing_with_a_person_far_behind_is_not_an_event():
    tracker, embed = make({1: LEADER, 2: OTHER})
    calibrate(tracker, embed, [cand(1, pos=(0, 0, 2.0))])
    res = tracker.update([cand(1, pos=(0, 0, 2.0)), cand(2, x=360, pos=(0.3, 0, 5.0))],
                         embed, FRAME, 1.0)
    assert res.state == State.TRACKING


def test_position_jump_makes_uncertain():
    tracker, embed = make({1: LEADER})
    calibrate(tracker, embed, [cand(1, pos=(0, 0, 2.0))])
    res = tracker.update([cand(1, pos=(3.0, 0, 4.0))], embed, FRAME, 0.6)
    assert res.event == 'uncertain' and res.reason == 'position_jump'


def test_motion_gate_rejects_a_lookalike_far_from_the_leader():
    vectors = {1: LEADER, 3: LEADER}               # candidate 3 looks exactly like the leader
    tracker, embed = make(vectors)
    calibrate(tracker, embed, [cand(1, pos=(0, 0, 2.0))])
    tracker.update([], embed, FRAME, 0.6)          # leader out of view
    assert tracker.state == State.LOST
    for t in (0.7, 0.8, 0.9, 1.0):
        res = tracker.update([cand(3, pos=(5.0, 0, 8.0))], embed, FRAME, t)
        assert res.state == State.LOST             # 9 m away after 0.5 s: impossible


def test_motion_gate_allows_the_leader_to_return_after_a_while():
    tracker, embed = make({1: LEADER, 3: LEADER})
    calibrate(tracker, embed, [cand(1, pos=(0, 0, 2.0))])
    tracker.update([], embed, FRAME, 0.6)
    tracker.update([cand(3, pos=(4.0, 0, 5.0))], embed, FRAME, 20.0)
    res = tracker.update([cand(3, pos=(4.0, 0, 5.0))], embed, FRAME, 20.1)
    assert res.event == 'reacquired'


def test_known_bystander_returning_alone_is_not_taken_for_the_leader():
    # id 9 is the bystander, whose look is fairly close to the leader's
    near = unit(1, 0.45, 0)                        # about 0.91 similar to LEADER
    vectors = {1: LEADER, 9: near, 12: near}
    tracker, embed = make(vectors, sim_threshold=0.8)
    calibrate(tracker, embed, [cand(1)])
    tracker.observe_other(9, near, 0.5)            # seen next to the leader earlier
    tracker.update([], embed, FRAME, 0.6)          # leader out of view
    for t in (1.0, 1.1, 1.2, 1.3):                 # the bystander comes back under a new id
        assert tracker.update([cand(12)], embed, FRAME, t).state == State.LOST


def test_uncertain_times_out_to_lost():
    tracker, embed = make({1: LEADER, 2: OTHER})
    calibrate(tracker, embed, [cand(1)])
    uncertain_after_crossing(tracker, embed)
    res = tracker.update([cand(1, x=100), cand(2, x=500)], embed, FRAME, 20.0)
    assert res.state == State.LOST and res.reason == 'uncertain_timeout'


def test_colour_helps_when_the_embedding_is_ambiguous():
    from vision.appearance import Appearance
    red = np.zeros((2, 4), dtype=np.float32)
    red[:, 0] = 1.0
    blue = np.zeros((2, 4), dtype=np.float32)
    blue[:, 3] = 1.0
    apps = {1: Appearance(LEADER, red), 2: Appearance(LEADER, blue)}   # same embedding
    tracker = LeaderTracker(calib_frames=5, min_calib_frames=3, confirm_frames=2,
                            reid_period=0.0)
    embed = lambda c: apps[c.track_id]  # noqa: E731
    calibrate(tracker, embed, [cand(1)])
    tracker.update([], embed, FRAME, 0.6)
    res = None
    for t in (1.0, 1.1, 1.2):
        res = tracker.update([cand(1, x=150), cand(2, x=500)], embed, FRAME, t)
    assert res.state == State.TRACKING and res.leader.track_id == 1
