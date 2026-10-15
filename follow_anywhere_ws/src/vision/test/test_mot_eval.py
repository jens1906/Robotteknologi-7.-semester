import pytest

pytest.importorskip('motmetrics')

from vision.mot_eval import ghost_quality, score  # noqa: E402

H = 100.0


def gt_row(tid, x, h=H, flag=1, cls=1, vis=1.0):
    return (tid, x, 50.0, 40.0, h, flag, cls, vis)


def hyp_row(frame, tid, x, h=H):
    return (frame, tid, x, 50.0, 40.0, h, 0.9)


def gt_of(per_frame):
    """per_frame: {frame: [gt rows]}"""
    return dict(per_frame)


def test_perfect_tracking_scores_one():
    gt = gt_of({f: [gt_row(1, 10.0 * f), gt_row(2, 300.0 + 10.0 * f)] for f in range(1, 7)})
    hyp = [r for f in range(1, 7)
           for r in (hyp_row(f, 11, 10.0 * f), hyp_row(f, 12, 300.0 + 10.0 * f))]
    s = score(gt, hyp, range(1, 7))
    assert s.idf1 == pytest.approx(1.0)
    assert s.switches == 0
    assert s.ids_per_person == pytest.approx(1.0)
    assert s.gt_people == 2


def test_id_change_without_gap_is_one_switch():
    gt = gt_of({f: [gt_row(1, 10.0 * f)] for f in range(1, 7)})
    hyp = [hyp_row(f, 5 if f <= 3 else 9, 10.0 * f) for f in range(1, 7)]
    s = score(gt, hyp, range(1, 7))
    assert s.switches == 1
    assert s.ids_per_person == pytest.approx(2.0)
    assert s.idf1 == pytest.approx(0.5)          # the better of the two IDs covers half the frames


def test_new_id_after_a_gap_is_a_switch_and_a_fragmentation():
    gt = gt_of({f: [gt_row(1, 10.0 * f)] for f in range(1, 9)})
    hyp = ([hyp_row(f, 5, 10.0 * f) for f in (1, 2, 3)]
           + [hyp_row(f, 9, 10.0 * f) for f in (6, 7, 8)])
    s = score(gt, hyp, range(1, 9))
    assert s.switches == 1
    assert s.fragmentations == 1
    assert s.misses == 2


def test_dropped_frames_are_not_scored():
    gt = gt_of({f: [gt_row(1, 10.0 * f)] for f in (1, 2, 3)})
    hyp = [hyp_row(1, 5, 10.0), hyp_row(3, 5, 30.0)]
    s = score(gt, hyp, [1, 3])                 # frame 2 was never processed
    assert s.misses == 0
    assert s.frames == 2


def test_small_ground_truth_is_ignored_and_tracks_on_it_are_not_false_positives():
    gt = gt_of({f: [gt_row(1, 10.0 * f), gt_row(2, 300.0, h=20.0)] for f in range(1, 5)})
    hyp = [r for f in range(1, 5) for r in (hyp_row(f, 7, 10.0 * f), hyp_row(f, 8, 300.0, h=20.0))]
    s = score(gt, hyp, range(1, 5), min_height=40.0)
    assert s.gt_people == 1
    assert s.false_positives == 0
    assert s.misses == 0
    assert s.gt_dropped_pct == pytest.approx(50.0)


def test_track_on_a_distractor_is_not_a_false_positive():
    gt = gt_of({f: [gt_row(1, 10.0 * f), gt_row(9, 300.0, cls=8)] for f in range(1, 5)})
    hyp = [r for f in range(1, 5) for r in (hyp_row(f, 7, 10.0 * f), hyp_row(f, 8, 300.0))]
    s = score(gt, hyp, range(1, 5))
    assert s.false_positives == 0


def test_nothing_to_score_returns_none():
    assert score({}, [], [1, 2]) is None


def test_ghosts_on_unseen_people_are_right_and_duplicates_are_wrong():
    # person 1 is visible, person 2 is hidden (no visible track on them)
    gt = gt_of({f: [gt_row(1, 10.0), gt_row(2, 300.0)] for f in range(1, 5)})
    seen = [hyp_row(f, 7, 10.0) for f in range(1, 5)]
    good = [hyp_row(f, 8, 305.0) for f in range(1, 5)]            # near the hidden person
    dup = [hyp_row(f, 9, 12.0) for f in range(1, 5)]              # on top of the visible one
    far = [hyp_row(f, 6, 600.0) for f in range(1, 5)]             # on nobody
    g = ghost_quality(gt, good + dup + far, seen, range(1, 5))
    assert g.per_frame == 3.0
    assert g.unseen_per_frame == 1.0
    assert g.precision == pytest.approx(1 / 3)
    assert g.recall == pytest.approx(1.0)


def test_no_ghosts_means_zero_recall():
    gt = gt_of({f: [gt_row(2, 300.0)] for f in range(1, 4)})
    g = ghost_quality(gt, [], [], range(1, 4))
    assert g.recall == 0.0 and g.precision == 0.0
