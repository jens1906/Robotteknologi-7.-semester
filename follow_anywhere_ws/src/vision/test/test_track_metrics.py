import pytest

from vision.track_metrics import TrackStats

BOX = (10.0, 20.0, 60.0, 120.0)


def run(frames, dt=0.1, **kw):
    """frames: list of lists of track ids, one list per processed frame."""
    stats = TrackStats(**kw)
    for n, ids in enumerate(frames):
        stats.add(n * dt, [(i, BOX, 0.9) for i in ids])
    return stats


def test_empty_run_is_all_zero():
    s = TrackStats().summary()
    assert s.frames == 0 and s.unique_ids == 0 and s.id_inflation == 0.0


def test_stable_ids_have_no_inflation():
    s = run([[1, 2, 3]] * 20).summary()
    assert s.unique_ids == 3
    assert s.max_people == 3
    assert s.id_inflation == pytest.approx(1.0)
    assert s.births_per_min == 0.0
    assert s.short_tracks == 0


def test_reissued_ids_show_up_as_inflation_and_births():
    # the same 3 people, but every 10 frames the tracker hands out three new IDs
    frames = [[1, 2, 3]] * 10 + [[4, 5, 6]] * 10 + [[7, 8, 9]] * 10
    s = run(frames).summary()
    assert s.unique_ids == 9
    assert s.id_inflation == pytest.approx(3.0)
    assert s.births_per_min == pytest.approx(6 * 60 / 2.9)     # 6 births over 2.9 s
    assert s.median_track_s == pytest.approx(0.9)


def test_flicker_tracks_are_counted_as_short():
    frames = [[1, 2]] * 20
    frames[5] = [1, 2, 99]                                      # lives one frame
    s = run(frames, short_s=0.5).summary()
    assert s.short_tracks == 1


def test_crowded_percentage():
    crowd = list(range(10))
    s = run([crowd] * 3 + [[0, 1]] * 7, crowd_size=10).summary()
    assert s.crowded_pct == pytest.approx(30.0)


def test_gap_keeps_the_same_track_alive():
    s = run([[1], [], [], [1]]).summary()
    assert s.unique_ids == 1
    assert s.births_per_min == 0.0


def test_mot_lines_use_frame_numbers_from_one():
    stats = run([[7], [7, 8]])
    lines = stats.mot_lines()
    assert lines[0] == '1,7,10.0,20.0,50.0,100.0,0.900,-1,-1,-1'
    assert [ln.split(',')[0] for ln in lines] == ['1', '2', '2']


def test_frame_number_and_scale_are_used_for_mot_rows():
    stats = TrackStats()
    stats.add(0.0, [(3, (10.0, 20.0, 60.0, 120.0), 0.8)], frame=17, scale=3.0)
    stats.add(0.1, [], frame=20)
    assert stats.mot_lines() == ['17,3,30.0,60.0,150.0,300.0,0.800,-1,-1,-1']
    assert stats.frame_numbers == [17, 20]
