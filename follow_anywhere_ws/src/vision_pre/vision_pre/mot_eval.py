"""Score tracks against MOTChallenge ground truth: IDF1, ID switches, IDs per person. No ROS.

Only the frames the detector actually processed are scored (the others were dropped, the tracker
never saw them). Ground-truth people that are too small or too hidden for the detector are not
counted as misses, and a track on them is not counted as a false positive. The same goes for the
MOT17 distractor classes (person on vehicle, static person, reflection, ...).
"""
import collections
from dataclasses import dataclass
from typing import Optional

import numpy as np

PEDESTRIAN = 1
IOU_MATCH = 0.5


@dataclass
class Score:
    frames: int               # processed frames that were scored
    gt_people: int            # different ground-truth people that counted
    gt_dropped_pct: float     # % of pedestrian boxes left out as too small / too hidden
    idf1: float
    idp: float
    idr: float
    mota: float
    switches: int             # ID switches
    fragmentations: int
    mostly_tracked_pct: float
    ids_per_person: float     # mean number of track IDs that were matched to one person
    misses: int
    false_positives: int
    matches: int

    def text(self):
        return (f'vs ground truth ({self.frames} frames, {self.gt_people} people, '
                f'{self.gt_dropped_pct:.0f}% of boxes too small/hidden left out): '
                f'IDF1 {self.idf1 * 100:.1f}% | ID switches {self.switches} | '
                f'IDs per person {self.ids_per_person:.2f} | '
                f'fragmentations {self.fragmentations} | '
                f'mostly tracked {self.mostly_tracked_pct:.0f}% | '
                f'MOTA {self.mota * 100:.1f}% (misses {self.misses}, false positives '
                f'{self.false_positives})')


def load_gt(path):
    """MOT gt.txt -> {frame: [(id, x, y, w, h, flag, class, visibility), ...]}."""
    gt = collections.defaultdict(list)
    with open(path) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 9:
                continue
            gt[int(p[0])].append((int(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5]),
                                  int(float(p[6])), int(float(p[7])), float(p[8])))
    return gt


def _overlaps(boxes, hyps, mm):
    """True per hypothesis if it overlaps any of `boxes` (x, y, w, h) with IoU of 0.5 or more."""
    if not len(boxes) or not len(hyps):
        return np.zeros(len(hyps), dtype=bool)
    d = mm.distances.iou_matrix(np.asarray(boxes, dtype=float), np.asarray(hyps, dtype=float),
                                max_iou=1.0 - IOU_MATCH)
    return np.isfinite(d).any(axis=0)


def accumulate(gt, hyp_rows, frames, min_height=0.0, min_vis=0.25):
    """Match tracks to ground truth frame by frame. Returns (accumulator, frames scored,
    pedestrian boxes seen, pedestrian boxes left out as too small or hidden)."""
    from vision_pre.venv_fallback import ensure
    mm = ensure('motmetrics')      # only needed here, the node runs without it

    hyp_by_frame = collections.defaultdict(list)
    for frame, tid, x, y, w, h, _ in hyp_rows:
        hyp_by_frame[frame].append((tid, x, y, w, h))

    acc = mm.MOTAccumulator(auto_id=False)
    scored = 0
    pedestrians = dropped = 0
    for frame in sorted(set(frames)):
        keep, ignore = [], []
        for tid, x, y, w, h, flag, cls, vis in gt.get(frame, []):
            counts = flag == 1 and cls == PEDESTRIAN and vis >= min_vis and h >= min_height
            if flag == 1 and cls == PEDESTRIAN:
                pedestrians += 1
                dropped += not counts
            (keep if counts else ignore).append((tid, x, y, w, h))
        hyps = hyp_by_frame.get(frame, [])
        hyp_boxes = [h[1:] for h in hyps]
        # a track on something that is not scored (small, hidden, distractor) is not an error
        stray = _overlaps([k[1:] for k in ignore], hyp_boxes, mm) \
            & ~_overlaps([k[1:] for k in keep], hyp_boxes, mm)
        hyps = [h for h, s in zip(hyps, stray) if not s]
        d = (mm.distances.iou_matrix(np.asarray([k[1:] for k in keep], dtype=float),
                                     np.asarray([h[1:] for h in hyps], dtype=float),
                                     max_iou=1.0 - IOU_MATCH)
             if keep and hyps else np.empty((len(keep), len(hyps))))
        acc.update([k[0] for k in keep], [h[0] for h in hyps], d, frameid=frame)
        scored += 1
    return acc, scored, pedestrians, dropped


def score(gt, hyp_rows, frames, min_height=0.0, min_vis=0.25) -> Optional[Score]:
    """Score MOT-style rows (frame, id, x, y, w, h, conf) over the given processed frames.

    min_height / min_vis: ground-truth boxes below these (height in the pixels of the boxes) are
    ignored. Returns None when there is no ground truth to score in those frames.
    """
    from vision_pre.venv_fallback import ensure
    mm = ensure('motmetrics')

    acc, scored, pedestrians, dropped = accumulate(gt, hyp_rows, frames, min_height, min_vis)
    if not scored or acc.mot_events.empty:
        return None
    names = ['idf1', 'idp', 'idr', 'mota', 'num_switches', 'num_fragmentations',
             'num_unique_objects', 'mostly_tracked', 'num_misses', 'num_false_positives',
             'num_matches', 'num_objects']
    m = mm.metrics.create().compute(acc, metrics=names, name='run').loc['run']
    if not m['num_objects']:
        return None
    events = acc.mot_events
    matched = events[events['Type'].isin(['MATCH', 'SWITCH'])]
    per_person = matched.groupby('OId')['HId'].nunique()
    people = int(m['num_unique_objects'])
    return Score(
        frames=scored, gt_people=people,
        gt_dropped_pct=100.0 * dropped / pedestrians if pedestrians else 0.0,
        idf1=float(m['idf1']), idp=float(m['idp']), idr=float(m['idr']), mota=float(m['mota']),
        switches=int(m['num_switches']), fragmentations=int(m['num_fragmentations']),
        mostly_tracked_pct=100.0 * float(m['mostly_tracked']) / people if people else 0.0,
        ids_per_person=float(per_person.mean()) if len(per_person) else 0.0,
        misses=int(m['num_misses']), false_positives=int(m['num_false_positives']),
        matches=int(m['num_matches']))


@dataclass
class GhostScore:
    per_frame: float          # remembered-but-hidden people reported per frame
    unseen_per_frame: float   # scored people nobody sees (no visible track on them) per frame
    precision: float          # share of reported hidden people that really are such a person
    recall: float             # share of those unseen people that were reported

    def text(self):
        return (f'hidden people: {self.per_frame:.1f} reported per frame vs '
                f'{self.unseen_per_frame:.1f} unseen real people | precision '
                f'{self.precision * 100:.0f}% | found {self.recall * 100:.0f}% of the unseen')


def ghost_quality(gt, ghost_rows, hyp_rows, frames, min_height=0.0, min_vis=0.25,
                  iou=0.2) -> Optional[GhostScore]:
    """Judge the hidden people the tracker reports (rows like the tracks: frame, id, x, y, w, h).

    An unseen person is a scored ground-truth person without a visible track on them. A reported
    hidden person is right when it overlaps one (the position is only a prediction, so the overlap
    needed is small); one sitting on somebody who is visible anyway is a duplicate and wrong.
    """
    from vision_pre.venv_fallback import ensure
    mm = ensure('motmetrics')

    by_frame = collections.defaultdict(list)
    for frame, _, x, y, w, h, _ in ghost_rows:
        by_frame[frame].append((x, y, w, h))
    seen = collections.defaultdict(list)
    for frame, _, x, y, w, h, _ in hyp_rows:
        seen[frame].append((x, y, w, h))

    def covers(boxes, others, thr):
        if not boxes or not others:
            return np.zeros(len(boxes), dtype=bool)
        d = mm.distances.iou_matrix(np.asarray(boxes, dtype=float),
                                    np.asarray(others, dtype=float), max_iou=1.0 - thr)
        return np.isfinite(d).any(axis=1)

    ghosts = unseen = good = found = n = 0
    for frame in sorted(set(frames)):
        people = [(x, y, w, h) for _, x, y, w, h, flag, cls, vis in gt.get(frame, [])
                  if flag == 1 and cls == PEDESTRIAN and vis >= min_vis
                  and h >= min_height]
        lost = [p for p, c in zip(people, covers(people, seen[frame], IOU_MATCH)) if not c]
        gh = by_frame.get(frame, [])
        right = covers(gh, lost, iou) & ~covers(gh, seen[frame], 0.5)
        n += 1
        ghosts += len(gh)
        unseen += len(lost)
        good += int(right.sum())
        found += int(covers(lost, gh, iou).sum())
    if not n:
        return None
    return GhostScore(per_frame=ghosts / n, unseen_per_frame=unseen / n,
                      precision=good / ghosts if ghosts else 0.0,
                      recall=found / unseen if unseen else 0.0)
