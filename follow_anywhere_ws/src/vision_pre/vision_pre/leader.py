"""Who is the leader? Enroll a person, follow them by track id, find them again when the track breaks.

Precision first: `leader_id` is only given while the identity is certain. A crossing, a lost
track or a different-looking person ends it. The leader is picked again only when one candidate is
clearly closest to the stored views, several checks in a row.

No ROS, no torch: embeddings come in through the `embed(ids) -> (emb, visible)` callable that
update() is given, and are only requested when needed (embedding is the expensive part).
"""
import os
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from kpr.match import part_distance

ENROLL_VIEWS = 8       # views stored when enrolling
ENROLL_GAP = 0.4       # s between two stored views, so that they differ a little
ENROLL_TIMEOUT = 20.0  # s
GRACE = 1.0            # s a locked leader may be missing from the detections (flicker) before they are lost
ENROLL_GRACE = 2.0     # s the person being enrolled may be missing before enrolment fails
REBIND_IOU = 0.6       # a new track this much on the enrolled person's last box is the same person (id change)
MIN_PARTS = 4          # body parts a view must show to be stored
ROLLING_VIEWS = 20     # views learned while following, the oldest go first
NOVELTY = 0.25         # a view closer than this to a stored one is not stored again
CROSSING_IOU = 0.15    # another person's box over the leader's: the identity is doubtful
MAX_CANDIDATES = 4     # people compared while searching (the biggest ones)
SEARCH_PERIOD = 0.3    # s between two searches
CONFIRM_WINDOW = 6     # a candidate's last comparisons that are counted when confirming it
COMPETITOR_AGE = 3.0   # s a visible person's latest distance still counts as a competitor
LOSE_AFTER = 2         # checks in a row that say 'different person' before the leader is doubted


class State(str, Enum):
    IDLE = 'IDLE'              # no leader yet
    ENROLLING = 'ENROLLING'    # collecting views of the chosen person
    FOLLOWING = 'FOLLOWING'    # leader identified, locked to a track id
    SEARCHING = 'SEARCHING'    # leader known, but not identified right now


@dataclass
class Result:
    state: State
    leader_id: Optional[int] = None    # track id of the leader, only while FOLLOWING
    event: Optional[str] = None        # enrolled, enroll_failed, lost, found
    reason: Optional[str] = None


def _iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _usable(visible):
    """Does a view show enough of the body to be stored?"""
    return len(visible) == 1 or visible[1:].sum() >= MIN_PARTS


class Leader:
    def __init__(self, accept=0.52, margin=0.05, drop=0.80, confirm=3, verify_period=1.0,
                 overlap_ok=False):
        self.accept = accept              # distance below which a candidate can be the leader
        self.margin = margin              # lead the best candidate needs over the second best
        self.drop = drop                  # distance above which the locked track looks wrong
        self.confirm = confirm            # of a candidate's last CONFIRM_WINDOW comparisons that must look like the leader
        self.verify_period = verify_period    # s between checks of the locked track
        self.overlap_ok = overlap_ok          # embed people that others stand over (needs KPR's negative prompts)
        self.scores = {}                  # track id -> (time, distance) of the latest comparison
        self.reset()

    def reset(self):
        self.state, self.track_id = State.IDLE, None
        self._anchors = None              # (emb (G, P, D), visible (G, P)): enrolment views, kept
        self._rolling = deque(maxlen=ROLLING_VIEWS)
        self._views, self._enroll_id = [], None
        self._t_enroll = self._t_sample = self._t_verify = self._t_search = 0.0
        self._t_seen, self._absent = 0.0, None     # last time the enrolled person was seen / since when the leader is missing
        self._enroll_box, self._rebound = None, False
        self._bad, self._evidence = 0, {}      # _evidence: track id -> did its last comparisons look like the leader
        self.scores.clear()

    @property
    def n_views(self):
        return len(self._views) if self._anchors is None else len(self._anchors[0]) + len(self._rolling)

    def enroll(self, track_id, now):
        """Collect views of this person; they become the leader. Replaces any earlier leader."""
        self.reset()
        self.state, self._enroll_id, self._t_enroll, self._t_seen = State.ENROLLING, track_id, now, now

    def save(self, path, tag):
        """Write the enrolment views (not the rolling ones) so that the next run knows the leader.
        tag names the embedder, so views are never used with another model."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        np.savez(path, emb=self._anchors[0], vis=self._anchors[1], tag=tag)

    def load(self, path, tag):
        """Use views saved earlier and search for that person. False if there is no usable file."""
        try:
            data = np.load(path, allow_pickle=False)                 # plain arrays only, never a pickle
            if str(data['tag']) != tag:
                return False
            emb, visible = data['emb'].astype(np.float32), data['vis'].astype(bool)
        except (OSError, KeyError, ValueError):
            return False
        self.reset()
        self._anchors, self.state = (emb, visible), State.SEARCHING
        return True

    def new_scene(self):
        """The video jumped back (loop): old track ids mean nothing. Keep the leader's views, search again."""
        if self._anchors is None:
            self.reset()
        else:
            self.state, self.track_id, self._absent, self._bad = State.SEARCHING, None, None, 0
            self._evidence.clear()

    def update(self, people, embed, now):
        """One step. people: this frame's tracked people (id, box, area). embed(ids) -> (emb, visible)."""
        if self.state == State.ENROLLING:
            return self._enrolling(people, embed, now)
        if self.state == State.FOLLOWING:
            return self._following(people, embed, now)
        if self.state == State.SEARCHING:
            return self._searching(people, embed, now)
        return Result(State.IDLE)

    # helpers -------------------------------------------------------------
    def _overlapped(self, person, people):
        return any(o.id != person.id and _iou(person.box, o.box) > CROSSING_IOU for o in people)

    def _distance(self, emb, visible):
        """Distance to the closest stored view; NaN when no view shares enough body parts."""
        gallery, gallery_vis = self._anchors
        if self._rolling:
            gallery = np.concatenate([gallery, np.stack([e for e, _ in self._rolling])])
            gallery_vis = np.concatenate([gallery_vis, np.stack([v for _, v in self._rolling])])
        d = part_distance(emb, visible, gallery, gallery_vis)
        return float('nan') if np.isnan(d).all() else float(np.nanmin(d))

    def _lose(self, reason):
        self.state, self.track_id, self._bad, self._absent = State.SEARCHING, None, 0, None
        self._evidence.clear()
        return Result(State.SEARCHING, event='lost', reason=reason)

    # states --------------------------------------------------------------
    def _enrolling(self, people, embed, now):
        person = next((p for p in people if p.id == self._enroll_id), None)
        if person is None and self._enroll_box is not None:      # the tracker may have given them a new id
            near = max(people, key=lambda p: _iou(p.box, self._enroll_box), default=None)
            if near is not None and _iou(near.box, self._enroll_box) >= REBIND_IOU:
                person, self._enroll_id, self._rebound = near, near.id, True
        if person is not None:
            self._t_seen, self._enroll_box = now, person.box
        if now - self._t_seen > ENROLL_GRACE or now - self._t_enroll > ENROLL_TIMEOUT:
            gone = now - self._t_seen > ENROLL_GRACE
            self.reset()
            return Result(State.IDLE, event='enroll_failed', reason='person gone' if gone else 'timeout')
        if person is None:
            return Result(State.ENROLLING)
        if now - self._t_sample < ENROLL_GAP or self._overlapped(person, people):
            return Result(State.ENROLLING)             # wait for a clean view
        emb, visible = embed([person.id])
        self._t_sample = now
        if self._rebound and self._views:             # after an id change: is it still the same person?
            known = (np.stack([e for e, _ in self._views]), np.stack([v for _, v in self._views]))
            d = part_distance(emb[0], visible[0], *known)
            if not np.all(np.isnan(d)) and np.nanmin(d) > self.drop:
                self.reset()
                return Result(State.IDLE, event='enroll_failed', reason='person changed')
        self._rebound = False
        if _usable(visible[0]):
            self._views.append((emb[0], visible[0]))
        if len(self._views) < ENROLL_VIEWS:
            return Result(State.ENROLLING)
        self._anchors = (np.stack([e for e, _ in self._views]), np.stack([v for _, v in self._views]))
        self.state, self.track_id, self._t_verify, self._absent = State.FOLLOWING, person.id, now, None
        return Result(State.FOLLOWING, person.id, event='enrolled')

    def _following(self, people, embed, now):
        person = next((p for p in people if p.id == self.track_id), None)
        if person is None:
            if self._absent is None:
                self._absent = now
            if now - self._absent > GRACE:
                return self._lose('track gone')
            return Result(State.FOLLOWING)             # out of view for a moment: nobody is flagged
        returned, self._absent = self._absent is not None, None
        crossed = self._overlapped(person, people)
        if crossed and not self.overlap_ok:
            return self._lose('crossing')
        if returned or crossed or now - self._t_verify >= self.verify_period:   # checked on every frame while crossed
            emb, visible = embed([person.id])
            self._t_verify = now
            d = self._distance(emb[0], visible[0])
            self.scores[person.id] = (now, d)
            if crossed or returned:
                if not d < self.accept:                # after a crossing or a gap only a clear match keeps the leader
                    return self._lose('crossing, not sure' if crossed else 'not sure after a gap')
            elif d > self.drop:                        # NaN (unknown) is neither this nor the next
                self._bad += 1
                if self._bad >= LOSE_AFTER:
                    return self._lose('looks different')
            elif d == d:
                self._bad = 0
                if NOVELTY < d < self.accept and _usable(visible[0]):
                    self._rolling.append((emb[0], visible[0]))
        return Result(State.FOLLOWING, person.id)

    def _searching(self, people, embed, now):
        if now - self._t_search < SEARCH_PERIOD:
            return Result(State.SEARCHING)
        # somebody standing in front of a person says nothing about who they are (unless KPR was told)
        pool = [p for p in people if self.overlap_ok or not self._overlapped(p, people)]
        if not pool:
            return Result(State.SEARCHING)

        def priority(p):                    # promising people first, then whoever was compared longest ago
            seen, d = self.scores.get(p.id, (-1e9, None))
            promising = d is not None and d < self.accept and now - seen < COMPETITOR_AGE
            return (not promising, seen, -p.area)

        pool = sorted(pool, key=priority)[:MAX_CANDIDATES]
        self._t_search = now
        emb, visible = embed([p.id for p in pool])
        dist = {p.id: self._distance(e, v) for p, e, v in zip(pool, emb, visible)}
        for pid, d in dist.items():
            self.scores[pid] = (now, d)
        visible_ids = {p.id for p in people}
        self._evidence = {i: e for i, e in self._evidence.items() if i in visible_ids}
        for pid, d in dist.items():
            # the closest other person we know of right now: the leader must be clearly closer than that
            others = [od for i, (t, od) in self.scores.items()
                      if i != pid and i in visible_ids and now - t < COMPETITOR_AGE and od == od]
            win = bool(d < self.accept and min(others, default=np.inf) - d >= self.margin)
            self._evidence.setdefault(pid, deque(maxlen=CONFIRM_WINDOW)).append(win)
        confirmed = [(dist.get(i, np.inf), i) for i, e in self._evidence.items() if sum(e) >= self.confirm]
        if confirmed:
            winner = min(confirmed)[1]
            self.state, self.track_id, self._t_verify, self._bad, self._absent = State.FOLLOWING, winner, now, 0, None
            self._evidence.clear()
            return Result(State.FOLLOWING, winner, event='found')
        return Result(State.SEARCHING)
