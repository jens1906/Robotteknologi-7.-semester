"""Leader state machine: calibration, matching, a sticky lock and re-acquisition.

No ROS or torch in here, so it can be unit tested with synthetic data.

Identity rules (why the lock does not wander):
  * While the leader's track is continuous and not crossed by anybody, it is trusted.
  * A crossing (overlap with someone at a similar distance), a position jump, or a very
    low score makes the state UNCERTAIN. The robot should stop/slow until the leader is
    positively found again.
  * Finding the leader needs the best score above a threshold, a margin over the other
    candidates, being closer to the leader than to known bystanders, a plausible position
    and several confirmations in a row.
"""
import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple

import numpy as np

from vision_pre.appearance import Appearance, colour_sim
from vision_pre.motion import LeaderMotion


class State(str, Enum):
    IDLE = 'IDLE'                # no leader known
    CALIBRATING = 'CALIBRATING'  # collecting views of the leader
    TRACKING = 'TRACKING'        # leader visible and followed by track id
    UNCERTAIN = 'UNCERTAIN'      # leader probably in view, identity must be re-verified
    LOST = 'LOST'                # leader known but not visible, searching


@dataclass
class Candidate:
    track_id: int
    cls: int
    box: Tuple[float, float, float, float]   # x1, y1, x2, y2 in pixels
    conf: float = 1.0
    pos: Optional[Tuple[float, float, float]] = None   # x, y, z in metres (camera frame)

    @property
    def center(self):
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2

    @property
    def area(self):
        return max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1])


@dataclass
class Score:
    fused: float                     # what decisions use
    emb: float                       # embedding cosine similarity
    col: Optional[float] = None      # clothing colour similarity, if available
    moved: bool = False              # outside the area the leader could have walked to


@dataclass
class Result:
    state: State
    leader: Optional[Candidate] = None
    sim: Optional[float] = None       # fused similarity of the leader / best candidate
    event: Optional[str] = None       # calibrated, calibration_failed, lost, uncertain,
    #                                   reacquired, verified, switched
    reason: Optional[str] = None


def _app(x):
    return x if isinstance(x, Appearance) else Appearance(np.asarray(x, dtype=np.float32))


class LeaderTracker:
    def __init__(self, target_class=0, sim_threshold=0.75, margin=0.05, other_margin=0.02,
                 calib_frames=30, min_calib_frames=5, calib_timeout=15.0,
                 confirm_frames=3, verify_period=1.0, reid_period=0.3, max_candidates=4,
                 lose_after=3, drop_floor=0.5, rolling_size=30, update_max_sim=0.97,
                 occlusion_iou=0.05, edge_margin=3,
                 w_emb=0.8, w_col=0.2,
                 uncertain_on_overlap=True, event_iou=0.15, depth_gap=0.8,
                 uncertain_timeout=5.0, motion_gate=True, max_speed=1.5, gate_radius=0.6):
        self.target_class = target_class          # -1 = any class
        self.sim_threshold = sim_threshold        # fused score needed to accept a candidate
        self.margin = margin                      # lead needed over the next best candidate
        self.other_margin = other_margin          # lead needed over known bystanders
        self.calib_frames = calib_frames
        self.min_calib_frames = min_calib_frames
        self.calib_timeout = calib_timeout
        self.confirm_frames = confirm_frames
        self.verify_period = verify_period
        self.reid_period = reid_period
        self.max_candidates = max_candidates
        self.lose_after = lose_after              # very low scores in a row before UNCERTAIN
        self.drop_floor = drop_floor              # below this the track is not the leader
        self.update_max_sim = update_max_sim      # above this a view adds nothing new
        self.occlusion_iou = occlusion_iou        # overlap that makes a view unfit to learn
        self.edge_margin = edge_margin            # px, box touching the border is cut off
        self.w_emb = w_emb
        self.w_col = w_col
        self.uncertain_on_overlap = uncertain_on_overlap
        self.event_iou = event_iou                # overlap that counts as a crossing
        self.depth_gap = depth_gap                # m, people further apart are not a crossing
        self.uncertain_timeout = uncertain_timeout
        self.motion_gate = motion_gate
        self.motion = LeaderMotion(max_speed, gate_radius)

        self.state = State.IDLE
        # anchors = calibration views, never evicted. rolling = views learned while
        # tracking, oldest dropped first, so the memory follows light/clothing changes.
        self.anchors: Optional[np.ndarray] = None        # N x D, unit vectors
        self.anchor_cols: Optional[np.ndarray] = None    # N x 2 x B, or None
        self._rolling = deque(maxlen=rolling_size)
        self._rolling_cols = deque(maxlen=rolling_size)
        self._others = {}                                # track_id -> (time, unit vector)
        self.leader_cls: Optional[int] = None
        self.locked_id: Optional[int] = None
        self.last_sim: Optional[float] = None
        self.lock_changes = 0                            # times the lock moved to another track
        self._last_leader_id: Optional[int] = None
        self._frame_size = (0, 0)

        self._calib_apps = []
        self._calib_t0 = 0.0
        self._calib_id: Optional[int] = None
        self._calib_cls: Optional[int] = None
        self._last_verify = 0.0
        self._last_reid = 0.0
        self._uncertain_t0 = 0.0
        self._bad_verifies = 0
        self._pending_id: Optional[int] = None
        self._pending_count = 0

    # properties and scoring ---------------------------------------------
    @property
    def gallery(self) -> Optional[np.ndarray]:
        """Embeddings the leader is compared against: anchors + rolling views."""
        if self.anchors is None:
            return None
        if not self._rolling:
            return self.anchors
        return np.vstack([self.anchors, np.stack(self._rolling)])

    @property
    def rolling_count(self):
        return len(self._rolling)

    def similarity(self, emb) -> float:
        """Best embedding cosine similarity to the leader (0 if none known)."""
        gallery = self.gallery
        return 0.0 if gallery is None else float(np.max(gallery @ emb))

    def _gallery_cols(self):
        n_anchor = len(self.anchors)
        rows = list(self.anchor_cols) if self.anchor_cols is not None else [None] * n_anchor
        return rows + list(self._rolling_cols)

    def score(self, app, cand=None, now=0.0) -> Score:
        """Fused similarity of one view to the leader, best over all stored views."""
        gallery = self.gallery
        if gallery is None:
            return Score(0.0, 0.0)
        app = _app(app)
        e = gallery @ app.emb
        col = np.full(len(e), np.nan)
        if self.w_col > 0 and app.col is not None:
            rows = self._gallery_cols()
            have = [i for i, r in enumerate(rows) if r is not None]
            if have:
                col[have] = colour_sim(app.col, np.stack([rows[i] for i in have]))
        fused = np.where(np.isnan(col), e,
                         (self.w_emb * e + self.w_col * np.nan_to_num(col))
                         / (self.w_emb + self.w_col))
        best = int(np.argmax(fused))
        moved = bool(self.motion_gate and cand is not None and cand.pos is not None
                     and not self.motion.allowed(cand.pos, now))
        return Score(float(fused[best]), float(e[best]),
                     None if np.isnan(col[best]) else float(col[best]), moved)

    def observe_other(self, track_id, app, now):
        """Remember what a bystander looks like, so a lookalike can be told apart."""
        self._others[track_id] = (now, _app(app).emb)
        if len(self._others) > 40:
            oldest = min(self._others, key=lambda k: self._others[k][0])
            del self._others[oldest]

    def _bystander_sim(self, emb, now):
        sims = [float(v @ emb) for t, v in self._others.values() if now - t < 60.0]
        return max(sims) if sims else -1.0

    # control -------------------------------------------------------------
    def load(self, gallery, cls, col=None):
        """Use a previously saved leader; start searching for it."""
        self.anchors = np.asarray(gallery, dtype=np.float32)
        self.anchor_cols = None if col is None else np.asarray(col, dtype=np.float32)
        self._clear_memory()
        self.leader_cls = int(cls)
        self._to_lost()

    def reset(self):
        self.state = State.IDLE
        self.anchors = None
        self.anchor_cols = None
        self._clear_memory()
        self.leader_cls = None
        self.locked_id = None
        self.last_sim = None

    def _clear_memory(self):
        self._rolling.clear()
        self._rolling_cols.clear()
        self._others.clear()
        self.motion.reset()

    def start_calibration(self, now):
        self.state = State.CALIBRATING
        self.locked_id = None
        self._calib_apps = []
        self._calib_t0 = now
        self._calib_id = None
        self._calib_cls = None
        self.motion.reset()

    # main entry ----------------------------------------------------------
    def update(self, candidates, embed_fn: Callable[[Candidate], Appearance],
               frame_size, now) -> Result:
        """Advance one step. embed_fn is only called when an embedding is needed."""
        self._frame_size = frame_size
        if self.state == State.CALIBRATING:
            return self._calibrate(candidates, embed_fn, frame_size, now)
        if self.state == State.TRACKING:
            return self._track(candidates, embed_fn, now)
        if self.state in (State.LOST, State.UNCERTAIN):
            return self._search(candidates, embed_fn, now)
        return Result(State.IDLE)

    # helpers -------------------------------------------------------------
    @staticmethod
    def _iou(a, b):
        iw = min(a[2], b[2]) - max(a[0], b[0])
        ih = min(a[3], b[3]) - max(a[1], b[1])
        if iw <= 0 or ih <= 0:
            return 0.0
        inter = iw * ih
        area = lambda r: (r[2] - r[0]) * (r[3] - r[1])  # noqa: E731
        return inter / (area(a) + area(b) - inter)

    def _overlapped(self, cand, cands, iou):
        """Is another person's box over this one, at a similar distance?"""
        for o in cands:
            if o.track_id == cand.track_id or self._iou(cand.box, o.box) <= iou:
                continue
            if (cand.pos is not None and o.pos is not None
                    and abs(cand.pos[2] - o.pos[2]) > self.depth_gap):
                continue                                   # far apart in depth: not a crossing
            return True
        return False

    def _clean_view(self, cur, cands):
        """True if the leader is fully visible: not cut by the border, not overlapped."""
        w, h = self._frame_size
        x1, y1, x2, y2 = cur.box
        m = self.edge_margin
        if x1 <= m or y1 <= m or (w and x2 >= w - m) or (h and y2 >= h - m):
            return False
        return not self._overlapped(cur, cands, self.occlusion_iou)

    def _maybe_learn(self, app, sim, cur, cands):
        """Remember this view if it is clearly the leader, new, and not occluded."""
        if (self.sim_threshold <= sim < self.update_max_sim
                and self._clean_view(cur, cands)):
            self._rolling.append(app.emb)
            self._rolling_cols.append(app.col)

    def _to_lost(self):
        if self.locked_id is not None:
            self._last_leader_id = self.locked_id
        self.state = State.LOST
        self.locked_id = None
        self._pending_id = None
        self._pending_count = 0
        self._last_reid = 0.0

    def _to_uncertain(self, now, reason, sim=None):
        self._to_lost()
        self.state = State.UNCERTAIN
        self._uncertain_t0 = now
        return Result(State.UNCERTAIN, sim=sim, event='uncertain', reason=reason)

    # calibration ---------------------------------------------------------
    def _calibrate(self, cands, embed_fn, frame_size, now):
        if now - self._calib_t0 > self.calib_timeout:
            return self._finish_calibration(now)

        pool = [c for c in cands
                if self.target_class < 0 or c.cls == self.target_class]
        if self._calib_id is not None:
            # once picked, only follow that same person/object
            pool = [c for c in pool if c.track_id == self._calib_id]
        if not pool:
            return Result(State.CALIBRATING)

        cx, cy = frame_size[0] / 2, frame_size[1] / 2
        pick = min(pool, key=lambda c: math.hypot(c.center[0] - cx, c.center[1] - cy))
        self._calib_id = pick.track_id
        self._calib_cls = pick.cls
        self._calib_apps.append(_app(embed_fn(pick)))
        self.motion.update(pick.pos, now)

        if len(self._calib_apps) >= self.calib_frames:
            return self._finish_calibration(now, pick)
        return Result(State.CALIBRATING, pick)

    def _finish_calibration(self, now, pick=None):
        if len(self._calib_apps) < self.min_calib_frames:
            # keep any earlier leader if calibration did not work out
            if self.anchors is not None:
                self._to_lost()
            else:
                self.state = State.IDLE
            return Result(self.state, event='calibration_failed')

        self.anchors = np.stack([a.emb for a in self._calib_apps]).astype(np.float32)
        cols = [a.col for a in self._calib_apps]
        self.anchor_cols = None if any(c is None for c in cols) else np.stack(cols)
        self._rolling.clear()
        self._rolling_cols.clear()
        self._others.clear()
        self.leader_cls = self._calib_cls
        self.locked_id = self._calib_id
        self._last_leader_id = self._calib_id
        self.last_sim = 1.0
        self.state = State.TRACKING
        self._last_verify = now
        self._bad_verifies = 0
        return Result(State.TRACKING, pick, 1.0, 'calibrated')

    # tracking ------------------------------------------------------------
    def _track(self, cands, embed_fn, now):
        cur = next((c for c in cands
                    if c.track_id == self.locked_id and c.cls == self.leader_cls), None)
        if cur is None:
            self._to_lost()
            return Result(State.LOST, event='lost', reason='track_gone')

        # events that make the identity doubtful (no embedding needed)
        if self.uncertain_on_overlap and self._overlapped(cur, cands, self.event_iou):
            return self._to_uncertain(now, 'crossing', self.last_sim)
        if (self.motion_gate and cur.pos is not None
                and not self.motion.allowed(cur.pos, now)):
            return self._to_uncertain(now, 'position_jump', self.last_sim)
        self.motion.update(cur.pos, now)

        if now - self._last_verify >= self.verify_period:
            app = _app(embed_fn(cur))
            s = self.score(app, cur, now)
            self.last_sim = s.fused
            self._last_verify = now
            if s.fused < self.drop_floor:
                # the track is continuous, so only a clearly wrong look counts, several times
                self._bad_verifies += 1
                if self._bad_verifies >= self.lose_after:
                    return self._to_uncertain(now, 'low_score', s.fused)
            else:
                self._bad_verifies = 0
                self._maybe_learn(app, s.fused, cur, cands)
        return Result(State.TRACKING, cur, self.last_sim)

    # searching (LOST and UNCERTAIN) --------------------------------------
    def _search(self, cands, embed_fn, now):
        if self.state == State.UNCERTAIN and now - self._uncertain_t0 > self.uncertain_timeout:
            self._to_lost()
            return Result(State.LOST, event='lost', reason='uncertain_timeout')
        if now - self._last_reid < self.reid_period:
            return Result(self.state)

        # views where somebody is in front of the person say nothing about identity
        pool = sorted((c for c in cands
                       if c.cls == self.leader_cls
                       and not self._overlapped(c, cands, self.event_iou)),
                      key=lambda c: c.area, reverse=True)[:self.max_candidates]
        if not pool:
            self._pending_id, self._pending_count = None, 0
            return Result(self.state)

        self._last_reid = now
        scored = []
        for c in pool:
            app = _app(embed_fn(c))
            scored.append((self.score(app, c, now), c, app))
        eligible = sorted((t for t in scored if not t[0].moved),
                          key=lambda t: t[0].fused, reverse=True)
        if not eligible:
            self._pending_id, self._pending_count = None, 0
            return Result(self.state, sim=max(t[0].fused for t in scored))

        best_s, best_c, best_app = eligible[0]
        second = eligible[1][0].fused if len(eligible) > 1 else -1.0
        accepted = (best_s.fused >= self.sim_threshold
                    and best_s.fused - second >= self.margin
                    and best_s.emb - self._bystander_sim(best_app.emb, now) >= self.other_margin)
        if not accepted:
            self._pending_id, self._pending_count = None, 0
            return Result(self.state, sim=best_s.fused)

        if best_c.track_id == self._pending_id:
            self._pending_count += 1
        else:
            self._pending_id, self._pending_count = best_c.track_id, 1
        if self._pending_count < self.confirm_frames:
            return Result(self.state, sim=best_s.fused)

        was = self.state
        changed = (self._last_leader_id is not None
                   and best_c.track_id != self._last_leader_id)
        self.state = State.TRACKING
        self.locked_id = best_c.track_id
        self.last_sim = best_s.fused
        self._last_verify = now
        self._bad_verifies = 0
        self.motion.update(best_c.pos, now)
        if changed:
            self.lock_changes += 1
        self._last_leader_id = best_c.track_id
        if was == State.UNCERTAIN and not changed:
            event = 'verified'
        elif was == State.UNCERTAIN:
            event = 'switched'
        else:
            event = 'reacquired'
        return Result(State.TRACKING, best_c, best_s.fused, event)
