"""Short-term identity relinking: someone hidden who comes back under a new tracker id gets their
old id back. No ROS, no torch, so it can be unit tested with synthetic data.

The tracker (ByteTrack / BoT-SORT) only knows boxes and motion, so after an occlusion it often
starts a new track. Here the appearance (an OSNet embedding) is compared, but only occasionally: a
person is described every `refresh_s` seconds while clearly visible, and a new tracker id is
described a few times when it appears. That is a few embeddings per second instead of one per
person per frame.

A new id is matched only against people who are hidden right now (listed by the tracker), only if
it appeared where they could have walked to, and with a clear lead over the next best candidate.
"""
from collections import deque

import numpy as np


def _center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _height(box):
    return max(1.0, box[3] - box[1])


def _iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])  # noqa: E731
    return inter / (area(a) + area(b) - inter)


class Relinker:
    def __init__(self, keep_s=10.0, sim_threshold=0.7, margin=0.05, refresh_s=1.5,
                 min_age=3, tries=3, retry_s=0.3, max_embeds=2, walk_heights_per_s=0.8,
                 gate_heights=1.0, gallery_size=6, edge_margin=3, clean_iou=0.05):
        self.keep_s = keep_s                  # s a hidden person can be relinked
        self.sim_threshold = sim_threshold    # cosine similarity needed to relink
        self.margin = margin                  # lead needed over the next best hidden person
        self.refresh_s = refresh_s            # s between descriptions of a visible person
        self.min_age = min_age                # frames a new tracker id exists before it is tried
        self.tries = tries                    # descriptions of a new id before giving up
        self.retry_s = retry_s                # s between those
        self.max_embeds = max_embeds          # embeddings per frame, they cost ~5 ms each on a CPU
        self.walk = walk_heights_per_s        # how far a person may have walked, in body heights/s
        self.gate = gate_heights              # plus this much slack
        self.gallery_size = gallery_size
        self.edge_margin = edge_margin
        self.clean_iou = clean_iou
        self.alias = {}                       # tracker id -> published id (only where they differ)
        self.relinks = 0                      # how many ids were given back, for the report
        self._gallery = {}                    # published id -> deque of unit vectors
        self._last_embed = {}                 # published id -> time of the last description
        self._age = {}                        # tracker id -> frames seen
        self._tries = {}                      # tracker id -> (attempts, time of the last one)
        self._settled = set()                 # tracker ids that are decided (relinked or given up)

    def published(self, tracker_id):
        return self.alias.get(tracker_id, tracker_id)

    def _clean(self, box, others, frame_size):
        """Fully in view and nobody else on top of it, so the description is of one person."""
        w, h = frame_size
        m = self.edge_margin
        if box[0] <= m or box[1] <= m or box[2] >= w - m or box[3] >= h - m:
            return False
        return all(_iou(box, o) <= self.clean_iou for o in others if o is not box)

    def update(self, now, visible, hidden, embed_fn, frame_size):
        """Advance one frame.

        visible: [(tracker_id, box)] people seen now.
        hidden: [(tracker_id, predicted_box, seconds)] people the tracker remembers but cannot
        see. embed_fn(box) -> unit vector or None.
        Returns the published id of each visible tracker id: {tracker_id: id}.
        """
        boxes = [b for _, b in visible]
        budget = self.max_embeds
        for tid, _ in visible:
            self._age[tid] = self._age.get(tid, 0) + 1
        in_view = {self.published(tid) for tid, _ in visible}

        # 1) describe visible people now and then, so there is something to compare a return to
        due = []
        for tid, box in visible:
            pid = self.published(tid)
            if (tid in self._settled and self._clean(box, boxes, frame_size)
                    and now - self._last_embed.get(pid, -1e9) >= self.refresh_s):
                due.append((self._last_embed.get(pid, -1e9), tid, pid, box))
        for _, tid, pid, box in sorted(due, key=lambda d: d[0]):
            if budget <= 0:
                break
            emb = embed_fn(box)
            if emb is not None:
                self._gallery.setdefault(pid, deque(maxlen=self.gallery_size)).append(emb)
                self._last_embed[pid] = now
                budget -= 1

        # 2) new tracker ids: describe them and look for a hidden person they could be
        candidates = {}
        for tid_h, box_h, secs in hidden:
            pid_h = self.published(tid_h)
            if secs <= self.keep_s and pid_h not in in_view and self._gallery.get(pid_h):
                candidates[pid_h] = (box_h, secs)
        proposals = []
        for tid, box in visible:
            if tid in self._settled or self._age[tid] < self.min_age:
                continue
            attempts, last = self._tries.get(tid, (0, -1e9))
            if now - last < self.retry_s or not self._clean(box, boxes, frame_size):
                continue
            if budget <= 0:
                break
            emb = embed_fn(box)
            budget -= 1
            if emb is None:
                continue
            self._tries[tid] = (attempts + 1, now)
            if not candidates:                # nobody hidden it could be: just remember its look
                self._settle(tid, self.published(tid), emb, now)
                continue
            scored = self._score(box, emb, candidates)
            if scored and scored[0][0] >= self.sim_threshold and (
                    len(scored) == 1 or scored[0][0] - scored[1][0] >= self.margin):
                proposals.append((scored[0][0], tid, scored[0][1], emb))
            elif attempts + 1 >= self.tries:
                self._settle(tid, self.published(tid), emb, now)
        taken = set()
        for sim, tid, pid, emb in sorted(proposals, key=lambda p: -p[0]):
            if pid in taken:
                continue
            taken.add(pid)
            self.alias[tid] = pid
            self.relinks += 1
            self._settle(tid, pid, emb, now)

        # forget what is long gone
        known = in_view | {self.published(t) for t, _, _ in hidden}
        for pid in [p for p in self._gallery if p not in known and
                    now - self._last_embed.get(p, now) > 2 * self.keep_s]:
            self._gallery.pop(pid, None)
            self._last_embed.pop(pid, None)
        return {tid: self.published(tid) for tid, _ in visible}

    def _score(self, box, emb, candidates):
        """[(similarity, published id)], best first, for hidden people it could have reached."""
        cx, cy = _center(box)
        out = []
        for pid, (hbox, secs) in candidates.items():
            hx, hy = _center(hbox)
            reach = (self.gate + self.walk * secs) * max(_height(box), _height(hbox))
            if np.hypot(cx - hx, cy - hy) > reach:
                continue
            out.append((float(np.max(np.stack(self._gallery[pid]) @ emb)), pid))
        return sorted(out, reverse=True)

    def _settle(self, tid, pid, emb, now):
        self._settled.add(tid)
        self._gallery.setdefault(pid, deque(maxlen=self.gallery_size)).append(emb)
        self._last_embed[pid] = now
