"""How stable are the track IDs of a run? Numbers from per-frame track lists (no ROS).

There is no ground truth for a stock video, so these are proxies. Compare configs on the same clip:
a tracker that keeps identities has fewer unique IDs, fewer new IDs per minute, fewer flicker
tracks and longer tracks than one that re-issues IDs.
"""
import os
import statistics
from dataclasses import dataclass


@dataclass
class Summary:
    frames: int
    seconds: float
    unique_ids: int
    max_people: int           # most tracks seen in one frame
    mean_people: float
    crowded_pct: float        # % of frames with at least `crowd_size` tracks
    births_per_min: float     # IDs that appeared after the first frame, per minute
    median_track_s: float
    mean_track_s: float
    short_tracks: int         # IDs that lived shorter than `short_s` (flicker / fragments)
    id_inflation: float       # unique IDs / max_people, 1.0 = IDs never re-issued

    def text(self, crowd_size=10, short_s=1.0):
        return (f'{self.frames} frames, {self.seconds:.1f} s | '
                f'unique IDs {self.unique_ids} (inflation {self.id_inflation:.2f}) | '
                f'people/frame mean {self.mean_people:.1f} max {self.max_people}, '
                f'>={crowd_size}: {self.crowded_pct:.0f}% of frames | '
                f'new IDs {self.births_per_min:.1f}/min | '
                f'track length median {self.median_track_s:.1f} s '
                f'mean {self.mean_track_s:.1f} s | '
                f'tracks <{short_s:g} s: {self.short_tracks}')


class TrackStats:
    def __init__(self, crowd_size=10, short_s=1.0):
        self.crowd_size = crowd_size
        self.short_s = short_s
        self.t0 = None
        self.t1 = 0.0
        self.counts = []                  # tracks per frame
        self.frame_numbers = []           # video frame number of every processed frame
        self.first = {}                   # track_id -> time first seen
        self.last = {}                    # track_id -> time last seen
        self.births_after_start = 0
        self.rows = []                    # (frame, id, x, y, w, h, conf), for MOT export

    @property
    def frames(self):
        return len(self.counts)

    def add(self, t, tracks, frame=None, scale=1.0):
        """Add one processed frame. tracks: iterable of (track_id, (x1, y1, x2, y2), conf).

        frame: number of this frame in the video (default: count of processed frames).
        scale: multiplies the boxes, e.g. to get back to the pixels of the original images.
        """
        tracks = list(tracks)
        if self.t0 is None:
            self.t0 = t
        self.t1 = max(self.t1, t)
        self.counts.append(len(tracks))
        number = len(self.counts) if frame is None else frame
        self.frame_numbers.append(number)
        for tid, box, conf in tracks:
            if tid not in self.first:
                self.first[tid] = t
                if len(self.counts) > 1:
                    self.births_after_start += 1
            self.last[tid] = t
            x1, y1, x2, y2 = (v * scale for v in box)
            self.rows.append((number, tid, x1, y1, x2 - x1, y2 - y1, conf))

    def summary(self) -> Summary:
        seconds = 0.0 if self.t0 is None else self.t1 - self.t0
        lives = [self.last[i] - self.first[i] for i in self.first]
        unique = len(self.first)
        peak = max(self.counts, default=0)
        crowded = sum(1 for n in self.counts if n >= self.crowd_size)
        return Summary(
            frames=self.frames,
            seconds=seconds,
            unique_ids=unique,
            max_people=peak,
            mean_people=statistics.fmean(self.counts) if self.counts else 0.0,
            crowded_pct=100.0 * crowded / self.frames if self.frames else 0.0,
            births_per_min=self.births_after_start * 60.0 / seconds if seconds > 0 else 0.0,
            median_track_s=statistics.median(lives) if lives else 0.0,
            mean_track_s=statistics.fmean(lives) if lives else 0.0,
            short_tracks=sum(1 for v in lives if v < self.short_s),
            id_inflation=unique / peak if peak else 0.0)

    def write_mot(self, path):
        """Write the tracks as a MOTChallenge result file, creating the folder if needed."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            f.write('\n'.join(self.mot_lines()) + '\n')

    def mot_lines(self):
        """MOTChallenge rows: frame,id,x,y,w,h,conf,-1,-1,-1 (frame = processed frame number)."""
        return [f'{f},{i},{x:.1f},{y:.1f},{w:.1f},{h:.1f},{c:.3f},-1,-1,-1'
                for f, i, x, y, w, h, c in self.rows]
