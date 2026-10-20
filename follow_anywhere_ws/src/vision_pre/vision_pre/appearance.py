"""Cheap appearance cues next to the neural embedding: crop helper and clothing colour."""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# HSV bins (H, S, V). Low-saturation clothes (black, white, grey) are separated by V.
_BINS = (8, 4, 4)
_RANGES = [0, 180, 0, 256, 0, 256]


@dataclass
class Appearance:
    """What we remember about one view of a person."""
    emb: np.ndarray                       # unit vector from the embedder
    col: Optional[np.ndarray] = None      # (2, bins): upper and lower body colour histograms


def inner_crop(frame, box, side=0.08, vert=0.03):
    """Crop a detection box, trimmed a little on every side to drop background."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = x1 + bw * side, x2 - bw * side
    y1, y2 = y1 + bh * vert, y2 - bh * vert
    return frame[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))]


def _hist(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, _BINS, _RANGES).ravel()
    total = hist.sum()
    return hist / total if total > 0 else hist


def colour_hist(crop):
    """Colour histograms of the shirt area and the trouser area, or None for a tiny crop."""
    h, w = crop.shape[:2]
    if h < 16 or w < 8:
        return None
    x1, x2 = int(w * 0.2), int(w * 0.8)               # middle columns, arms and background out
    upper = crop[int(h * 0.15):int(h * 0.50), x1:x2]
    lower = crop[int(h * 0.50):int(h * 0.85), x1:x2]
    if upper.size == 0 or lower.size == 0:
        return None
    return np.stack([_hist(upper), _hist(lower)]).astype(np.float32)


def colour_sim(a, b):
    """Bhattacharyya coefficient averaged over upper/lower body, 0..1 (1 = same colours).

    `b` may hold many views at once, shape (N, 2, bins); returns one value per view.
    """
    bc = np.sqrt(a[None] * np.asarray(b).reshape(-1, *a.shape)).sum(axis=-1)   # (N, 2)
    return bc.mean(axis=-1)
