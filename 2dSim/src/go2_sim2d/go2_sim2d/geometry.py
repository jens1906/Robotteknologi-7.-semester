"""2D shapes of the world, ray casting and line of sight.

Everything lives in the xy plane of the odom (= world) frame. A box is a
rectangle centred on (cx, cy), rotated by yaw, with side lengths sx and sy; a
circle is a cylinder or a person seen from above.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Circle:
    cx: float
    cy: float
    r: float

    @property
    def radius(self):
        return self.r

    def corners(self, n=24):
        a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
        return np.column_stack((self.cx + self.r * np.cos(a), self.cy + self.r * np.sin(a)))


@dataclass
class Box:
    cx: float
    cy: float
    yaw: float
    sx: float
    sy: float

    @property
    def radius(self):
        return math.hypot(self.sx, self.sy) / 2

    def corners(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        hx, hy = self.sx / 2, self.sy / 2
        local = ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy))
        return np.array([(self.cx + c * x - s * y, self.cy + s * x + c * y) for x, y in local])

    def _local(self, x, y):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = x - self.cx, y - self.cy
        return c * dx + s * dy, -s * dx + c * dy


def from_scene(obj, x=None, y=None, yaw=None):
    """Shape of a scene object (go2_scenarios static or mover dict).

    Static objects carry their pose; movers get theirs from the caller.
    """
    if x is None:
        pose = (list(map(float, obj['pose'])) + [0.0])[:3]
        x, y, yaw = pose
    if yaw is None:
        yaw = float(obj.get('yaw', 0.0))
    if obj['type'] == 'cylinder':
        return Circle(x, y, float(obj['radius']))
    sx, sy = map(float, obj['size'][:2])
    return Box(x, y, yaw, sx, sy)


def distance(shape, x, y):
    """Distance from (x, y) to the outline of shape; negative inside a circle, 0 inside a box."""
    if isinstance(shape, Circle):
        return math.hypot(x - shape.cx, y - shape.cy) - shape.r
    lx, ly = shape._local(x, y)
    return math.hypot(max(abs(lx) - shape.sx / 2, 0.0), max(abs(ly) - shape.sy / 2, 0.0))


def nearest_point(shape, x, y):
    """Point of shape's outline (or inside it, for a box) closest to (x, y)."""
    if isinstance(shape, Circle):
        d = math.hypot(x - shape.cx, y - shape.cy)
        if d < 1e-9:
            return shape.cx + shape.r, shape.cy
        f = shape.r / d
        return shape.cx + f * (x - shape.cx), shape.cy + f * (y - shape.cy)
    lx, ly = shape._local(x, y)
    lx = max(-shape.sx / 2, min(shape.sx / 2, lx))
    ly = max(-shape.sy / 2, min(shape.sy / 2, ly))
    c, s = math.cos(shape.yaw), math.sin(shape.yaw)
    return shape.cx + c * lx - s * ly, shape.cy + s * lx + c * ly


def _split(shapes):
    """Box edges as segments (K, 4) and circles as (C, 3), each with the index of its shape."""
    segments, seg_owner, circles, circle_owner = [], [], [], []
    for i, shape in enumerate(shapes):
        if isinstance(shape, Circle):
            circles.append((shape.cx, shape.cy, shape.r))
            circle_owner.append(i)
        else:
            c = shape.corners()
            for a, b in zip(c, np.roll(c, -1, axis=0)):
                segments.append((*a, *b))
                seg_owner.append(i)
    return (np.array(segments, dtype=float).reshape(-1, 4), np.array(seg_owner, dtype=int),
            np.array(circles, dtype=float).reshape(-1, 3), np.array(circle_owner, dtype=int))


def ray_cast(ox, oy, angles, shapes, max_range=math.inf):
    """Range along each ray from (ox, oy) to the first shape hit, and that shape's index.

    Rays that hit nothing within max_range get range inf and index -1.
    """
    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    d = np.column_stack((np.cos(angles), np.sin(angles)))  # (M, 2)
    best = np.full(len(angles), math.inf)
    owner = np.full(len(angles), -1, dtype=int)
    segments, seg_owner, circles, circle_owner = _split(shapes)

    if len(segments):
        p = segments[:, :2] - (ox, oy)            # (K, 2)
        e = segments[:, 2:] - segments[:, :2]     # (K, 2)
        denom = np.outer(d[:, 0], e[:, 1]) - np.outer(d[:, 1], e[:, 0])  # (M, K)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (p[:, 0] * e[:, 1] - p[:, 1] * e[:, 0]) / denom
            u = (np.outer(d[:, 1], p[:, 0]) - np.outer(d[:, 0], p[:, 1])) / denom
        hit = (np.abs(denom) > 1e-12) & (t > 0.0) & (u >= 0.0) & (u <= 1.0)
        t = np.where(hit, t, math.inf)
        k = np.argmin(t, axis=1)
        tk = t[np.arange(len(angles)), k]
        closer = tk < best
        best[closer] = tk[closer]
        owner[closer] = seg_owner[k[closer]]

    if len(circles):
        f = (ox, oy) - circles[:, :2]             # (C, 2)
        b = d @ f.T                               # (M, C)
        c = np.sum(f * f, axis=1) - circles[:, 2] ** 2
        disc = b * b - c
        root = np.sqrt(np.maximum(disc, 0.0))
        t1, t2 = -b - root, -b + root
        # From outside: entry point. From inside a circle: its exit point.
        t = np.where(t1 > 0.0, t1, t2)
        t = np.where((disc >= 0.0) & (t > 0.0), t, math.inf)
        k = np.argmin(t, axis=1)
        tk = t[np.arange(len(angles)), k]
        closer = tk < best
        best[closer] = tk[closer]
        owner[closer] = circle_owner[k[closer]]

    far = best > max_range
    best[far] = math.inf
    owner[far] = -1
    return best, owner


def line_of_sight(ox, oy, tx, ty, shapes, ignore=(), eps=1e-3):
    """True if no shape (except those with an index in ignore) is between (ox, oy) and (tx, ty)."""
    others = [s for i, s in enumerate(shapes) if i not in ignore]
    dist = math.hypot(tx - ox, ty - oy)
    if not others or dist < eps:
        return True
    r, _ = ray_cast(ox, oy, [math.atan2(ty - oy, tx - ox)], others)
    return r[0] >= dist - eps


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))
