"""Where can the leader be? A people-cannot-teleport check on 3D positions (no ROS)."""
import math
from typing import Optional, Tuple


class LeaderMotion:
    """Remembers the last leader position and allows only plausible new ones.

    Positions are (x, y, z) in metres in the camera frame; only x and z (the floor plane) are used.
    """

    def __init__(self, max_speed=1.5, base_radius=0.6):
        self.max_speed = max_speed        # m/s, a walking person plus margin
        self.base_radius = base_radius    # m, measurement noise and body width
        self.pos: Optional[Tuple[float, float]] = None
        self.vel = (0.0, 0.0)
        self.t = 0.0

    def reset(self):
        self.pos = None
        self.vel = (0.0, 0.0)

    def update(self, pos, now):
        """Feed the leader's measured position."""
        if pos is None:
            return
        p = (pos[0], pos[2])
        if self.pos is not None and now > self.t:
            dt = now - self.t
            v = ((p[0] - self.pos[0]) / dt, (p[1] - self.pos[1]) / dt)
            speed = math.hypot(*v)
            if speed > self.max_speed:                 # clamp noise spikes
                v = (v[0] * self.max_speed / speed, v[1] * self.max_speed / speed)
            self.vel = (0.7 * self.vel[0] + 0.3 * v[0], 0.7 * self.vel[1] + 0.3 * v[1])
        self.pos, self.t = p, now

    def predicted(self, now):
        if self.pos is None:
            return None
        dt = max(0.0, now - self.t)
        return self.pos[0] + self.vel[0] * min(dt, 1.0), self.pos[1] + self.vel[1] * min(dt, 1.0)

    def distance(self, pos, now):
        """Metres between a candidate and where the leader should be (None if unknown)."""
        pred = self.predicted(now)
        if pred is None or pos is None:
            return None
        return math.hypot(pos[0] - pred[0], pos[2] - pred[1])

    def allowed(self, pos, now):
        """False only when a candidate is further away than the leader could have walked."""
        d = self.distance(pos, now)
        if d is None:
            return True                                # unknown depth: do not exclude
        return d <= self.base_radius + self.max_speed * max(0.0, now - self.t)
