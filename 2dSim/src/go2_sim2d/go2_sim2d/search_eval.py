"""Scores one run of the lost-leader search (leader_search) in a search_* scene.

Listens to the sim's ground truth and the search node's output, decides the spec's pass
criterion for the scene (follow_anywhere_ws/docs/leader_search_spec.md, "Sim scenarios"),
writes one JSON line to result_file and exits after duration seconds (or once decided).

  /sim/odometry      robot ground truth (world = odom frame)
  /sim/leader/pose   leader ground truth
  /sim/collisions    contacts so far
  /search/mode       0 TRACKING, 1 SEARCH, 2 GIVE_UP
  /search/markers    id 0 = search target
"""

import json
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import UInt8, UInt32
from visualization_msgs.msg import MarkerArray

TRACKING, SEARCH, GIVE_UP = 0, 1, 2

# Per scene: run length (s after start) and the spec's pass criterion.
SCENARIOS = {
    'search_corridor': {'duration': 80.0, 'check': 'reacquired', 'need': 0.95},
    'search_side_door': {'duration': 60.0, 'check': 'door_before_pass', 'need': 0.70,
                         'door': (7.4, 8.6, 1.25), 'pass_x': 8.0},
    'search_t_junction': {'duration': 65.0, 'check': 'few_switches', 'need': 1.0,
                          'max_switches': 2},
    'search_corner_stop': {'duration': 60.0, 'check': 'reacquired', 'need': 0.80},
    'search_walk_out': {'duration': 60.0, 'check': 'gave_up', 'need': 1.0},
}
T_MAX = 20.0          # leader_search default t_max
SWITCH_JUMP = 1.0     # target jump [m] between two marker messages counted as a switch (rho)


class SearchEval(Node):

    def __init__(self):
        super().__init__('search_eval')
        self.declare_parameter('world', 'search_corridor')
        self.declare_parameter('result_file', '')
        self.declare_parameter('seed', -1)
        self.world = self.get_parameter('world').value
        if self.world not in SCENARIOS:
            raise ValueError(f'search_eval: no scenario for world {self.world}')
        self.cfg = SCENARIOS[self.world]
        self.robot = None
        self.leader = None
        self.mode = TRACKING
        self.target = None
        self.t0 = self.now()
        self.r = {'world': self.world, 'seed': self.get_parameter('seed').value,
                  'losses': 0, 'reacquired': 0, 'switches': 0, 'gave_up': False,
                  'give_up_after': None, 'door_before_pass': None, 'collisions': 0, 'collisions_searching': 0,
                  'first_loss_t': None, 'max_search_s': 0.0}
        self.loss_t = None
        self.create_subscription(Odometry, '/sim/odometry', self._robot_cb, 10)
        self.create_subscription(PoseStamped, '/sim/leader/pose', self._leader_cb, 10)
        self.create_subscription(UInt32, '/sim/collisions', self._collisions_cb, 10)
        self.create_subscription(UInt8, '/search/mode', self._mode_cb, 10)
        self.create_subscription(MarkerArray, '/search/markers', self._markers_cb, 10)
        self.create_timer(0.5, self._check_done)
        self.done = False

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _robot_cb(self, m):
        self.robot = (m.pose.pose.position.x, m.pose.pose.position.y)

    def _leader_cb(self, m):
        self.leader = (m.pose.position.x, m.pose.position.y)

    def _collisions_cb(self, m):
        new = int(m.data) - self.r['collisions']
        if new > 0 and self.mode != TRACKING:
            self.r['collisions_searching'] += new
        self.r['collisions'] = int(m.data)

    def _mode_cb(self, m):
        mode, t = int(m.data), self.now() - self.t0
        if mode == SEARCH and self.mode == TRACKING:
            self.r['losses'] += 1
            self.loss_t = t
            self.target = None
            if self.r['first_loss_t'] is None:
                self.r['first_loss_t'] = round(t, 2)
        if mode == TRACKING and self.mode != TRACKING:
            self.r['reacquired'] += 1
            if self.loss_t is not None:
                self.r['max_search_s'] = round(max(self.r['max_search_s'], t - self.loss_t), 2)
        if mode == GIVE_UP and self.mode != GIVE_UP:
            self.r['gave_up'] = True
            if self.loss_t is not None:
                self.r['give_up_after'] = round(t - self.loss_t, 2)
        self.mode = mode

    def _markers_cb(self, m):
        if self.mode != SEARCH:
            return
        for marker in m.markers:
            if marker.id != 0:
                continue
            p = (marker.pose.position.x, marker.pose.position.y)
            if self.target is not None and math.dist(p, self.target) > SWITCH_JUMP:
                self.r['switches'] += 1
            self.target = p
            door = self.cfg.get('door')
            if door and self.r['door_before_pass'] is None and self.robot is not None:
                x0, x1, ymin = door
                if x0 - 0.5 <= p[0] <= x1 + 0.5 and p[1] >= ymin:
                    self.r['door_before_pass'] = self.robot[0] < self.cfg['pass_x']

    def passed(self):
        """True / False, or None when the leader was never lost (the scenario did not happen)."""
        c, r = self.cfg['check'], self.r
        if r['losses'] == 0:
            return None
        if c == 'reacquired':
            return r['losses'] > 0 and r['reacquired'] > 0
        if c == 'door_before_pass':
            return bool(r['door_before_pass'])
        if c == 'few_switches':
            return r['losses'] > 0 and r['switches'] <= self.cfg['max_switches']
        if c == 'gave_up':
            return r['gave_up'] and r['give_up_after'] is not None and \
                r['give_up_after'] <= T_MAX + 1.0
        return False

    def _check_done(self):
        if self.done or self.now() - self.t0 < self.cfg['duration']:
            return
        self.done = True
        self.r['passed'] = self.passed()
        line = json.dumps(self.r)
        self.get_logger().info(f'result: {line}')
        path = self.get_parameter('result_file').value
        if path:
            with open(path, 'a') as f:
                f.write(line + '\n')
        raise SystemExit(0)


def main():
    rclpy.init()
    node = SearchEval()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
