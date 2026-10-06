# SPDX-License-Identifier: BSD-3-Clause

"""
Keyboard teleop for the Go2 over native DDS (CONN_TYPE=cyclonedds / ethernet).

W/S      forward / backward
A/D      strafe left / right
Left/Right arrows   turn left / right
Up/Down arrows      increase / decrease speed
U / J / R           stand up / stand down / recovery stand
Space    stop
Q / Ctrl-C   quit (sends stop)

A terminal only reports key presses (and auto-repeat while a key is held), not
releases. Each axis therefore stays active for `hold_time` seconds after its last
key event and drops to zero afterwards; when every axis has expired a single
StopMove is sent.
"""

import json
import os
import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from unitree_api.msg import Request

API_STOP_MOVE = 1003
API_STAND_UP = 1004
API_STAND_DOWN = 1005
API_RECOVERY_STAND = 1006
API_MOVE = 1008

# axis name -> (key, direction)
KEY_AXES = {
    'w': ('x', 1.0), 's': ('x', -1.0),
    'a': ('y', 1.0), 'd': ('y', -1.0),
    '\x1b[D': ('z', 1.0), '\x1b[C': ('z', -1.0),
}
KEY_ACTIONS = {
    'u': API_STAND_UP, 'j': API_STAND_DOWN, 'r': API_RECOVERY_STAND,
    ' ': API_STOP_MOVE,
}
SPEED_UP, SPEED_DOWN = '\x1b[A', '\x1b[B'

HELP = (
    '\r\nGo2 keyboard teleop\r\n'
    '  w/s  forward/back    a/d  strafe left/right\r\n'
    '  <-/-> turn left/right    up/down arrows  speed +/-\r\n'
    '  u stand up   j stand down   r recovery stand   space stop   q quit\r\n'
)


def read_keys(fd: int) -> list:
    """Read all pending bytes and split into keys (arrow keys are 3-byte escapes)."""
    data = os.read(fd, 64).decode(errors='ignore')
    keys, i = [], 0
    while i < len(data):
        if data.startswith('\x1b[', i) and i + 2 < len(data):
            keys.append(data[i:i + 3])
            i += 3
        else:
            keys.append(data[i].lower() if data[i].isalpha() else data[i])
            i += 1
    return keys


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('manual_controller')
        self.declare_parameter('linear_speed', 0.3)
        self.declare_parameter('lateral_speed', 0.2)
        self.declare_parameter('angular_speed', 0.8)
        self.declare_parameter('max_scale', 2.0)
        self.declare_parameter('hold_time', 0.4)
        self.declare_parameter('publish_rate', 10.0)

        self.base = {
            'x': self.get_parameter('linear_speed').value,
            'y': self.get_parameter('lateral_speed').value,
            'z': self.get_parameter('angular_speed').value,
        }
        self.max_scale = self.get_parameter('max_scale').value
        self.hold_time = self.get_parameter('hold_time').value
        self.scale = 1.0
        # axis -> (direction, time of last key event)
        self.active = {}
        self.moving = False

        self.pub = self.create_publisher(Request, '/api/sport/request', 10)
        self.create_timer(1.0 / self.get_parameter('publish_rate').value, self._tick)

    def _send(self, api_id: int, parameter: str = '') -> None:
        msg = Request()
        msg.header.identity.id = time.monotonic_ns() // 1000
        msg.header.identity.api_id = api_id
        msg.parameter = parameter
        self.pub.publish(msg)

    def handle_key(self, key: str) -> None:
        now = time.monotonic()
        if key in KEY_AXES:
            axis, direction = KEY_AXES[key]
            self.active[axis] = (direction, now)
        elif key in KEY_ACTIONS:
            self.active.clear()
            self.moving = False
            self._send(KEY_ACTIONS[key])
        elif key == SPEED_UP:
            self.scale = min(self.scale + 0.25, self.max_scale)
            self._report_speed()
        elif key == SPEED_DOWN:
            self.scale = max(self.scale - 0.25, 0.25)
            self._report_speed()

    def _report_speed(self) -> None:
        sys.stdout.write(f'\r\nspeed scale: {self.scale:.2f}\r\n')
        sys.stdout.flush()

    def velocity(self, now: float) -> dict:
        self.active = {a: (d, t) for a, (d, t) in self.active.items()
                       if now - t <= self.hold_time}
        return {a: d * self.base[a] * self.scale for a, (d, _) in self.active.items()}

    def _tick(self) -> None:
        vel = self.velocity(time.monotonic())
        if vel:
            payload = {'x': vel.get('x', 0.0), 'y': vel.get('y', 0.0), 'z': vel.get('z', 0.0)}
            self._send(API_MOVE, json.dumps(payload))
            self.moving = True
        elif self.moving:
            self._send(API_STOP_MOVE)
            self.moving = False

    def stop(self) -> None:
        self._send(API_STOP_MOVE)


def main(args=None):
    if not sys.stdin.isatty():
        print('keyboard_teleop needs an interactive terminal (stdin is not a tty).')
        return 1

    rclpy.init(args=args)
    node = KeyboardTeleop()
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    sys.stdout.write(HELP)
    sys.stdout.flush()
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            if select.select([fd], [], [], 0)[0]:
                for key in read_keys(fd):
                    if key in ('q', '\x03'):
                        raise KeyboardInterrupt
                    node.handle_key(key)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        if rclpy.ok():
            node.stop()
            rclpy.spin_once(node, timeout_sec=0.1)
            node.destroy_node()
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
