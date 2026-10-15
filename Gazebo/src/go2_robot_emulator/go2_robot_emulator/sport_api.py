"""Sport API of the Go2, served by the CHAMP gait controller.

Lets nodes written for the physical Go2 (e.g. follow_anywhere_ws/manual_controller)
drive the simulated robot without any sim-specific code.

Subscribes:
  /api/sport/request   (unitree_api/msg/Request)

Publishes:
  /api/sport/response  (unitree_api/msg/Response)  one per request, like the robot
  /cmd_vel             (geometry_msgs/msg/Twist)   to CHAMP
  /body_pose           (geometry_msgs/msg/Pose)    to CHAMP, body height offset

Handled api_ids:
  1008 Move          parameter JSON {"x": vx, "y": vy, "z": wz}
  1003 StopMove      zero velocity
  1005 StandDown, 1001 Damp                    lower the body, ignore Move
  1004 StandUp, 1006 RecoveryStand, 1002 BalanceStand   stand at nominal height
Every other api_id is answered with code 0 and otherwise ignored.

Like the real robot, motion stops if no Move arrives for `cmd_timeout` seconds.
Commands are clamped to the limits the real robot was driven at, then scaled
by `velocity_gain`: CHAMP's open-loop gait reaches only ~63% (x), ~74% (y) and
~72% (yaw) of what it is told, while the real robot reaches ~85% (x) and ~98%
(yaw) in bag go2_all_topics_6_10_11_28. Recalibrate with
tools/replay_commands.py after changing gait.yaml. `response_time` adds a
first-order lag on top of CHAMP's own (0 = none).
/cmd_vel is only published while moving (plus zeros until stopped), so manual
`ros2 topic pub /cmd_vel` still works when no sport requests are coming in.
"""

import json
import math
import time

from geometry_msgs.msg import Pose, Twist
from rclpy.node import Node
from unitree_api.msg import Request, Response

from go2_robot_emulator.common import run

API_DAMP = 1001
API_BALANCE_STAND = 1002
API_STOP_MOVE = 1003
API_STAND_UP = 1004
API_STAND_DOWN = 1005
API_RECOVERY_STAND = 1006
API_MOVE = 1008

DOWN_APIS = {API_STAND_DOWN, API_DAMP}
UP_APIS = {API_STAND_UP, API_RECOVERY_STAND, API_BALANCE_STAND}


class SportApi(Node):

    def __init__(self):
        super().__init__('sport_api')
        self.declare_parameter('cmd_timeout', 0.5)
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('response_time', 0.0)
        self.declare_parameter('velocity_gain', [1.35, 1.35, 1.36])
        self.declare_parameter('max_vx', 0.6)
        self.declare_parameter('max_vy', 0.4)
        self.declare_parameter('max_wz', 1.6)
        # Body height change for StandDown, relative to CHAMP's nominal height.
        self.declare_parameter('stand_down_offset', -0.18)
        self._cmd_timeout = self.get_parameter('cmd_timeout').value
        self._dt = 1.0 / self.get_parameter('publish_rate').value
        self._tau = self.get_parameter('response_time').value
        self._limits = [self.get_parameter(n).value for n in ('max_vx', 'max_vy', 'max_wz')]
        self._gain = list(self.get_parameter('velocity_gain').value)

        self._target = [0.0, 0.0, 0.0]
        self._current = [0.0, 0.0, 0.0]
        # Steady-clock time of the last Move; None when not commanded to move.
        self._last_move: float | None = None
        self._down = False

        self.create_subscription(Request, '/api/sport/request', self._request_callback, 10)
        self.response_pub = self.create_publisher(Response, '/api/sport/response', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.body_pose_pub = self.create_publisher(Pose, '/body_pose', 1)
        self.create_timer(self._dt, self._tick)

        self.get_logger().info('sport_api started — /api/sport/request -> /cmd_vel')

    def _request_callback(self, msg: Request) -> None:
        api_id = msg.header.identity.api_id
        if api_id == API_MOVE:
            self._move(msg.parameter)
        elif api_id == API_STOP_MOVE:
            self._last_move = None
            self._target = [0.0, 0.0, 0.0]
        elif api_id in DOWN_APIS:
            self._set_down(True)
        elif api_id in UP_APIS:
            self._set_down(False)
        else:
            self.get_logger().debug(f'Unhandled api_id {api_id}', throttle_duration_sec=5.0)

        if not msg.header.policy.noreply:
            response = Response()
            response.header.identity = msg.header.identity
            response.header.status.code = 0
            self.response_pub.publish(response)

    def _move(self, parameter: str) -> None:
        try:
            p = json.loads(parameter)
            cmd = [float(p.get('x', 0.0)), float(p.get('y', 0.0)), float(p.get('z', 0.0))]
        except (ValueError, TypeError, AttributeError):
            self.get_logger().warn(f'Bad Move parameter: {parameter!r}', throttle_duration_sec=2.0)
            return
        if self._down:
            self.get_logger().info('Move ignored while lying down', throttle_duration_sec=2.0)
            return
        self._target = [max(-lim, min(lim, v)) * gain
                        for v, lim, gain in zip(cmd, self._limits, self._gain)]
        self._last_move = time.monotonic()

    def _set_down(self, down: bool) -> None:
        self._down = down
        if down:
            self._last_move = None
            self._target = [0.0, 0.0, 0.0]
        pose = Pose()
        pose.position.z = self.get_parameter('stand_down_offset').value if down else 0.0
        pose.orientation.w = 1.0
        self.body_pose_pub.publish(pose)

    def _tick(self) -> None:
        if self._last_move is not None and time.monotonic() - self._last_move > self._cmd_timeout:
            self._last_move = None
            self._target = [0.0, 0.0, 0.0]
        if self._last_move is None and not any(self._current):
            return

        alpha = 1.0 - math.exp(-self._dt / self._tau) if self._tau > 0 else 1.0
        self._current = [c + alpha * (t - c) for c, t in zip(self._current, self._target)]
        if self._last_move is None and all(abs(c) < 1e-3 for c in self._current):
            self._current = [0.0, 0.0, 0.0]

        twist = Twist()
        twist.linear.x, twist.linear.y, twist.angular.z = self._current
        self.cmd_vel_pub.publish(twist)


def main():
    run(SportApi)


if __name__ == '__main__':
    main()
