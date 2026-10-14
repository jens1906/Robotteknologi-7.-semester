#!/usr/bin/env python3
"""
Converts Unitree sport-mode requests to /cmd_vel for the CHAMP controller.
Lets nodes written for the physical Go2 (e.g. follow_anywhere_ws/manual_controller)
drive the simulated robot without any sim-specific code.

Subscribes:
  /api/sport/request  (unitree_api/msg/Request)

Publishes:
  /cmd_vel  (geometry_msgs/msg/Twist)

Handled api_ids:
  1008 Move          parameter JSON {"x": vx, "y": vy, "z": wz} -> Twist
  1003 StopMove      zero Twist
  1004/1005/1006     StandUp / StandDown / RecoveryStand — no CHAMP equivalent, ignored

Like the real robot, motion stops if no Move arrives for `cmd_timeout` seconds.
/cmd_vel is only published while moving (plus one zero on stop), so manual
`ros2 topic pub /cmd_vel` still works when no sport requests are coming in.
"""

import json
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from unitree_api.msg import Request

API_STOP_MOVE = 1003
API_STAND_UP = 1004
API_STAND_DOWN = 1005
API_RECOVERY_STAND = 1006
API_MOVE = 1008

IGNORED_APIS = {
    API_STAND_UP: 'StandUp',
    API_STAND_DOWN: 'StandDown',
    API_RECOVERY_STAND: 'RecoveryStand',
}


class SportRequestBridge(Node):

    def __init__(self):
        super().__init__('sim_bridge')
        self.declare_parameter('cmd_timeout', 0.5)
        self.declare_parameter('publish_rate', 20.0)
        self._cmd_timeout = self.get_parameter('cmd_timeout').value

        self._target = Twist()
        # Steady-clock time of the last Move; None when not moving.
        self._last_move: float | None = None

        self.request_sub = self.create_subscription(
            Request, '/api/sport/request', self._request_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(1.0 / self.get_parameter('publish_rate').value, self._tick)

        self.get_logger().info('sim_bridge started — /api/sport/request -> /cmd_vel')

    def _request_callback(self, msg: Request) -> None:
        api_id = msg.header.identity.api_id
        if api_id == API_MOVE:
            try:
                p = json.loads(msg.parameter)
                twist = Twist()
                twist.linear.x = float(p.get('x', 0.0))
                twist.linear.y = float(p.get('y', 0.0))
                twist.angular.z = float(p.get('z', 0.0))
            except (ValueError, TypeError, AttributeError):
                self.get_logger().warn(
                    f'Bad Move parameter: {msg.parameter!r}',
                    throttle_duration_sec=2.0,
                )
                return
            self._target = twist
            self._last_move = time.monotonic()
        elif api_id == API_STOP_MOVE:
            self._stop()
        elif api_id in IGNORED_APIS:
            self.get_logger().info(
                f'{IGNORED_APIS[api_id]} ignored in simulation',
                throttle_duration_sec=2.0,
            )
        else:
            self.get_logger().debug(f'Unhandled api_id {api_id}',
                                    throttle_duration_sec=5.0)

    def _stop(self) -> None:
        if self._last_move is not None:
            self._last_move = None
            self._target = Twist()
            self.cmd_vel_pub.publish(Twist())

    def _tick(self) -> None:
        if self._last_move is None:
            return
        if time.monotonic() - self._last_move > self._cmd_timeout:
            self._stop()
            return
        self.cmd_vel_pub.publish(self._target)


def main(args=None):
    rclpy.init(args=args)
    node = SportRequestBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
