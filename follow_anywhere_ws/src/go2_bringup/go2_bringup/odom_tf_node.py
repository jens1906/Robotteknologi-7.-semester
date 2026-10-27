"""Broadcast odom -> base_link from /utlidar/robot_odom.

The Go2 publishes its odometry but no /tf. The transform keeps the message's
stamp (the robot's clock), so consumers looking up TF should use the stamps
of the sensor messages, not now().
"""

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster


class OdomTf(Node):

    def __init__(self):
        super().__init__('odom_tf')
        self._broadcaster = TransformBroadcaster(self)
        self.create_subscription(Odometry, '/utlidar/robot_odom', self._odom_callback, 10)

    def _odom_callback(self, msg: Odometry) -> None:
        t = TransformStamped()
        t.header = msg.header
        t.child_frame_id = msg.child_frame_id
        p = msg.pose.pose.position
        t.transform.translation.x = p.x
        t.transform.translation.y = p.y
        t.transform.translation.z = p.z
        t.transform.rotation = msg.pose.pose.orientation
        self._broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdomTf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
