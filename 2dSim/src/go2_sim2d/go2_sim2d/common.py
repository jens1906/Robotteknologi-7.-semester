"""Helpers shared by the 2D sim nodes."""

import math

import rclpy
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

# The robot offers every topic as reliable, keep_last 1 (as in Gazebo's go2_robot_emulator).
ROBOT_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


def yaw_of(pose):
    """Yaw of a geometry_msgs/Pose (rotation about z only)."""
    q = pose.orientation
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def stamp_seconds(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def run(node_class):
    rclpy.init()
    node = node_class()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
