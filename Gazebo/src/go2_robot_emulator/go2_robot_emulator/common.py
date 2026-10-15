"""Helpers shared by the emulator nodes.

All emulator nodes run on wall-clock time (use_sim_time false) and stamp their
output with it, like the real robot, which stamps with its own clock.
"""

import math
import os

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

# The robot offers every topic as reliable, keep_last 1.
ROBOT_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

# Inputs from ros_gz_bridge: only the latest sample matters.
SIM_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5,
)


def quat_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rpy_to_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def quat_to_rpy(x, y, z, w):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def load_mount(name):
    """Rotation and translation of a sensor frame in base_link (go2_mounts.yaml)."""
    path = os.path.join(get_package_share_directory('go2_bringup'), 'config', 'go2_mounts.yaml')
    with open(path) as f:
        m = yaml.safe_load(f)[name]
    return rpy_to_matrix(*m['rpy']), np.array(m['xyz'], dtype=float)


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
