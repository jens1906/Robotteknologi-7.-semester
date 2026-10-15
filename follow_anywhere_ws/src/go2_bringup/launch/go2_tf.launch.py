"""TF for the Go2, identical for the real robot and the Gazebo sim.

Neither publishes /tf, so start this next to any node that needs frames:
  - static base_link -> utlidar_lidar / utlidar_imu / camera_link
    (from config/go2_mounts.yaml)
  - realsense2_camera optical frames under camera_link
  - odom -> base_link from /utlidar/robot_odom (odom:=false to skip it)
"""

import math
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def static_tf(parent, child, xyz, rpy):
    return Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name=f'tf_{child}',
        arguments=[
            '--x', str(xyz[0]), '--y', str(xyz[1]), '--z', str(xyz[2]),
            '--roll', str(rpy[0]), '--pitch', str(rpy[1]), '--yaw', str(rpy[2]),
            '--frame-id', parent, '--child-frame-id', child,
        ],
    )


def generate_launch_description():
    mounts_file = os.path.join(
        get_package_share_directory('go2_bringup'), 'config', 'go2_mounts.yaml')
    with open(mounts_file) as f:
        mounts = yaml.safe_load(f)

    optical = [-math.pi / 2, 0.0, -math.pi / 2]
    nodes = [static_tf('base_link', name, m['xyz'], m['rpy']) for name, m in mounts.items()]
    nodes += [
        static_tf('camera_link', 'camera_color_optical_frame', [0, 0, 0], optical),
        static_tf('camera_link', 'camera_depth_optical_frame', [0, 0, 0], optical),
    ]

    return LaunchDescription([
        DeclareLaunchArgument('odom', default_value='true',
                              description='Broadcast odom -> base_link from /utlidar/robot_odom'),
        *nodes,
        Node(package='go2_bringup', executable='odom_tf', name='odom_tf',
             condition=IfCondition(LaunchConfiguration('odom'))),
    ])
