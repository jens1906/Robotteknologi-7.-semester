"""Start the perception viewer on the laptop (not on the robot).

    ros2 launch perception_interface viewer.launch.py
    ros2 launch perception_interface viewer.launch.py params_file:=/path/to/other.yaml

It only reads the compressed camera image and the small perception messages, so it is light on the
network. Perception must run (on the robot, or from a rosbag) for anything to show.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('perception_interface')
    return LaunchDescription([
        DeclareLaunchArgument('params_file',
                              default_value=os.path.join(share, 'config', 'viewer_params.yaml'),
                              description='viewer parameters'),
        Node(package='perception_interface', executable='perception_viewer', name='perception_viewer',
             output='screen', parameters=[LaunchConfiguration('params_file')]),
    ])
