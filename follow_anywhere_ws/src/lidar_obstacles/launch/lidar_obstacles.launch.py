"""Start the lidar obstacles node.

    ros2 launch lidar_obstacles lidar_obstacles.launch.py
    ros2 launch lidar_obstacles lidar_obstacles.launch.py useSimTime:=true

Topic names are written in lidar_obstacles_node.cpp: remap them in this file.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('lidar_obstacles')
    args = [
        DeclareLaunchArgument('paramsFile',
                              default_value=os.path.join(share, 'config', 'lidar_obstacles_params.yaml'),
                              description='lidar obstacles node parameters'),
        DeclareLaunchArgument('useSimTime', default_value='false',
                              description='use /clock from the sim or a rosbag played with --clock'),
    ]

    lidarObstacles = Node(
        package='lidar_obstacles', executable='lidar_obstacles_node', name='lidar_obstacles_node', output='screen',
        parameters=[LaunchConfiguration('paramsFile'), {'use_sim_time': LaunchConfiguration('useSimTime')}])

    return LaunchDescription([*args, lidarObstacles])
