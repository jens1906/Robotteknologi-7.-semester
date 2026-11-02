"""Start tmp_lidar_node alone (perception_node must publish on /perception/camera/*, see perception_fused.launch.py).

    ros2 launch tmp_lidar tmp_lidar.launch.py
    ros2 launch tmp_lidar tmp_lidar.launch.py useSimTime:=true          # bag played with --clock
    ros2 launch tmp_lidar tmp_lidar.launch.py paramsFile:=/path/to.yaml
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('tmp_lidar')
    return LaunchDescription([
        DeclareLaunchArgument('paramsFile', default_value=os.path.join(share, 'config', 'tmp_lidar.yaml'),
                              description='tmp_lidar parameters'),
        DeclareLaunchArgument('useSimTime', default_value='false', description='use /clock'),
        Node(package='tmp_lidar', executable='tmp_lidar_node', name='tmp_lidar', output='screen',
             parameters=[LaunchConfiguration('paramsFile'), {'use_sim_time': LaunchConfiguration('useSimTime')}]),
    ])
