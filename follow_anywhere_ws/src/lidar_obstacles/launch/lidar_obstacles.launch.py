"""Start lidar_obstacles_node alone (it needs perception_node's /perception/camera/*, see perception_fused.launch.py).

    ros2 launch lidar_obstacles lidar_obstacles.launch.py
    ros2 launch lidar_obstacles lidar_obstacles.launch.py useSimTime:=true          # bag played with --clock
    ros2 launch lidar_obstacles lidar_obstacles.launch.py paramsFile:=/path/to.yaml
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Folder where this package's config files are installed.
    share = get_package_share_directory('lidar_obstacles')
    # Two launch arguments (paramsFile:=..., useSimTime:=true) and the node itself, which loads the parameter
    # file and sets use_sim_time (true when a rosbag is played with --clock).
    return LaunchDescription([
        DeclareLaunchArgument('paramsFile', default_value=os.path.join(share, 'config', 'lidar_obstacles_params.yaml'),
                              description='lidar_obstacles parameters'),
        DeclareLaunchArgument('useSimTime', default_value='false', description='use /clock'),
        Node(package='lidar_obstacles', executable='lidar_obstacles_node', name='lidar_obstacles', output='screen',
             parameters=[LaunchConfiguration('paramsFile'), {'use_sim_time': LaunchConfiguration('useSimTime')}]),
    ])
