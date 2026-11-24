"""Full perception: the camera part (perception package) + the lidar part and fusion (lidar_obstacles).

perception_node publishes /perception/camera/leader and /perception/camera/obstacles; lidar_obstacles_node
publishes the fused /perception/leader, /perception/obstacles_static (lidar only), /perception/obstacles_combined
(lidar + camera people) and /perception/scan, only while both the lidar and the camera are fresh.
Every argument of perception.launch.py is passed through.

    ros2 launch lidar_obstacles perception_fused.launch.py
    ros2 launch lidar_obstacles perception_fused.launch.py startCamera:=false useSimTime:=true   # rosbag with --clock
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Folder where this package's config files are installed, and the useSimTime launch argument.
    share = get_package_share_directory('lidar_obstacles')
    use_sim_time = LaunchConfiguration('useSimTime')

    # Camera part: perception's own launch file.
    camera_part = IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(
        get_package_share_directory('perception'), 'launch', 'perception.launch.py')))

    # Lidar part: lidar_obstacles_node with its parameter file and use_sim_time.
    lidar_part = Node(
        package='lidar_obstacles', executable='lidar_obstacles_node', name='lidar_obstacles', output='screen',
        parameters=[LaunchConfiguration('lidarParamsFile'), {'use_sim_time': use_sim_time}])

    # Launch arguments (lidarParamsFile:=..., useSimTime:=true), then both parts.
    return LaunchDescription([
        DeclareLaunchArgument('lidarParamsFile', default_value=os.path.join(share, 'config', 'lidar_obstacles_params.yaml'),
                              description='lidar_obstacles parameters'),
        DeclareLaunchArgument('useSimTime', default_value='false', description='use /clock'),
        camera_part,
        lidar_part,
    ])
