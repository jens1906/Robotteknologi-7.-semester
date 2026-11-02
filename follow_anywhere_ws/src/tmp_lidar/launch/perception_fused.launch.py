"""Full perception: the camera part (perception package) + the lidar part and fusion (tmp_lidar).

perception_node's /perception/leader and /perception/obstacles are remapped to /perception/camera/*;
tmp_lidar_node publishes the fused /perception/leader, /perception/obstacles and /perception/scan in their place,
so navigation runs unchanged. Every argument of perception.launch.py is passed through.

    ros2 launch tmp_lidar perception_fused.launch.py
    ros2 launch tmp_lidar perception_fused.launch.py startCamera:=false useSimTime:=true   # rosbag with --clock
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetRemap


def generate_launch_description():
    share = get_package_share_directory('tmp_lidar')
    use_sim_time = LaunchConfiguration('useSimTime')

    camera_part = GroupAction([
        SetRemap(src='/perception/leader', dst='/perception/camera/leader'),
        SetRemap(src='/perception/obstacles', dst='/perception/camera/obstacles'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('perception'), 'launch', 'perception.launch.py'))),
    ])

    lidar_part = Node(
        package='tmp_lidar', executable='tmp_lidar_node', name='tmp_lidar', output='screen',
        parameters=[LaunchConfiguration('lidarParamsFile'), {'use_sim_time': use_sim_time}])

    return LaunchDescription([
        DeclareLaunchArgument('lidarParamsFile', default_value=os.path.join(share, 'config', 'tmp_lidar.yaml'),
                              description='tmp_lidar parameters'),
        DeclareLaunchArgument('useSimTime', default_value='false', description='use /clock'),
        camera_part,
        lidar_part,
    ])
