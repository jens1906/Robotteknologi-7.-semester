"""Lost-leader search node (docs/leader_search_spec.md)."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory('leader_search'), 'config',
                                  'leader_search.yaml')
    params = LaunchConfiguration('params')
    return LaunchDescription([
        DeclareLaunchArgument('params', default_value=default_params),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        Node(package='leader_search', executable='leader_search_node', name='leader_search',
             output='screen',
             parameters=[params, {'params_file': params,
                                  'use_sim_time': LaunchConfiguration('use_sim_time')}]),
    ])
