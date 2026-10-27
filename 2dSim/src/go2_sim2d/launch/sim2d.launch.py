"""The 2D Go2 sim: sim2d (robot + world), fake_perception and the viewer window.

  ros2 launch go2_sim2d sim2d.launch.py world:=plaza
  ros2 launch go2_sim2d sim2d.launch.py viewer:=false noise_scale:=0.0

Perception parameters: config/perception.yaml (perception:=<file> for another).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    world = LaunchConfiguration('world')
    perception = LaunchConfiguration('perception')
    default_perception = os.path.join(
        get_package_share_directory('go2_sim2d'), 'config', 'perception.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='open_field',
                              description='Scene in go2_scenarios/scenes'),
        DeclareLaunchArgument('viewer', default_value='true',
                              description='Open the top-down window'),
        DeclareLaunchArgument('perception', default_value=default_perception,
                              description='fake_perception parameter file'),
        DeclareLaunchArgument('noise_scale', default_value='1.0',
                              description='Scales all perception noise (0 = ground truth)'),
        DeclareLaunchArgument('odom_drift', default_value='0.0',
                              description='Relative velocity error integrated into odometry'),
        Node(package='go2_sim2d', executable='sim2d', name='sim2d', output='screen',
             parameters=[{'world': world,
                          'odom_drift': ParameterValue(LaunchConfiguration('odom_drift'),
                                                       value_type=float)}]),
        Node(package='go2_sim2d', executable='fake_perception', name='fake_perception',
             output='screen',
             parameters=[perception,
                         {'world': world,
                          'noise_scale': ParameterValue(LaunchConfiguration('noise_scale'),
                                                        value_type=float)}]),
        Node(package='go2_sim2d', executable='viewer', name='sim2d_viewer', output='screen',
             parameters=[perception, {'world': world}],
             condition=IfCondition(LaunchConfiguration('viewer'))),
    ])
