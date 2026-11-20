"""One scored run of the lost-leader search in the 2D sim (headless).

Starts sim2d, fake_perception, the follower MPC (navigation_cpp), leader_search and
search_eval; everything stops when search_eval has written its result.
Needs follow_anywhere_ws (leader_search, navigation_cpp) sourced as well as this sim.

  ros2 launch go2_sim2d search_eval.launch.py world:=search_side_door seed:=3
  ros2 launch go2_sim2d search_eval.launch.py world:=search_corridor viewer:=true belief_rate_hz:=10.0
  ros2 run go2_sim2d search_batch --runs 50 --jobs 4     # all scenes, pass-rate table
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    world = LaunchConfiguration('world')
    seed = ParameterValue(LaunchConfiguration('seed'), value_type=int)
    perception = os.path.join(get_package_share_directory('go2_sim2d'), 'config',
                              'perception.yaml')
    search_params = os.path.join(get_package_share_directory('leader_search'), 'config',
                                 'leader_search.yaml')
    evaluator = Node(package='go2_sim2d', executable='search_eval', name='search_eval',
                     output='screen',
                     parameters=[{'world': world, 'seed': seed,
                                  'result_file': LaunchConfiguration('result_file')}])
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='search_corridor'),
        DeclareLaunchArgument('seed', default_value='0'),
        DeclareLaunchArgument('result_file', default_value=''),
        DeclareLaunchArgument('viewer', default_value='false'),
        DeclareLaunchArgument('belief_rate_hz', default_value='2.0'),
        DeclareLaunchArgument('perception_rate', default_value='15.0'),
        Node(package='go2_sim2d', executable='sim2d', name='sim2d', output='log',
             parameters=[{'world': world}]),
        Node(package='go2_sim2d', executable='fake_perception', name='fake_perception',
             output='log', parameters=[perception, {
                 'world': world, 'seed': seed, 'leader_always_obstacle': True,
                 'rate': ParameterValue(LaunchConfiguration('perception_rate'),
                                        value_type=float)}]),
        Node(package='go2_sim2d', executable='viewer', name='sim2d_viewer', output='log',
             parameters=[perception, {'world': world}],
             condition=IfCondition(LaunchConfiguration('viewer'))),
        # The sim publishes the camera/lidar obstacles on /perception/obstacles.
        Node(package='navigation_cpp', executable='follow_leader_cpp', name='leader_follower',
             output='log',
             remappings=[('/perception/obstacles_combined', '/perception/obstacles')]),
        Node(package='leader_search', executable='leader_search_node', name='leader_search',
             output='log', parameters=[search_params, {
                 'params_file': search_params,
                 'belief_rate_hz': ParameterValue(LaunchConfiguration('belief_rate_hz'),
                                                  value_type=float)}]),
        evaluator,
        RegisterEventHandler(OnProcessExit(target_action=evaluator,
                                           on_exit=[EmitEvent(event=Shutdown())])),
    ])
