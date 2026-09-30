import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    go2_description_share = get_package_share_directory('go2_description')
    go2_sim_share = get_package_share_directory('go2_sim')
    ros_gz_sim_share = get_package_share_directory('ros_gz_sim')

    xacro_file = os.path.join(go2_description_share, 'urdf', 'go2_gazebo.urdf.xacro')
    bridge_config = os.path.join(go2_sim_share, 'config', 'bridge.yaml')
    world_file = os.path.join(go2_sim_share, 'config', 'go2_world.sdf')
    gait_config = os.path.join(go2_description_share, 'config', 'gait.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    robot_description = Command(['xacro ', xacro_file])

    # 1. Gazebo Harmonic simulator with our custom world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': ['-r -v 4 ', world_file],
            'on_exit_shutdown': 'true',
        }.items(),
    )

    # 2. robot_state_publisher — publishes TF from URDF + /joint_states
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time,
        }],
    )

    # 3. Spawn the Go2 into Gazebo
    #    -world must match go2_world.sdf <world name="go2_world">
    #    -name must match bridge.yaml joint state path (.../model/go2/...)
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_go2',
        output='screen',
        arguments=[
            '-world', 'go2_world',
            '-name', 'go2',
            '-param', 'robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.4',
        ],
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time,
        }],
    )

    # 4. ros_gz_bridge — all sensor topics, clock, and joint states
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        output='screen',
        parameters=[{
            'config_file': bridge_config,
            'use_sim_time': use_sim_time,
            'expand_gz_topic_names': False,
        }],
    )

    # 5. Load ros2_control controllers (delayed to let spawner settle)
    #    joint_state_broadcaster must load first — it publishes /joint_states
    load_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    load_joint_position_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=load_joint_state_broadcaster,
            on_exit=[Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_group_position_controller'],
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}],
            )],
        )
    )

    # 6. CHAMP gait controller — accepts /cmd_vel, outputs JointTrajectory
    champ_controller = Node(
        package='champ_base',
        executable='champ_controller',
        name='champ_controller',
        output='screen',
        parameters=[
            gait_config,
            {
                'use_sim_time': use_sim_time,
                'joint_controller_topic': 'joint_group_position_controller/command',
                'publish_joint_states': False,
                'publish_foot_contacts': True,
            },
        ],
    )

    # 7. go2_hw_bridge — converts /joint_states + /imu/data → /lowstate
    hw_bridge = Node(
        package='go2_hw_bridge',
        executable='lowstate_publisher',
        name='go2_hw_bridge',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Use simulation clock'),
        gazebo,
        robot_state_publisher,
        spawn_robot,
        bridge,
        # Controllers load after a short delay to let the controller_manager start
        TimerAction(period=3.0, actions=[load_joint_state_broadcaster]),
        load_joint_position_controller,
        TimerAction(period=5.0, actions=[champ_controller]),
        hw_bridge,
    ])
