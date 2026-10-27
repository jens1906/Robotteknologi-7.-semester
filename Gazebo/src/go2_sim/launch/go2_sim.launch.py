import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from go2_scenarios import scene


def world_actions(context):
    """Gazebo with the chosen world, and the scene_runner that animates it."""
    world = LaunchConfiguration('world').perform(context)
    # The world SDF is generated from go2_scenarios/scenes/<world>.yaml.
    sdf = tempfile.NamedTemporaryFile(
        'w', prefix=f'go2_{world}_', suffix='.sdf', delete=False)
    with sdf:
        sdf.write(scene.to_sdf(scene.load(world)))

    # headless:=true runs the Gazebo server only, rendering sensors offscreen
    # (no X11 needed), e.g. for tools/check_interface.py and replay_commands.py.
    headless = LaunchConfiguration('headless').perform(context) == 'true'
    gz_mode = '-s --headless-rendering ' if headless else ''

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': f'-r -v 4 {gz_mode}{sdf.name}',
            'on_exit_shutdown': 'true',
        }.items(),
    )

    # Moves the movers and publishes /sim/leader/pose and /sim/people.
    scene_runner = Node(
        package='go2_scenarios',
        executable='scene_runner',
        name='scene_runner',
        output='screen',
        parameters=[{'use_sim_time': True, 'world': world}],
    )
    return [gazebo, scene_runner]


def generate_launch_description():

    go2_description_share = get_package_share_directory('go2_description')
    go2_sim_share = get_package_share_directory('go2_sim')

    xacro_file = os.path.join(go2_description_share, 'urdf', 'go2_gazebo.urdf.xacro')
    bridge_config = os.path.join(go2_sim_share, 'config', 'bridge.yaml')
    gait_config = os.path.join(go2_description_share, 'config', 'gait.yaml')
    controllers_config = os.path.join(go2_description_share, 'config', 'ros2_controllers.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    robot_description = ParameterValue(Command(['xacro ', xacro_file]), value_type=str)

    # Gazebo resolves package://go2_description/... as model://go2_description/...,
    # so the parent of the package share dir must be on its resource path.
    gz_resource_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH', os.path.dirname(go2_description_share))

    # 1. Gazebo Jetty simulator with the chosen world, plus its scene_runner
    gazebo = OpaqueFunction(function=world_actions)

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
        # The real Go2 publishes no /tf; keep the sim's TF out of the way.
        # Robot code gets its TF from follow_anywhere_ws/go2_bringup on both.
        remappings=[('/tf', '/sim/tf'), ('/tf_static', '/sim/tf_static'),
                    ('joint_states', '/sim/joint_states')],
    )

    # 3. Spawn the Go2 into Gazebo
    #    -world is the <world name> of every generated world
    #    -name must match bridge.yaml joint state path (.../model/go2/...)
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_go2',
        output='screen',
        arguments=[
            '-world', scene.WORLD_NAME,
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
        arguments=['joint_state_broadcaster', '--param-file', controllers_config],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    load_joint_position_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=load_joint_state_broadcaster,
            on_exit=[Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_group_position_controller', '--param-file', controllers_config],
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}],
            )],
        )
    )

    # 6. CHAMP gait controller — accepts /cmd_vel, outputs JointTrajectory
    #    to the JointTrajectoryController's ~/joint_trajectory topic
    champ_controller = Node(
        package='champ_base',
        executable='quadruped_controller_node',
        name='champ_controller',
        output='screen',
        parameters=[
            gait_config,
            {
                'use_sim_time': use_sim_time,
                'urdf': robot_description,
                'gazebo': True,
                'joint_controller_topic': 'joint_group_position_controller/joint_trajectory',
                'publish_joint_control': True,
                'publish_joint_states': False,
                'publish_foot_contacts': True,
            },
        ],
        remappings=[('cmd_vel/smooth', 'cmd_vel')],
    )

    # 7. go2_robot_emulator — publishes the physical Go2's ROS interface
    #    (topics, types, frames, rates, wall-clock stamps) from the sim, and
    #    serves /api/sport/request, so robot code runs on the sim unchanged.
    #    Runs on wall-clock time like the robot, not on /clock.
    emulator = [
        Node(
            package='go2_robot_emulator',
            executable=executable,
            name=name,
            output='screen',
            parameters=[{'use_sim_time': False}],
        )
        for executable, name in [
            ('sport_api', 'sport_api'),
            ('state', 'go2_state'),
            ('lidar', 'utlidar'),
            ('video', 'frontvideostream'),
            ('realsense', 'camera'),
        ]
    ]

    # 8. sim_bridge — converts /api/sport/request (unitree_api) → /cmd_vel,
    #    so code written for the physical Go2 drives the sim unchanged
    sport_bridge = Node(
        package='sim_bridge',
        executable='sport_request_bridge',
        name='sim_bridge',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Use simulation clock'),
        DeclareLaunchArgument('headless', default_value='false',
                              description='Gazebo server only, no GUI window'),
        DeclareLaunchArgument('world', default_value='empty',
                              description='Scene from go2_scenarios/scenes: empty, '
                                          'open_field, office, plaza, warehouse'),
        gz_resource_path,
        gazebo,
        robot_state_publisher,
        spawn_robot,
        bridge,
        # Controllers load after a short delay to let the controller_manager start
        TimerAction(period=3.0, actions=[load_joint_state_broadcaster]),
        load_joint_position_controller,
        TimerAction(period=5.0, actions=[champ_controller]),
        *emulator,
    ])
