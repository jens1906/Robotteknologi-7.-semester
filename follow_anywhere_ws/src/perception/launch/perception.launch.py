"""Start perception: RealSense driver, the camera's mounting TF and the perception node.

    ros2 launch perception perception.launch.py
    ros2 launch perception perception.launch.py start_camera:=false     # replaying a rosbag
    ros2 launch perception perception.launch.py camera_x:=0.28 camera_z:=0.10

The Go2 lidar (/utlidar/cloud_base) and odometry (/utlidar/robot_odom) come from the robot itself;
nothing is started for them here. Which topics are used is set in config/perception_params.yaml.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# where the RealSense sits on the Go2, relative to base_link (m, rad). Not measured yet: all 0.
CAMERA_MOUNT = {'camera_x': '0.0', 'camera_y': '0.0', 'camera_z': '0.0',
                'camera_roll': '0.0', 'camera_pitch': '0.0', 'camera_yaw': '0.0'}


def generate_launch_description():
    share = get_package_share_directory('perception')
    args = [
        DeclareLaunchArgument('params_file',
                              default_value=os.path.join(share, 'config', 'perception_params.yaml'),
                              description='perception node parameters'),
        DeclareLaunchArgument('start_camera', default_value='true',
                              description='start the RealSense driver (false when replaying a rosbag)'),
        *[DeclareLaunchArgument(name, default_value=value,
                                description='camera mounting on base_link (m / rad), measure on the robot')
          for name, value in CAMERA_MOUNT.items()],
    ]

    # colour + depth aligned to colour, 640x480 at 30 fps. Topics: /camera/camera/color/image_raw,
    # /camera/camera/aligned_depth_to_color/image_raw, /camera/camera/color/camera_info.
    # The driver also publishes the TF from camera_link to its optical frames.
    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')),
        launch_arguments={
            'enable_color': 'true',
            'enable_depth': 'true',
            'align_depth.enable': 'true',
            'rgb_camera.color_profile': '640x480x30',
            'depth_module.depth_profile': '640x480x30',
        }.items(),
        condition=IfCondition(LaunchConfiguration('start_camera')))

    # base_link -> camera_link: connects the camera to the robot (the Go2 publishes no TF for it)
    camera_mount = Node(
        package='tf2_ros', executable='static_transform_publisher', name='camera_mount_tf',
        arguments=['--x', LaunchConfiguration('camera_x'), '--y', LaunchConfiguration('camera_y'),
                   '--z', LaunchConfiguration('camera_z'), '--roll', LaunchConfiguration('camera_roll'),
                   '--pitch', LaunchConfiguration('camera_pitch'), '--yaw', LaunchConfiguration('camera_yaw'),
                   '--frame-id', 'base_link', '--child-frame-id', 'camera_link'])

    perception = Node(
        package='perception', executable='perception_node', name='perception_node', output='screen',
        parameters=[LaunchConfiguration('params_file')])

    return LaunchDescription([*args, realsense, camera_mount, perception])
