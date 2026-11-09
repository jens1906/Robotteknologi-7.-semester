"""Start perception: RealSense driver, the camera's mounting TF and the perception node.

    ros2 launch perception perception.launch.py
    ros2 launch perception perception.launch.py startCamera:=false     # replaying a rosbag (bag has no TF)
    ros2 launch perception perception.launch.py startCamera:=false useSimTime:=true   # bag played with --clock
    ros2 launch perception perception.launch.py depthCloud:=true   # + a point cloud rebuilt from the aligned depth

The camera mounting (base_link -> camera_link) is cameraX..cameraYaw in the params file (written by
`ros2 run perception calibrate_camera`); it is read at every start, so a new calibration is used from the next start.

The Go2 lidar (/utlidar/cloud_base) and odometry (/utlidar/robot_odom) come from the robot itself;
nothing is started for them here. Topic names are written in perception_node.py: remap them in this file if the
camera uses another name.
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Camera mounting in the params file -> static_transform_publisher arguments
CAMERA_MOUNT_ARGUMENTS = {'cameraX': '--x', 'cameraY': '--y', 'cameraZ': '--z',
                          'cameraRoll': '--roll', 'cameraPitch': '--pitch', 'cameraYaw': '--yaw'}


def StartCameraMount(context):
    with open(LaunchConfiguration('paramsFile').perform(context)) as paramsFile:
        nodeParameters = yaml.safe_load(paramsFile)['perception_node']['ros__parameters']
    mountArguments = []
    for mountName, argumentName in CAMERA_MOUNT_ARGUMENTS.items():
        mountArguments += [argumentName, str(nodeParameters[mountName])]
    return [Node(package='tf2_ros', executable='static_transform_publisher', name='camera_mount_tf',
                 arguments=[*mountArguments, '--frame-id', 'base_link', '--child-frame-id', 'camera_link'])]


def generate_launch_description():
    share = get_package_share_directory('perception')
    args = [
        DeclareLaunchArgument('paramsFile',
                              default_value=os.path.join(share, 'config', 'perception_params.yaml'),
                              description='perception node parameters'),
        DeclareLaunchArgument('startCamera', default_value='true',
                              description='start the RealSense driver (false when replaying a rosbag)'),
        DeclareLaunchArgument('useSimTime', default_value='false',
                              description='use the /clock of a rosbag played with --clock (needed with odometry in the bag)'),
        DeclareLaunchArgument('depthCloud', default_value='false',
                              description='rebuild a point cloud from the aligned depth for the web viewer (costs CPU)'),
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
        condition=IfCondition(LaunchConfiguration('startCamera')))

    # base_link -> camera_link: connects the camera to the robot (the Go2 publishes no TF for it), from the params file
    camera_mount = OpaqueFunction(function=StartCameraMount)

    # camera_link -> camera_color_optical_frame (standard optical rotation: z forward, x right, y down).
    # Live the RealSense driver publishes it; a replayed rosbag has no TF, so it is published here instead.
    camera_optical = Node(
        package='tf2_ros', executable='static_transform_publisher', name='camera_optical_tf',
        arguments=['--roll', '-1.5708', '--yaw', '-1.5708',
                   '--frame-id', 'camera_link', '--child-frame-id', 'camera_color_optical_frame'],
        condition=UnlessCondition(LaunchConfiguration('startCamera')))

    # point cloud from the aligned depth (camera_info is found next to the image topic); for display only,
    # the bags record the depth image instead of this cloud
    depth_cloud = Node(
        package='depth_image_proc', executable='point_cloud_xyz_node', name='depth_cloud',
        remappings=[('image_rect', '/camera/camera/aligned_depth_to_color/image_raw'),
                    ('camera_info', '/camera/camera/aligned_depth_to_color/camera_info'),
                    ('points', '/camera/camera/aligned_depth_to_color/points')],
        parameters=[{'use_sim_time': LaunchConfiguration('useSimTime')}],
        condition=IfCondition(LaunchConfiguration('depthCloud')))

    perception = Node(
        package='perception', executable='perception_node', name='perception_node', output='screen',
        parameters=[LaunchConfiguration('paramsFile'), {'use_sim_time': LaunchConfiguration('useSimTime')}])

    return LaunchDescription([*args, realsense, camera_mount, camera_optical, depth_cloud, perception])
