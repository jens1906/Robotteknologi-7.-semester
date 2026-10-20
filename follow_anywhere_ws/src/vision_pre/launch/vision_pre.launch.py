from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription,
                            RegisterEventHandler)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import AndSubstitution, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    yolo_model = DeclareLaunchArgument(
        'yolo_model', default_value='yolov8n.pt',
        description='Ultralytics weights, e.g. yolov8n.pt or yolo26n.pt (downloaded on first use)')
    use_dataset = DeclareLaunchArgument(
        'use_dataset', default_value='false',
        description='true: play video_path as the camera. false: use the RealSense')
    video_path = DeclareLaunchArgument(
        'video_path', default_value='',
        description='Video file or image folder, used when use_dataset:=true')
    loop = DeclareLaunchArgument(
        'loop', default_value='false', description='Repeat the video instead of stopping')
    video_width = DeclareLaunchArgument(
        'video_width', default_value='640',
        description='Resize the video to this width (RealSense profile is 640 wide), 0 = keep')
    monitor = DeclareLaunchArgument(
        'monitor', default_value='false',
        description='true: print track ID stability numbers; with use_dataset the launch ends '
                    'when the video ends')
    tracker_cfg = DeclareLaunchArgument(
        'tracker_cfg',
        default_value=PathJoinSubstitution([FindPackageShare('vision_pre'), 'config',
                                            'botsort_tuned.yaml']),
        description='Ultralytics tracker yaml: bytetrack.yaml, botsort.yaml or a path')

    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py'])),
        launch_arguments={
            'rgb_camera.color_profile': '640x480x30',
            'depth_module.depth_profile': '640x480x30',
            'align_depth.enable': 'true',
        }.items(),
        condition=UnlessCondition(LaunchConfiguration('use_dataset')))

    video_source = Node(
        package='vision_pre', executable='video_source', output='screen',
        parameters=[{
            'video_path': ParameterValue(LaunchConfiguration('video_path'), value_type=str),
            'loop': ParameterValue(LaunchConfiguration('loop'), value_type=bool),
            'width': ParameterValue(LaunchConfiguration('video_width'), value_type=int),
        }],
        condition=IfCondition(LaunchConfiguration('use_dataset')))

    detector = Node(package='vision_pre', executable='detector', output='screen',
                    parameters=[{
                        'yolo_model': LaunchConfiguration('yolo_model'),
                        'tracker_cfg': ParameterValue(LaunchConfiguration('tracker_cfg'),
                                                      value_type=str),
                    }])

    track_monitor = Node(package='vision_pre', executable='track_monitor', output='screen',
                         condition=IfCondition(LaunchConfiguration('monitor')))

    # When the video has played to the end, stop everything so the monitor prints its summary
    stop_at_end = RegisterEventHandler(
        OnProcessExit(target_action=video_source,
                      on_exit=[EmitEvent(event=Shutdown(reason='video finished'))]),
        condition=IfCondition(AndSubstitution(LaunchConfiguration('use_dataset'),
                                              LaunchConfiguration('monitor'))))

    # If the detector dies (missing package, bad model path), the rest has nothing to run for
    stop_on_detector_exit = RegisterEventHandler(
        OnProcessExit(target_action=detector,
                      on_exit=[EmitEvent(event=Shutdown(reason='detector exited'))]))

    return LaunchDescription([
        yolo_model, use_dataset, video_path, loop, video_width, monitor, tracker_cfg,
        realsense, video_source, track_monitor, detector, stop_at_end, stop_on_detector_exit])
