"""calibrate_camera: measures where the RealSense sits on the Go2 (base_link -> camera_link) against the Go2 lidar and
saves it in perception_params.yaml (cameraX..cameraYaw). perception.launch.py publishes the saved values as the static
TF base_link -> camera_link, so the camera and the lidar agree from the next start of perception.

    ros2 run perception calibrate_camera
    ros2 run perception calibrate_camera --ros-args -p paramsFile:=/path/to/copy.yaml     # read and write only that file

- Needs the RealSense driver running (aligned depth, colour camera_info, TF camera_link -> camera_color_optical_frame;
  e.g. perception.launch.py) and the Go2 lidar /utlidar/cloud_base. The robot stands still in front of a room corner
  with a board leaned against a wall (scene: README, "Camera calibration").
- Stacks lidarCloudCount lidar clouds and the newest depthFrameCount depth frames of that time, aligns the camera points onto the lidar
  (camera_lidar_registration.py) starting from the saved mount, and saves the result only when the fit is good.
- Default file: this package's src/.../config/perception_params.yaml and the installed copy when it is a separate file
  (build without --symlink-install), so a restart uses the new mount without a rebuild. Settings: its
  camera_calibration section.
- Publishes /perception/calibration_cloud every second until Ctrl+C (lidar points intensity 0, camera points moved with
  the result intensity 1, base_link) and errors on /perception/debug.
"""

import dataclasses
import datetime
import math
import os
import re
from types import SimpleNamespace

import numpy as np
import rclpy
import tf2_ros
import yaml
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String

from perception.camera_lidar_registration import (CameraMount, AlignCameraToLidar, BuildCameraPoints,
                                                  DownsampleVoxels, FindQualityProblems, MakeCameraMount,
                                                  MakeMountTransform, MakeRotationFromQuaternion, MakeTransform,
                                                  MeasureMountChange, StackLidarPoints, TransformPoints)
from perception.coordinate_conversion import ConvertDepthImage

# Parameter defaults (set in the camera_calibration section of config/perception_params.yaml; paramsFile only with -p)
PARAMS_FILE = ''
CALIBRATION_SETTING_DEFAULTS = {
    'lidarCloudCount': 150,
    'depthFrameCount': 30,
    'depthMin': 0.5,
    'depthMax': 4.5,
    'voxelSize': 0.03,
    'patchSizes': [0.4, 0.2, 0.1],
    'maximumResidual': 0.025,
    'minimumMatchedPoints': 2000,
    'minimumDirectionPoints': 100,
    'maximumChangeDistance': 0.15,
    'maximumChangeAngle': 0.26,
}

# Params file
PARAMS_FILE_NAME = 'perception_params.yaml'
SOURCE_PARAMS_PATH = os.path.join('src', 'perception', 'config', PARAMS_FILE_NAME)
MOUNT_SECTION = 'perception_node'
CAMERA_MOUNT_NAMES = ('cameraX', 'cameraY', 'cameraZ', 'cameraRoll', 'cameraPitch', 'cameraYaw')
COMMENT_GAP = 2

# Frames
BASE_FRAME = 'base_link'
CAMERA_FRAME = 'camera_link'

# Progress and check cloud
PROGRESS_PERIOD_SECONDS = 1.0
SILENT_INPUT_SECONDS = 5.0
LIDAR_INTENSITY = 0.0
CAMERA_INTENSITY = 1.0
CLOUD_FIELD_NAMES = ('x', 'y', 'z', 'intensity')


class CameraCalibrationNode(Node):
    def __init__(self):
        super().__init__('camera_calibration')

        self.declare_parameter('paramsFile', PARAMS_FILE)
        self.paramsFilePaths = FindParamsFiles(self.get_parameter('paramsFile').value)
        fileSettings = LoadNodeSettings(self.paramsFilePaths[0], self.get_name())
        for settingName, defaultValue in CALIBRATION_SETTING_DEFAULTS.items():
            self.declare_parameter(settingName, fileSettings.get(settingName, defaultValue))
        self.settings = SimpleNamespace(
            **{settingName: self.get_parameter(settingName).value for settingName in CALIBRATION_SETTING_DEFAULTS})

        self.savedMount = LoadCameraMount(self.paramsFilePaths[0])
        self.imageBridge = CvBridge()
        self.lastDebugMessage = ''
        self.lidarClouds = []
        self.depthFrames = []
        self.depthFrameId = ''
        self.cameraInfo = None
        self.startSeconds = self.GetNowSeconds()
        self.lastLidarSeconds = None
        self.lastDepthSeconds = None
        self.lastCameraInfoSeconds = None
        self.calibrationFinished = False
        self.checkCloud = None
        self.transformBuffer = tf2_ros.Buffer()
        self.transformListener = tf2_ros.TransformListener(self.transformBuffer, self)

        # Output topics
        self.calibrationCloudPublisher = self.create_publisher(PointCloud2, '/perception/calibration_cloud', 1)
        self.debugPublisher = self.create_publisher(String, '/perception/debug', 10)

        # Input topics
        self.create_subscription(PointCloud2, '/utlidar/cloud_base', self.HandleLidarCloud, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.HandleDepthImage,
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.HandleCameraInfo,
                                 qos_profile_sensor_data)

        self.create_timer(PROGRESS_PERIOD_SECONDS, self.HandleProgressTimer)
        self.get_logger().info(f'CameraCalibration: saved mount {FormatMount(self.savedMount)} '
                               f'from {", ".join(self.paramsFilePaths)}. Keep the robot still.')

    def HandleLidarCloud(self, lidarCloudMessage):
        self.lastLidarSeconds = self.GetNowSeconds()
        if len(self.lidarClouds) >= self.settings.lidarCloudCount:
            return
        if lidarCloudMessage.header.frame_id != BASE_FRAME:
            self.PublishDebugMessage('CameraCalibration',
                                     f'lidar cloud in {lidarCloudMessage.header.frame_id}, expected {BASE_FRAME}')
            return
        self.lidarClouds.append(point_cloud2.read_points_numpy(lidarCloudMessage, field_names=('x', 'y', 'z'),
                                                               skip_nans=True))

    def HandleDepthImage(self, depthImageMessage):
        self.lastDepthSeconds = self.GetNowSeconds()
        lidarDone = len(self.lidarClouds) >= self.settings.lidarCloudCount
        if lidarDone and len(self.depthFrames) >= self.settings.depthFrameCount:
            return
        self.depthFrames.append(ConvertDepthImage(depthImageMessage, self.imageBridge))
        self.depthFrameId = depthImageMessage.header.frame_id
        if len(self.depthFrames) > self.settings.depthFrameCount:
            self.depthFrames.pop(0)

    def HandleCameraInfo(self, cameraInfoMessage):
        self.lastCameraInfoSeconds = self.GetNowSeconds()
        self.cameraInfo = cameraInfoMessage

    def HandleProgressTimer(self):
        if self.calibrationFinished:
            if self.checkCloud is not None:
                self.checkCloud.header.stamp = self.get_clock().now().to_msg()
                self.calibrationCloudPublisher.publish(self.checkCloud)
            return
        self.ReportSilentInputs()
        lidarDone = len(self.lidarClouds) >= self.settings.lidarCloudCount
        depthDone = len(self.depthFrames) >= self.settings.depthFrameCount
        if not (lidarDone and depthDone and self.cameraInfo is not None):
            self.get_logger().info(f'CameraCalibration: collecting, lidar {len(self.lidarClouds)}/'
                                   f'{self.settings.lidarCloudCount} clouds, depth {len(self.depthFrames)}/'
                                   f'{self.settings.depthFrameCount} frames')
            return
        opticalToCameraTransform = self.LookUpOpticalTransform()
        if opticalToCameraTransform is None:
            return
        self.calibrationFinished = True
        try:
            self.RunCalibration(opticalToCameraTransform)
        except Exception as error:
            self.PublishDebugMessage('CameraCalibration', f'calibration failed: {error}')

    def RunCalibration(self, opticalToCameraTransform):
        self.get_logger().info('CameraCalibration: data collected, aligning (robot may move again)')
        cameraMatrix = np.array(self.cameraInfo.k, dtype=np.float64).reshape(3, 3)
        cameraPoints = BuildCameraPoints(self.depthFrames, cameraMatrix, np.array(self.cameraInfo.d), self.settings)
        lidarPoints = StackLidarPoints(self.lidarClouds, self.settings)
        startTransform = MakeMountTransform(self.savedMount) @ opticalToCameraTransform
        result = AlignCameraToLidar(cameraPoints, lidarPoints, startTransform, self.settings)
        newMount = MakeCameraMount(result.transform @ np.linalg.inv(opticalToCameraTransform))
        changeDistance, changeAngle = MeasureMountChange(self.savedMount, newMount)
        self.LogResult(newMount, result, changeDistance, changeAngle)
        qualityProblems = FindQualityProblems(result, changeDistance, changeAngle, self.settings)
        if qualityProblems:
            self.PublishDebugMessage('CameraCalibration', 'not saved: ' + '; '.join(qualityProblems))
        else:
            self.SaveMount(newMount, result)
        self.checkCloud = BuildCheckCloud(DownsampleVoxels(lidarPoints, self.settings.voxelSize),
                                          TransformPoints(result.transform, cameraPoints))
        self.get_logger().info('CameraCalibration: check cloud on /perception/calibration_cloud (lidar intensity 0, '
                               'camera 1), Ctrl+C to quit')

    def LogResult(self, newMount, result, changeDistance, changeAngle):
        weakestX, weakestY, weakestZ = result.quality.weakestDirection
        self.get_logger().info(
            f'CameraCalibration: {result.cameraPointCount} camera points, {result.lidarPointCount} lidar points\n'
            f'  start:  {FormatQuality(result.startQuality)}\n'
            f'  result: {FormatQuality(result.quality)}, weakest direction ({weakestX:.2f}, {weakestY:.2f}, '
            f'{weakestZ:.2f})\n'
            f'  saved mount: {FormatMount(self.savedMount)}\n'
            f'  new mount:   {FormatMount(newMount)}\n'
            f'  change: {changeDistance * 1000:.0f} mm, {math.degrees(changeAngle):.1f} deg\n'
            f'  base_link <- camera_link:\n{np.array2string(MakeMountTransform(newMount), precision=5)}')

    def SaveMount(self, newMount, result):
        note = (f'MEASURED: calibrate_camera {datetime.datetime.now():%Y-%m-%d %H:%M}, residual '
                f'{result.quality.residual * 1000:.0f} mm, {result.quality.matchedPoints} points matched')
        for paramsFilePath in self.paramsFilePaths:
            try:
                SaveCameraMount(paramsFilePath, newMount, note)
                self.get_logger().info(f'CameraCalibration: saved to {paramsFilePath}')
            except (OSError, ValueError) as error:
                self.PublishDebugMessage('CameraCalibration', f'could not save to {paramsFilePath}: {error}')
        self.get_logger().info('CameraCalibration: restart perception.launch.py to use the new mount')

    def LookUpOpticalTransform(self):
        try:
            opticalTransform = self.transformBuffer.lookup_transform(CAMERA_FRAME, self.depthFrameId, Time())
        except tf2_ros.TransformException as error:
            self.PublishDebugMessage('CameraCalibration', f'no TF {CAMERA_FRAME} <- {self.depthFrameId} '
                                     f'(is the RealSense driver running?): {error}')
            return None
        return ConvertTransformToMatrix(opticalTransform.transform)

    def ReportSilentInputs(self):
        nowSeconds = self.GetNowSeconds()
        missingInputs = []
        for inputName, lastSeconds, stillNeeded in (
                ('/utlidar/cloud_base', self.lastLidarSeconds,
                 len(self.lidarClouds) < self.settings.lidarCloudCount),
                ('/camera/camera/aligned_depth_to_color/image_raw', self.lastDepthSeconds,
                 len(self.depthFrames) < self.settings.depthFrameCount),
                ('/camera/camera/color/camera_info', self.lastCameraInfoSeconds, self.cameraInfo is None)):
            lastHeardSeconds = lastSeconds if lastSeconds is not None else self.startSeconds
            if stillNeeded and nowSeconds - lastHeardSeconds > SILENT_INPUT_SECONDS:
                missingInputs.append(inputName)
        if missingInputs:
            self.PublishDebugMessage('CameraCalibration', f'nothing for {SILENT_INPUT_SECONDS:.0f} s on '
                                     f'{", ".join(missingInputs)} (Go2 connected? RealSense driver running?)')

    def GetNowSeconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def PublishDebugMessage(self, area, whatHappened):
        debugMessage = f'{area}: {whatHappened}'
        if debugMessage == self.lastDebugMessage:
            return
        self.lastDebugMessage = debugMessage
        self.get_logger().warning(debugMessage)
        self.debugPublisher.publish(String(data=debugMessage))


def FindParamsFiles(paramsFile):
    if paramsFile:
        return [os.path.realpath(os.path.expanduser(paramsFile))]
    installedPath = os.path.join(get_package_share_directory('perception'), 'config', PARAMS_FILE_NAME)
    paramsFilePaths = [os.path.realpath(installedPath)]
    folder = os.path.dirname(installedPath)
    while folder != os.path.dirname(folder):
        sourcePath = os.path.join(folder, SOURCE_PARAMS_PATH)
        if os.path.isfile(sourcePath):
            paramsFilePaths.insert(0, os.path.realpath(sourcePath))
            break
        folder = os.path.dirname(folder)
    return list(dict.fromkeys(paramsFilePaths))


def LoadNodeSettings(paramsFilePath, nodeName):
    with open(paramsFilePath) as paramsFile:
        fileContent = yaml.safe_load(paramsFile) or {}
    return (fileContent.get(nodeName) or {}).get('ros__parameters') or {}


def LoadCameraMount(paramsFilePath):
    mountSettings = LoadNodeSettings(paramsFilePath, MOUNT_SECTION)
    missingNames = [mountName for mountName in CAMERA_MOUNT_NAMES if mountName not in mountSettings]
    if missingNames:
        raise KeyError(f'{", ".join(missingNames)} missing in {MOUNT_SECTION} of {paramsFilePath}')
    return CameraMount(*(float(mountSettings[mountName]) for mountName in CAMERA_MOUNT_NAMES))


def SaveCameraMount(paramsFilePath, cameraMount, note):
    with open(paramsFilePath) as paramsFile:
        lines = paramsFile.read().split('\n')
    for mountName, mountValue in zip(CAMERA_MOUNT_NAMES, dataclasses.astuple(cameraMount)):
        linePattern = re.compile(rf'^(\s*){mountName}:')
        lineIndices = [lineIndex for lineIndex, line in enumerate(lines) if linePattern.match(line)]
        if len(lineIndices) != 1:
            raise ValueError(f'{mountName} found {len(lineIndices)} times in the file, expected once')
        oldLine = lines[lineIndices[0]]
        newLine = f'{linePattern.match(oldLine).group(1)}{mountName}: {mountValue:.5f}'
        if mountName == CAMERA_MOUNT_NAMES[0]:
            commentColumn = max(oldLine.find('#'), len(newLine) + COMMENT_GAP)
            newLine = f'{newLine:<{commentColumn}}# {note}'
        lines[lineIndices[0]] = newLine
    with open(paramsFilePath, 'w') as paramsFile:
        paramsFile.write('\n'.join(lines))


def ConvertTransformToMatrix(transform):
    rotation = transform.rotation
    translation = transform.translation
    return MakeTransform(MakeRotationFromQuaternion(rotation.x, rotation.y, rotation.z, rotation.w),
                         (translation.x, translation.y, translation.z))


def BuildCheckCloud(lidarPoints, cameraPoints):
    cloudFields = [PointField(name=fieldName, offset=4 * fieldIndex, datatype=PointField.FLOAT32, count=1)
                   for fieldIndex, fieldName in enumerate(CLOUD_FIELD_NAMES)]
    cloudPoints = np.vstack([np.column_stack([lidarPoints, np.full(len(lidarPoints), LIDAR_INTENSITY)]),
                             np.column_stack([cameraPoints, np.full(len(cameraPoints), CAMERA_INTENSITY)])])
    return point_cloud2.create_cloud(Header(frame_id=BASE_FRAME), cloudFields, cloudPoints.astype(np.float32))


def FormatMount(cameraMount):
    return (f'x {cameraMount.x:.3f} y {cameraMount.y:.3f} z {cameraMount.z:.3f} m, roll '
            f'{math.degrees(cameraMount.roll):.1f} pitch {math.degrees(cameraMount.pitch):.1f} yaw '
            f'{math.degrees(cameraMount.yaw):.1f} deg')


def FormatQuality(fitQuality):
    return (f'residual {fitQuality.residual * 1000:.1f} mm, {fitQuality.matchedPoints} points matched, '
            f'{fitQuality.directionPoints:.0f} direction points')


def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibrationNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
