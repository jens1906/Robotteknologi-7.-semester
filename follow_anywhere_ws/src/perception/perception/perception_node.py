"""The perception ROS node: start-up, threads, publishing. The only file that creates topics (coordinate_conversion.py
uses ROS libraries, but creates no topics).

To include:
- main(): rclpy init, create the node, spin (entry point `perception_node` in setup.py).
- Read every parameter from config/perception_params.yaml and create the parts with them:
  SensorInput, PersonDetector, ReidModel, LeaderManager, CoordinateConversion, VelocityTracker,
  ObstacleMapper.
- Device: pick the GPU once ('auto' / 'cuda' / 'cpu'); warn loudly when it falls back to the CPU.
- Python packages: the node runs with the system python (plain `colcon build`); append the workspace
  .venv site-packages at the END of sys.path so only missing packages (ultralytics, onnxruntime, torch)
  come from the venv and the system numpy / cv_bridge keep priority.
- Two threads, so re-identification never stalls the video:
  1. frames: newest frame -> person_detector -> coordinate_conversion -> velocity_tracker ->
     obstacle_mapper -> publish tracked_people + obstacles.
  2. leader: re-identification for leader_manager (enrol / verify / search), using the newest people;
     its result is merged into the next published leader_state.
  (For now the leader update runs inline in the frame callback; the second thread comes when it is needed.)
- Inputs: colour image, aligned depth image (paired with the colour image by stamp), camera_info, and the TF
  base_link <- camera_color_optical_frame (camera mounting from the launch file + the RealSense driver).
- Publish (all stamped with the camera image's time, frame base_link):
  /perception/leader_state (the leader, with position / distance / angle / last seen position),
  /perception/obstacles (every other tracked person as a dynamic obstacle; the leader is not in it),
  /perception/tracked_people, /perception/frame_stats (cost of the frame, for PlotJuggler).
- No depth, camera_info or TF: positions are invalid (one debug line), tracking and the leader keep working.
- Service /perception/calibrate_leader (follow_anywhere_interfaces/PerceptionCalibrateLeader): enrol the
  person with the given track id, -1 = the person nearest the image centre. Called by the viewer
  (perception_interface, click), the terminal, later a remote-control button.
- Service /perception/clear_leader (std_srvs/Trigger): forget the leader and delete the saved file.
- The leader is always loaded at start (~/.ros/follow_anywhere/perception_leader.npz).
- Headless: no window here. The viewer runs in perception_interface on a laptop and reads the
  compressed image the RealSense driver publishes (raw images never leave the robot).
- Fill velocity fields with NaN and velocity_valid = false while use_velocity is off.
"""

import math
import os
import time
from types import SimpleNamespace

import rclpy
import tf2_ros
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from follow_anywhere_interfaces.msg import (PerceptionFrameStats, PerceptionLeaderState, PerceptionObstacle,
                                            PerceptionObstacleArray, PerceptionTrackedPerson,
                                            PerceptionTrackedPersonArray)
from follow_anywhere_interfaces.srv import PerceptionCalibrateLeader
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger

from perception.body_view import BACK_GROUP, FRONT_GROUP, PARTLY_HIDDEN_GROUP, SIDE_LEFT_GROUP, SIDE_RIGHT_GROUP
from perception.coordinate_conversion import (COORDINATE_SETTING_DEFAULTS, BuildCameraModel,
                                              CalculateDistanceAndAngle, ConvertDepthImage, PlacePerson)
from perception.leader_manager import (ENROLL_FAILED_EVENT, ENROLLED_EVENT, LEADER_SETTING_DEFAULTS, NO_EVENT,
                                       NO_TRACK_ID, LeaderManager)
from perception.person_detector import PersonDetector
from perception.reid_model import ReidModel

# Parameter defaults (set in config/perception_params.yaml, the leader settings have their defaults in leader_manager.py)
DETECTOR_MODEL_FILE = 'yolov8n-pose.pt'
DETECTOR_IMAGE_SIZE = 640
MINIMUM_BOX_SIZE_PIXELS = 24
TRACKER_CONFIG_FILE = 'botsort.yaml'
REQUESTED_DEVICE = 'auto'
REID_MODEL_FILE = 'osnet.onnx'
REID_THREADS = 2
LEADER_FILE = '~/.ros/follow_anywhere/perception_leader.npz'

# Image and calibration
COLOR_IMAGE_ENCODING = 'bgr8'
CENTER_PERSON_TRACK_ID = -1

# Depth and coordinates
BASE_FRAME = 'base_link'
DEPTH_BUFFER_SIZE = 10
DEPTH_MATCH_TOLERANCE_SECONDS = 0.05
PERSON_CLASS_NAME = 'person'


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        self.declare_parameter('detectorModelFile', DETECTOR_MODEL_FILE)
        self.declare_parameter('detectorImageSize', DETECTOR_IMAGE_SIZE)
        self.declare_parameter('minimumBoxSizePixels', MINIMUM_BOX_SIZE_PIXELS)
        self.declare_parameter('trackerConfigFile', TRACKER_CONFIG_FILE)
        self.declare_parameter('requestedDevice', REQUESTED_DEVICE)
        self.declare_parameter('reidModelFile', REID_MODEL_FILE)
        self.declare_parameter('reidThreads', REID_THREADS)
        self.declare_parameter('leaderFile', LEADER_FILE)
        for settingName, defaultValue in {**LEADER_SETTING_DEFAULTS, **COORDINATE_SETTING_DEFAULTS}.items():
            self.declare_parameter(settingName, defaultValue)
        self.detectorModelFile = self.get_parameter('detectorModelFile').value
        self.detectorImageSize = self.get_parameter('detectorImageSize').value
        self.minimumBoxSizePixels = self.get_parameter('minimumBoxSizePixels').value
        self.trackerConfigFile = self.get_parameter('trackerConfigFile').value
        self.requestedDevice = self.get_parameter('requestedDevice').value
        self.reidModelFile = self.get_parameter('reidModelFile').value
        self.reidThreads = self.get_parameter('reidThreads').value
        self.leaderFilePath = os.path.expanduser(self.get_parameter('leaderFile').value)
        leaderSettings = {settingName: self.get_parameter(settingName).value for settingName in LEADER_SETTING_DEFAULTS}
        self.coordinateSettings = SimpleNamespace(
            **{settingName: self.get_parameter(settingName).value for settingName in COORDINATE_SETTING_DEFAULTS},
            keypointConfidence=leaderSettings['keypointConfidence'])

        self.imageBridge = CvBridge()
        self.lastDebugMessage = ''
        self.latestDetectedPeople = []
        self.latestImageSeconds = 0.0
        self.latestImageWidth = 0
        self.latestImageHeight = 0
        self.reidSecondsInFrame = 0.0
        self.reidCropCountInFrame = 0
        self.depthImagesByStamp = {}
        self.cameraModel = None
        self.lastLeaderPosition = None
        self.transformBuffer = tf2_ros.Buffer()
        self.transformListener = tf2_ros.TransformListener(self.transformBuffer, self)

        # Output topics
        self.trackedPeoplePublisher = self.create_publisher(PerceptionTrackedPersonArray, '/perception/tracked_people', 10)
        self.leaderStatePublisher = self.create_publisher(PerceptionLeaderState, '/perception/leader_state', 10)
        self.obstaclesPublisher = self.create_publisher(PerceptionObstacleArray, '/perception/obstacles', 10)
        self.frameStatsPublisher = self.create_publisher(PerceptionFrameStats, '/perception/frame_stats', 10)
        self.debugPublisher = self.create_publisher(String, '/perception/debug', 10)

        # Services
        self.create_service(PerceptionCalibrateLeader, '/perception/calibrate_leader', self.HandleCalibrateLeader)
        self.create_service(Trigger, '/perception/clear_leader', self.HandleClearLeader)

        trackerConfigPath = os.path.join(get_package_share_directory('perception'), 'config', self.trackerConfigFile)
        self.personDetector = PersonDetector(self.ResolveModelPath(self.detectorModelFile), trackerConfigPath,
                                             self.detectorImageSize, self.minimumBoxSizePixels, self.requestedDevice)
        if self.personDetector.inferenceDeviceWarning:
            self.PublishDebugMessage('PersonDetector', self.personDetector.inferenceDeviceWarning)
        self.reidModel = ReidModel(self.ResolveModelPath(self.reidModelFile), self.requestedDevice, self.reidThreads)
        if self.reidModel.deviceWarning:
            self.PublishDebugMessage('ReidModel', self.reidModel.deviceWarning)
        self.leaderManager = LeaderManager(leaderSettings)
        if self.leaderManager.LoadLeader(self.leaderFilePath, self.reidModelFile):
            self.get_logger().info('LeaderManager: saved leader loaded, searching for them')
        elif os.path.exists(self.leaderFilePath):
            self.PublishDebugMessage('LeaderManager', 'saved leader is from an older version or another model, calibrate again')

        # Input topics
        newestImageQos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=1)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.HandleDepthImage,
                                 newestImageQos)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.HandleCameraInfo, newestImageQos)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.HandleColorImage, newestImageQos)

    def HandleColorImage(self, colorImageMessage):
        imageSeconds = ConvertStampToSeconds(colorImageMessage.header.stamp)
        try:
            imageBgr = self.imageBridge.imgmsg_to_cv2(colorImageMessage, COLOR_IMAGE_ENCODING)
            detectionStartSeconds = time.perf_counter()
            detectedPeople = self.personDetector.DetectAndTrackPeople(imageBgr)
            detectorTimeMilliseconds = (time.perf_counter() - detectionStartSeconds) * 1000.0
        except Exception as error:
            self.PublishDebugMessage('PersonDetector', f'detection failed: {error}')
            return
        self.latestDetectedPeople = detectedPeople
        self.latestImageSeconds = imageSeconds
        self.latestImageHeight, self.latestImageWidth = imageBgr.shape[:2]
        leaderEvent = self.UpdateLeader(detectedPeople, imageBgr, imageSeconds)
        self.HandleLeaderEvent(leaderEvent)
        peoplePositions = self.PlacePeople(detectedPeople, imageSeconds)
        leaderPosition = peoplePositions.get(self.leaderManager.visibleLeaderTrackId)
        if leaderPosition is not None:
            self.lastLeaderPosition = leaderPosition.position
        header = Header(stamp=colorImageMessage.header.stamp, frame_id=BASE_FRAME)
        self.leaderStatePublisher.publish(self.BuildLeaderStateMessage(imageSeconds, header, leaderPosition))
        self.trackedPeoplePublisher.publish(
            self.BuildTrackedPeopleMessage(detectedPeople, peoplePositions, imageSeconds, header))
        self.obstaclesPublisher.publish(self.BuildObstaclesMessage(detectedPeople, peoplePositions, header))
        self.frameStatsPublisher.publish(
            self.BuildFrameStatsMessage(detectorTimeMilliseconds, len(detectedPeople), header))

    def HandleDepthImage(self, depthImageMessage):
        self.depthImagesByStamp[ConvertStampToSeconds(depthImageMessage.header.stamp)] = depthImageMessage
        while len(self.depthImagesByStamp) > DEPTH_BUFFER_SIZE:
            self.depthImagesByStamp.pop(next(iter(self.depthImagesByStamp)))

    def HandleCameraInfo(self, cameraInfoMessage):
        self.cameraModel = BuildCameraModel(cameraInfoMessage)

    def HandleCalibrateLeader(self, request, response):
        if request.track_id == CENTER_PERSON_TRACK_ID:
            person = self.FindPersonNearestImageCenter()
        else:
            person = next((candidate for candidate in self.latestDetectedPeople
                           if candidate.trackId == request.track_id), None)
        if person is None:
            response.success = False
            response.message = 'nobody there'
            return response
        self.leaderManager.Enroll(person.trackId, self.latestImageSeconds)
        self.lastLeaderPosition = None
        response.success = True
        response.message = f'enrolling #{person.trackId}'
        return response

    def HandleClearLeader(self, request, response):
        self.leaderManager.Reset()
        self.lastLeaderPosition = None
        if os.path.exists(self.leaderFilePath):
            os.remove(self.leaderFilePath)
        response.success = True
        response.message = 'leader cleared'
        return response

    def UpdateLeader(self, detectedPeople, imageBgr, imageSeconds):
        self.reidSecondsInFrame = 0.0
        self.reidCropCountInFrame = 0
        try:
            return self.leaderManager.Update(
                detectedPeople, lambda trackIds: self.EmbedTrackedPeople(imageBgr, detectedPeople, trackIds),
                imageSeconds, imageBgr.shape[1])
        except Exception as error:
            self.PublishDebugMessage('LeaderManager', f'update failed: {error}')
            return NO_EVENT

    def EmbedTrackedPeople(self, imageBgr, detectedPeople, trackIds):
        embedStartSeconds = time.perf_counter()
        peopleByTrackId = {person.trackId: person for person in detectedPeople}
        embeddings = self.reidModel.EmbedPeople(imageBgr, [peopleByTrackId[trackId] for trackId in trackIds])
        self.reidSecondsInFrame += time.perf_counter() - embedStartSeconds
        self.reidCropCountInFrame += len(trackIds)
        return embeddings

    def HandleLeaderEvent(self, leaderEvent):
        if leaderEvent == NO_EVENT:
            return
        self.get_logger().info(f'LeaderManager: {leaderEvent} {self.leaderManager.eventReason}'.strip())
        if leaderEvent == ENROLL_FAILED_EVENT:
            self.PublishDebugMessage('LeaderManager', f'enrolment failed: {self.leaderManager.eventReason}')
        if leaderEvent == ENROLLED_EVENT:
            try:
                self.leaderManager.SaveLeader(self.leaderFilePath, self.reidModelFile)
            except OSError as error:
                self.PublishDebugMessage('LeaderManager', f'could not save the leader: {error}')

    def PlacePeople(self, detectedPeople, imageSeconds):
        if not detectedPeople:
            return {}
        depthImageMessage = self.FindDepthImage(imageSeconds)
        if depthImageMessage is None:
            self.PublishDebugMessage('CoordinateConversion', 'no depth image with the colour stamp, positions invalid')
            return {}
        if self.cameraModel is None:
            self.PublishDebugMessage('CoordinateConversion', 'no camera_info yet, positions invalid')
            return {}
        try:
            opticalToBaseTransform = self.transformBuffer.lookup_transform(
                BASE_FRAME, depthImageMessage.header.frame_id, Time())
        except tf2_ros.TransformException as error:
            self.PublishDebugMessage('CoordinateConversion', f'no TF {BASE_FRAME} <- {depthImageMessage.header.frame_id}: {error}')
            return {}
        try:
            depthMeters = ConvertDepthImage(depthImageMessage, self.imageBridge)
            peoplePositions = {}
            for person in detectedPeople:
                personPosition = PlacePerson(person, depthMeters, self.cameraModel, opticalToBaseTransform,
                                             self.coordinateSettings)
                if personPosition is not None:
                    peoplePositions[person.trackId] = personPosition
            return peoplePositions
        except Exception as error:
            self.PublishDebugMessage('CoordinateConversion', f'placing people failed: {error}')
            return {}

    def FindDepthImage(self, imageSeconds):
        nearestDepthSeconds = min(self.depthImagesByStamp, key=lambda depthSeconds: abs(depthSeconds - imageSeconds),
                                  default=None)
        if nearestDepthSeconds is None or abs(nearestDepthSeconds - imageSeconds) > DEPTH_MATCH_TOLERANCE_SECONDS:
            return None
        return self.depthImagesByStamp[nearestDepthSeconds]

    def BuildTrackedPeopleMessage(self, detectedPeople, peoplePositions, imageSeconds, header):
        trackedPeopleMessage = PerceptionTrackedPersonArray()
        trackedPeopleMessage.header = header
        for detectedPerson in detectedPeople:
            trackedPerson = PerceptionTrackedPerson()
            trackedPerson.id = detectedPerson.trackId
            trackedPerson.confidence = detectedPerson.confidence
            trackedPerson.x_min = detectedPerson.left
            trackedPerson.y_min = detectedPerson.top
            trackedPerson.x_max = detectedPerson.right
            trackedPerson.y_max = detectedPerson.bottom
            trackedPerson.is_leader = detectedPerson.trackId == self.leaderManager.visibleLeaderTrackId
            trackedPerson.reid_distance = self.leaderManager.GetRecentDistance(detectedPerson.trackId, imageSeconds)
            personPosition = peoplePositions.get(detectedPerson.trackId)
            trackedPerson.position_valid = personPosition is not None
            if personPosition is not None:
                trackedPerson.position = personPosition.position
            trackedPeopleMessage.people.append(trackedPerson)
        return trackedPeopleMessage

    def BuildLeaderStateMessage(self, imageSeconds, header, leaderPosition):
        leaderStateMessage = PerceptionLeaderState()
        leaderStateMessage.header = header
        leaderStateMessage.state = self.leaderManager.state
        leaderStateMessage.track_id = self.leaderManager.visibleLeaderTrackId
        leaderStateMessage.leader_visible = self.leaderManager.visibleLeaderTrackId != NO_TRACK_ID
        leaderStateMessage.seconds_since_seen = self.CalculateSecondsSinceSeen(imageSeconds)
        leaderStateMessage.distance = math.nan
        leaderStateMessage.angle = math.nan
        for point in (leaderStateMessage.position, leaderStateMessage.last_seen_position, leaderStateMessage.velocity):
            point.x = point.y = point.z = math.nan
        if leaderPosition is not None:
            leaderStateMessage.position = leaderPosition.position
            leaderStateMessage.distance, leaderStateMessage.angle = CalculateDistanceAndAngle(leaderPosition.position)
        if self.lastLeaderPosition is not None:
            leaderStateMessage.last_seen_position = self.lastLeaderPosition
        leaderStateMessage.velocity_valid = False
        leaderStateMessage.velocity_compensated = False
        storedViewCounts = self.leaderManager.CountStoredViews()
        leaderStateMessage.front_views = storedViewCounts[FRONT_GROUP]
        leaderStateMessage.back_views = storedViewCounts[BACK_GROUP]
        leaderStateMessage.side_left_views = storedViewCounts[SIDE_LEFT_GROUP]
        leaderStateMessage.side_right_views = storedViewCounts[SIDE_RIGHT_GROUP]
        leaderStateMessage.partly_hidden_views = storedViewCounts[PARTLY_HIDDEN_GROUP]
        return leaderStateMessage

    def BuildObstaclesMessage(self, detectedPeople, peoplePositions, header):
        obstaclesMessage = PerceptionObstacleArray()
        obstaclesMessage.header = header
        obstaclesMessage.lidar_used = False
        for person in detectedPeople:
            personPosition = peoplePositions.get(person.trackId)
            if personPosition is None or person.trackId == self.leaderManager.visibleLeaderTrackId:
                continue
            obstacle = PerceptionObstacle()
            obstacle.id = person.trackId
            obstacle.kind = PerceptionObstacle.DYNAMIC
            obstacle.source = PerceptionObstacle.SOURCE_CAMERA
            obstacle.class_name = PERSON_CLASS_NAME
            obstacle.position = personPosition.position
            obstacle.size.x = obstacle.size.y = personPosition.width
            obstacle.size.z = personPosition.height
            obstacle.position_uncertainty = personPosition.uncertainty
            obstacle.velocity.x = obstacle.velocity.y = obstacle.velocity.z = math.nan
            obstacle.velocity_valid = False
            obstacle.velocity_compensated = False
            obstaclesMessage.obstacles.append(obstacle)
        return obstaclesMessage

    def CalculateSecondsSinceSeen(self, imageSeconds):
        if self.leaderManager.lastSeenSeconds == 0.0:
            return math.nan
        return max(0.0, imageSeconds - self.leaderManager.lastSeenSeconds)

    def BuildFrameStatsMessage(self, detectorTimeMilliseconds, peopleCount, header):
        frameStatsMessage = PerceptionFrameStats()
        frameStatsMessage.header = header
        frameStatsMessage.detector_time_ms = detectorTimeMilliseconds
        frameStatsMessage.reid_time_ms = self.reidSecondsInFrame * 1000.0
        frameStatsMessage.reid_crop_count = self.reidCropCountInFrame
        frameStatsMessage.people_count = peopleCount
        return frameStatsMessage

    def FindPersonNearestImageCenter(self):
        imageCenterX = self.latestImageWidth / 2
        imageCenterY = self.latestImageHeight / 2
        return min(self.latestDetectedPeople,
                   key=lambda person: math.hypot((person.left + person.right) / 2 - imageCenterX,
                                                 (person.top + person.bottom) / 2 - imageCenterY),
                   default=None)

    def ResolveModelPath(self, modelFile):
        modelPath = os.path.join(get_package_share_directory('perception'), 'models', modelFile)
        if not os.path.isfile(modelPath):
            self.PublishDebugMessage(
                'Models', f'{modelFile} not found in {os.path.dirname(modelPath)}: run scripts/fetch_models.py, then colcon build')
            raise FileNotFoundError(modelPath)
        return modelPath

    def PublishDebugMessage(self, area, whatHappened):
        debugMessage = f'{area}: {whatHappened}'
        if debugMessage == self.lastDebugMessage:
            return
        self.lastDebugMessage = debugMessage
        self.get_logger().warning(debugMessage)
        self.debugPublisher.publish(String(data=debugMessage))


def ConvertStampToSeconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
