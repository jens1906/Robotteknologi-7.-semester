"""The perception ROS node: start-up, threads, publishing. The only file besides sensor_input.py that uses ROS.

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
- Publish (all stamped with the camera image's time, frame base_link):
  /perception/leader_state, /perception/obstacles, /perception/tracked_people, /perception/frame_stats
  (cost of the frame, for PlotJuggler).
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

import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from follow_anywhere_interfaces.msg import (PerceptionFrameStats, PerceptionLeaderState, PerceptionTrackedPerson,
                                            PerceptionTrackedPersonArray)
from follow_anywhere_interfaces.srv import PerceptionCalibrateLeader
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

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
        for settingName, defaultValue in LEADER_SETTING_DEFAULTS.items():
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

        self.imageBridge = CvBridge()
        self.lastDebugMessage = ''
        self.latestDetectedPeople = []
        self.latestImageSeconds = 0.0
        self.latestImageWidth = 0
        self.latestImageHeight = 0
        self.reidSecondsInFrame = 0.0
        self.reidCropCountInFrame = 0

        # Output topics
        self.trackedPeoplePublisher = self.create_publisher(PerceptionTrackedPersonArray, '/perception/tracked_people', 10)
        self.leaderStatePublisher = self.create_publisher(PerceptionLeaderState, '/perception/leader_state', 10)
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

        # Input topics
        newestImageQos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=1)
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
        header = colorImageMessage.header
        self.leaderStatePublisher.publish(self.BuildLeaderStateMessage(imageSeconds, header))
        self.trackedPeoplePublisher.publish(self.BuildTrackedPeopleMessage(detectedPeople, imageSeconds, header))
        self.frameStatsPublisher.publish(
            self.BuildFrameStatsMessage(detectorTimeMilliseconds, len(detectedPeople), header))

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
        response.success = True
        response.message = f'enrolling #{person.trackId}'
        return response

    def HandleClearLeader(self, request, response):
        self.leaderManager.Reset()
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
                imageSeconds)
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

    def BuildTrackedPeopleMessage(self, detectedPeople, imageSeconds, header):
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
            trackedPerson.position_valid = False
            trackedPeopleMessage.people.append(trackedPerson)
        return trackedPeopleMessage

    def BuildLeaderStateMessage(self, imageSeconds, header):
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
        leaderStateMessage.velocity_valid = False
        leaderStateMessage.velocity_compensated = False
        return leaderStateMessage

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
