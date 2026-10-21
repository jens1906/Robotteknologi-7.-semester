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
- Publish (all stamped with the camera image's time, frame base_link):
  /perception/leader_state, /perception/obstacles, /perception/tracked_people.
- Service /perception/calibrate_leader (follow_anywhere_interfaces/PerceptionCalibrateLeader): enrol the
  person with the given track id, -1 = the person nearest the image centre. Called by the viewer
  (perception_interface, click), the terminal, later a remote-control button.
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
from follow_anywhere_interfaces.msg import (PerceptionDetectorStats, PerceptionTrackedPerson,
                                            PerceptionTrackedPersonArray)
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from perception.person_detector import PersonDetector

# Parameter defaults (set in config/perception_params.yaml)
DETECTOR_MODEL_PATH = 'yolov8n-pose.pt'
DETECTOR_IMAGE_SIZE = 640
MINIMUM_BOX_SIZE_PIXELS = 24
TRACKER_CONFIG_FILE = 'botsort.yaml'
REQUESTED_DEVICE = 'auto'

# Image
COLOR_IMAGE_ENCODING = 'bgr8'


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        self.declare_parameter('detectorModelPath', DETECTOR_MODEL_PATH)
        self.declare_parameter('detectorImageSize', DETECTOR_IMAGE_SIZE)
        self.declare_parameter('minimumBoxSizePixels', MINIMUM_BOX_SIZE_PIXELS)
        self.declare_parameter('trackerConfigFile', TRACKER_CONFIG_FILE)
        self.declare_parameter('requestedDevice', REQUESTED_DEVICE)
        self.detectorModelPath = self.get_parameter('detectorModelPath').value
        self.detectorImageSize = self.get_parameter('detectorImageSize').value
        self.minimumBoxSizePixels = self.get_parameter('minimumBoxSizePixels').value
        self.trackerConfigFile = self.get_parameter('trackerConfigFile').value
        self.requestedDevice = self.get_parameter('requestedDevice').value

        self.imageBridge = CvBridge()
        self.lastDebugMessage = ''

        # Output topics
        self.trackedPeoplePublisher = self.create_publisher(PerceptionTrackedPersonArray, '/perception/tracked_people', 10)
        self.detectorStatsPublisher = self.create_publisher(PerceptionDetectorStats, '/perception/detector_stats', 10)
        self.debugPublisher = self.create_publisher(String, '/perception/debug', 10)

        trackerConfigPath = os.path.join(get_package_share_directory('perception'), 'config', self.trackerConfigFile)
        self.personDetector = PersonDetector(self.detectorModelPath, trackerConfigPath, self.detectorImageSize,
                                             self.minimumBoxSizePixels, self.requestedDevice)
        if self.personDetector.inferenceDeviceWarning:
            self.PublishDebugMessage('PersonDetector', self.personDetector.inferenceDeviceWarning)

        # Input topics
        newestImageQos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=1)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.HandleColorImage, newestImageQos)

    def HandleColorImage(self, colorImageMessage):
        try:
            imageBgr = self.imageBridge.imgmsg_to_cv2(colorImageMessage, COLOR_IMAGE_ENCODING)
            detectionStartSeconds = time.perf_counter()
            detectedPeople = self.personDetector.DetectAndTrackPeople(imageBgr)
            detectorTimeMilliseconds = (time.perf_counter() - detectionStartSeconds) * 1000.0
        except Exception as error:
            self.PublishDebugMessage('PersonDetector', f'detection failed: {error}')
            return
        self.trackedPeoplePublisher.publish(
            self.BuildTrackedPeopleMessage(detectedPeople, colorImageMessage.header))
        self.detectorStatsPublisher.publish(
            self.BuildDetectorStatsMessage(detectorTimeMilliseconds, len(detectedPeople), colorImageMessage.header))

    def BuildTrackedPeopleMessage(self, detectedPeople, header):
        trackedPeopleMessage = PerceptionTrackedPersonArray()
        trackedPeopleMessage.header = header
        for detectedPerson in detectedPeople:
            trackedPerson = PerceptionTrackedPerson()
            trackedPerson.id = detectedPerson.trackId
            trackedPerson.confidence = detectedPerson.confidence
            trackedPerson.x_min, trackedPerson.y_min, trackedPerson.x_max, trackedPerson.y_max = \
                detectedPerson.boundingBox
            trackedPerson.is_leader = False
            trackedPerson.reid_distance = math.nan
            trackedPerson.position_valid = False
            trackedPeopleMessage.people.append(trackedPerson)
        return trackedPeopleMessage

    def BuildDetectorStatsMessage(self, detectorTimeMilliseconds, peopleCount, header):
        detectorStatsMessage = PerceptionDetectorStats()
        detectorStatsMessage.header = header
        detectorStatsMessage.detector_time_ms = detectorTimeMilliseconds
        detectorStatsMessage.people_count = peopleCount
        return detectorStatsMessage

    def PublishDebugMessage(self, area, whatHappened):
        debugMessage = f'{area}: {whatHappened}'
        if debugMessage == self.lastDebugMessage:
            return
        self.lastDebugMessage = debugMessage
        self.get_logger().warning(debugMessage)
        self.debugPublisher.publish(String(data=debugMessage))


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
