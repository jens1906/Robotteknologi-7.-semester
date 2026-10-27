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
     obstacle_mapper -> publish leader + obstacles + debug_frame.
  2. leader: re-identification for leader_manager (enrol / verify / search), using the newest people;
     its result is merged into the next published leader.
  (For now the leader update runs inline in the frame callback; the second thread comes when it is needed.)
  Odometry is newest-only (queue 1), like the images: with a deeper queue it fell ~0.35 s behind the ~0.1 s frames
  (one message per subscription per executor cycle). Its subscription is created first, so each cycle reads the
  newest odometry before the frame that uses it. A second executor thread is not used: under sim time it starved
  rclpy's /clock subscription (same callback group as the frames) and the clock stood still.
- Inputs: colour image, aligned depth image (paired with the colour image by stamp), camera_info, the TF
  base_link <- camera_color_optical_frame (camera mounting from the launch file + the RealSense driver), and the
  Go2's odometry /utlidar/robot_odom (odom -> base_link). The odometry is stamped by the Go2's own clock: the stamps
  are moved onto this computer's clock (ClockOffsetEstimator) and stored in this node's tf2 buffer (nothing is
  published on /tf), which gives the robot's pose at each image's time.
- No odometry yet: the robot counts as standing still. Odometry silent for odometryTimeout: one debug line, the
  robot's pose uncertainty grows (velocity_tracker). Odometry starting or coming back: velocity filters restart.
- Publish (all stamped with the camera image's time, frame base_link):
  Output: /perception/leader (status, closest point, velocity, uncertainties, lost duration, last known point),
  /perception/obstacles (every other tracked person: closest point to the robot and to the line of sight,
  velocity, uncertainties; the leader is not in it).
  Debug: /perception/debug_frame (boxes, leader details, cost of the frame; for the viewer and PlotJuggler),
  /perception/debug (error lines).
- No depth, camera_info or TF: positions are invalid (one debug line), tracking and the leader keep working.
- Service /perception/calibrate_leader (follow_anywhere_interfaces/PerceptionCalibrateLeader): enrol the
  person with the given track id, -1 = the person nearest the image centre. Called by the viewer
  (perception_interface, click), the terminal, later a remote-control button.
- Service /perception/clear_leader (std_srvs/Trigger): forget the leader and delete the saved file.
- The leader is always loaded at start (~/.ros/follow_anywhere/perception_leader.npz).
- Headless: no window here. The viewer runs in perception_interface on a laptop and reads the
  compressed image the RealSense driver publishes (raw images never leave the robot).
- Velocity and uncertainties: velocity_tracker (Kalman filter per person and one for the leader, in the odom frame,
  so velocities are over the ground), turned into base_link at the image's time before publishing. The published
  points come from the filtered centre; the leader's last known point is kept in odom so it stays in place while
  the robot moves.
"""

import math
import os
import time
from types import SimpleNamespace

import rclpy
import tf2_ros
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from follow_anywhere_interfaces.msg import (PerceptionDebugFrame, PerceptionDebugPerson, PerceptionLeader,
                                            PerceptionObstacle, PerceptionObstacleArray)
from geometry_msgs.msg import Point, TransformStamped, Vector3
from follow_anywhere_interfaces.srv import PerceptionCalibrateLeader
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger

from perception.body_view import BACK_GROUP, FRONT_GROUP, PARTLY_HIDDEN_GROUP, SIDE_LEFT_GROUP, SIDE_RIGHT_GROUP
from perception.coordinate_conversion import (COORDINATE_SETTING_DEFAULTS, BuildCameraModel,
                                              CalculateHorizontalDistance, ConvertDepthImage, PlacePerson)
from perception.leader_manager import (ENROLL_FAILED_EVENT, ENROLLED_EVENT, ENROLLING_STATE, IDLE_STATE,
                                       LEADER_SETTING_DEFAULTS, NO_EVENT, NO_TRACK_ID, LeaderManager)
from perception.obstacle_mapper import (OBSTACLE_SETTING_DEFAULTS, BuildPersonCircle, FindClosestPointToLine,
                                        FindClosestPointToRobot, MakeFloorPoint, MakeUnknownPoint, PersonCircle)
from perception.person_detector import PersonDetector
from perception.reid_model import ReidModel
from perception.velocity_tracker import (VELOCITY_SETTING_DEFAULTS, BuildMeasurement, ClockOffsetEstimator,
                                         ConvertEstimateToBase, ConvertPointToBase, ConvertPointToOdom, MakeRobotPose,
                                         MakeStillRobotPose, VelocityTracker)

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

# Robot odometry
ODOMETRY_FRAME = 'odom'
ODOMETRY_AUTHORITY = 'robot_odom'


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
        allSettingDefaults = {**LEADER_SETTING_DEFAULTS, **COORDINATE_SETTING_DEFAULTS, **OBSTACLE_SETTING_DEFAULTS,
                              **VELOCITY_SETTING_DEFAULTS}
        for settingName, defaultValue in allSettingDefaults.items():
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
        self.obstacleSettings = SimpleNamespace(
            **{settingName: self.get_parameter(settingName).value for settingName in OBSTACLE_SETTING_DEFAULTS})
        self.velocitySettings = SimpleNamespace(
            **{settingName: self.get_parameter(settingName).value for settingName in VELOCITY_SETTING_DEFAULTS})
        self.graceTime = leaderSettings['graceTime']

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
        self.cameraPosition = Point()
        self.lastLeaderPosition = None
        self.lastKnownLeaderPoint = None
        self.robotPose = MakeStillRobotPose()
        self.lastOdometrySeconds = None
        self.lastOdometryArrivalSeconds = None
        self.odometryRestarted = False
        self.velocityTracker = VelocityTracker(self.velocitySettings)
        self.clockOffsetEstimator = ClockOffsetEstimator(self.velocitySettings.clockOffsetWindow)
        self.transformBuffer = tf2_ros.Buffer()
        self.transformListener = tf2_ros.TransformListener(self.transformBuffer, self)

        # Output topics
        self.leaderPublisher = self.create_publisher(PerceptionLeader, '/perception/leader', 10)
        self.obstaclesPublisher = self.create_publisher(PerceptionObstacleArray, '/perception/obstacles', 10)
        self.debugFramePublisher = self.create_publisher(PerceptionDebugFrame, '/perception/debug_frame', 10)
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
        newestMessageQos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=1)
        self.create_subscription(Odometry, '/utlidar/robot_odom', self.HandleRobotOdometry, newestMessageQos)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.HandleDepthImage,
                                 newestMessageQos)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.HandleCameraInfo, newestMessageQos)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.HandleColorImage, newestMessageQos)

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
        self.robotPose = self.FindRobotPose(imageSeconds)
        peopleEstimates, peopleCircles, leaderEstimate, leaderCircle = self.TrackPeopleMotion(peoplePositions,
                                                                                               imageSeconds)
        leaderPosition = peoplePositions.get(self.leaderManager.visibleLeaderTrackId)
        leaderClosestPoint = None
        if leaderCircle is not None:
            leaderClosestPoint = FindClosestPointToRobot(leaderCircle)
            self.RememberLeaderPosition(leaderPosition, leaderClosestPoint)
        header = Header(stamp=colorImageMessage.header.stamp, frame_id=BASE_FRAME)
        self.leaderPublisher.publish(self.BuildLeaderMessage(leaderClosestPoint, leaderEstimate, imageSeconds, header))
        self.obstaclesPublisher.publish(
            self.BuildObstaclesMessage(detectedPeople, peopleEstimates, peopleCircles, leaderClosestPoint, header))
        self.debugFramePublisher.publish(self.BuildDebugFrameMessage(
            detectedPeople, peoplePositions, peopleEstimates, leaderPosition, detectorTimeMilliseconds, imageSeconds,
            header))

    def HandleDepthImage(self, depthImageMessage):
        self.depthImagesByStamp[ConvertStampToSeconds(depthImageMessage.header.stamp)] = depthImageMessage
        while len(self.depthImagesByStamp) > DEPTH_BUFFER_SIZE:
            self.depthImagesByStamp.pop(next(iter(self.depthImagesByStamp)))

    def HandleCameraInfo(self, cameraInfoMessage):
        self.cameraModel = BuildCameraModel(cameraInfoMessage)

    def HandleRobotOdometry(self, odometryMessage):
        arrivalSeconds = self.get_clock().now().nanoseconds * 1e-9
        if self.lastOdometryArrivalSeconds is None:
            self.get_logger().info('RobotOdometry: odometry started, velocities are now over the ground')
            self.odometryRestarted = True
        elif arrivalSeconds - self.lastOdometryArrivalSeconds > self.velocitySettings.odometryTimeout:
            self.PublishDebugMessage('RobotOdometry', 'odometry back, velocity filters restarted')
            self.clockOffsetEstimator.Reset()
            self.odometryRestarted = True
        self.lastOdometryArrivalSeconds = arrivalSeconds
        correctedSeconds = self.clockOffsetEstimator.CorrectStamp(
            ConvertStampToSeconds(odometryMessage.header.stamp), arrivalSeconds)
        odometryTransform = TransformStamped()
        odometryTransform.header.stamp = Time(seconds=correctedSeconds).to_msg()
        odometryTransform.header.frame_id = ODOMETRY_FRAME
        odometryTransform.child_frame_id = BASE_FRAME
        robotPosition = odometryMessage.pose.pose.position
        odometryTransform.transform.translation = Vector3(x=robotPosition.x, y=robotPosition.y, z=robotPosition.z)
        odometryTransform.transform.rotation = odometryMessage.pose.pose.orientation
        self.transformBuffer.set_transform(odometryTransform, ODOMETRY_AUTHORITY)
        self.lastOdometrySeconds = correctedSeconds

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
        self.ForgetLeaderPosition()
        response.success = True
        response.message = f'enrolling #{person.trackId}'
        return response

    def HandleClearLeader(self, request, response):
        self.leaderManager.Reset()
        self.ForgetLeaderPosition()
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
        cameraTranslation = opticalToBaseTransform.transform.translation
        self.cameraPosition = Point(x=cameraTranslation.x, y=cameraTranslation.y, z=cameraTranslation.z)
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

    def FindRobotPose(self, imageSeconds):
        if self.lastOdometrySeconds is None:
            return MakeStillRobotPose()
        silentSeconds = imageSeconds - self.lastOdometrySeconds
        lookupTime = Time() if silentSeconds > 0.0 else Time(seconds=imageSeconds)
        try:
            odometryTransform = self.transformBuffer.lookup_transform(ODOMETRY_FRAME, BASE_FRAME, lookupTime)
        except tf2_ros.TransformException:
            odometryTransform = self.transformBuffer.lookup_transform(ODOMETRY_FRAME, BASE_FRAME, Time())
        if silentSeconds > self.velocitySettings.odometryTimeout:
            self.PublishDebugMessage('RobotOdometry', 'no /utlidar/robot_odom, robot motion uncertain')
        robotTranslation = odometryTransform.transform.translation
        robotPose = MakeRobotPose(robotTranslation.x, robotTranslation.y,
                                  ConvertQuaternionToHeading(odometryTransform.transform.rotation), silentSeconds,
                                  self.velocitySettings)
        if self.odometryRestarted:
            self.odometryRestarted = False
            self.velocityTracker.ResetAll()
            self.MoveLastKnownLeaderPoint(robotPose)
        return robotPose

    def MoveLastKnownLeaderPoint(self, newRobotPose):
        lastKnownPoint = self.FindLastKnownLeaderPoint()
        if lastKnownPoint is not None:
            self.lastKnownLeaderPoint = MakeFloorPoint(*ConvertPointToOdom(newRobotPose, lastKnownPoint.x,
                                                                           lastKnownPoint.y))

    def TrackPeopleMotion(self, peoplePositions, imageSeconds):
        peopleMeasurements = {}
        measuredRadii = {}
        for trackId, personPosition in peoplePositions.items():
            measuredCircle = BuildPersonCircle(personPosition, self.cameraPosition, self.obstacleSettings)
            measuredRadii[trackId] = measuredCircle.radius
            peopleMeasurements[trackId] = BuildMeasurement(measuredCircle, personPosition.uncertainty, self.robotPose,
                                                           self.velocitySettings)
        peopleEstimates = {}
        peopleCircles = {}
        for trackId, motionEstimate in self.velocityTracker.UpdatePeople(peopleMeasurements, imageSeconds).items():
            peopleEstimates[trackId] = ConvertEstimateToBase(self.robotPose, motionEstimate)
            peopleCircles[trackId] = PersonCircle(peopleEstimates[trackId].positionX,
                                                  peopleEstimates[trackId].positionY, measuredRadii[trackId])
        leaderTrackId = self.leaderManager.visibleLeaderTrackId
        leaderEstimate = self.velocityTracker.UpdateLeader(peopleMeasurements.get(leaderTrackId), imageSeconds,
                                                           self.graceTime)
        if leaderEstimate is None:
            return peopleEstimates, peopleCircles, None, None
        leaderEstimate = ConvertEstimateToBase(self.robotPose, leaderEstimate)
        leaderCircle = None
        if leaderTrackId in peopleMeasurements:
            leaderCircle = PersonCircle(leaderEstimate.positionX, leaderEstimate.positionY, measuredRadii[leaderTrackId])
        return peopleEstimates, peopleCircles, leaderEstimate, leaderCircle

    def RememberLeaderPosition(self, leaderPosition, leaderClosestPoint):
        self.lastLeaderPosition = leaderPosition.position
        self.lastKnownLeaderPoint = MakeFloorPoint(*ConvertPointToOdom(self.robotPose, leaderClosestPoint.x,
                                                                       leaderClosestPoint.y))

    def FindLastKnownLeaderPoint(self):
        if self.lastKnownLeaderPoint is None:
            return None
        return MakeFloorPoint(*ConvertPointToBase(self.robotPose, self.lastKnownLeaderPoint.x,
                                                  self.lastKnownLeaderPoint.y))

    def ForgetLeaderPosition(self):
        self.lastLeaderPosition = None
        self.lastKnownLeaderPoint = None
        self.velocityTracker.ResetLeader()

    def BuildLeaderMessage(self, leaderClosestPoint, leaderEstimate, imageSeconds, header):
        leaderMessage = PerceptionLeader()
        leaderMessage.header = header
        leaderMessage.status = self.FindLeaderStatus()
        leaderMessage.closest_point = leaderClosestPoint if leaderClosestPoint is not None else MakeUnknownPoint()
        leaderMessage.lost_duration = self.CalculateSecondsSinceSeen(imageSeconds)
        lastKnownPoint = self.FindLastKnownLeaderPoint()
        leaderMessage.last_known_point = lastKnownPoint if lastKnownPoint is not None else MakeUnknownPoint()
        leaderMessage.position_uncertainty = math.nan
        leaderMessage.velocity_uncertainty = self.velocitySettings.unknownVelocitySigma
        if leaderEstimate is not None:
            leaderMessage.velocity = MakeVelocityVector(leaderEstimate)
            leaderMessage.position_uncertainty = leaderEstimate.positionSigma
            leaderMessage.velocity_uncertainty = leaderEstimate.velocitySigma
        return leaderMessage

    def FindLeaderStatus(self):
        if self.leaderManager.state in (IDLE_STATE, ENROLLING_STATE):
            return PerceptionLeader.NO_LEADER
        if self.leaderManager.visibleLeaderTrackId != NO_TRACK_ID:
            return PerceptionLeader.VISIBLE
        return PerceptionLeader.LOST

    def CalculateSecondsSinceSeen(self, imageSeconds):
        if self.leaderManager.lastSeenSeconds == 0.0:
            return math.nan
        return max(0.0, imageSeconds - self.leaderManager.lastSeenSeconds)

    def BuildObstaclesMessage(self, detectedPeople, peopleEstimates, peopleCircles, leaderClosestPoint, header):
        lineOfSightEnd = leaderClosestPoint if leaderClosestPoint is not None else self.FindLastKnownLeaderPoint()
        obstaclesMessage = PerceptionObstacleArray()
        obstaclesMessage.header = header
        for person in detectedPeople:
            motionEstimate = peopleEstimates.get(person.trackId)
            if motionEstimate is None or person.trackId == self.leaderManager.visibleLeaderTrackId:
                continue
            personCircle = peopleCircles[person.trackId]
            obstacle = PerceptionObstacle()
            obstacle.closest_point = FindClosestPointToRobot(personCircle)
            obstacle.line_of_sight_point = MakeUnknownPoint()
            if lineOfSightEnd is not None:
                obstacle.line_of_sight_point = FindClosestPointToLine(personCircle, self.cameraPosition, lineOfSightEnd)
            obstacle.velocity = MakeVelocityVector(motionEstimate)
            obstacle.position_uncertainty = motionEstimate.positionSigma
            obstacle.velocity_uncertainty = motionEstimate.velocitySigma
            obstaclesMessage.obstacles.append(obstacle)
        return obstaclesMessage

    def BuildDebugFrameMessage(self, detectedPeople, peoplePositions, peopleEstimates, leaderPosition,
                               detectorTimeMilliseconds, imageSeconds, header):
        debugFrameMessage = PerceptionDebugFrame()
        debugFrameMessage.header = header
        debugFrameMessage.odometry_age = math.nan
        if self.lastOdometrySeconds is not None:
            debugFrameMessage.odometry_age = max(0.0, imageSeconds - self.lastOdometrySeconds)
        debugFrameMessage.clock_offset = self.clockOffsetEstimator.offsetSeconds
        debugFrameMessage.leader_state = self.leaderManager.state
        debugFrameMessage.leader_track_id = self.leaderManager.visibleLeaderTrackId
        debugFrameMessage.seconds_since_seen = self.CalculateSecondsSinceSeen(imageSeconds)
        debugFrameMessage.leader_distance = math.nan
        if leaderPosition is not None:
            debugFrameMessage.leader_distance = CalculateHorizontalDistance(leaderPosition.position)
        debugFrameMessage.last_seen_position = MakeUnknownPoint()
        if self.lastLeaderPosition is not None:
            debugFrameMessage.last_seen_position = self.lastLeaderPosition
        storedViewCounts = self.leaderManager.CountStoredViews()
        debugFrameMessage.front_views = storedViewCounts[FRONT_GROUP]
        debugFrameMessage.back_views = storedViewCounts[BACK_GROUP]
        debugFrameMessage.side_left_views = storedViewCounts[SIDE_LEFT_GROUP]
        debugFrameMessage.side_right_views = storedViewCounts[SIDE_RIGHT_GROUP]
        debugFrameMessage.partly_hidden_views = storedViewCounts[PARTLY_HIDDEN_GROUP]
        debugFrameMessage.detector_time_ms = detectorTimeMilliseconds
        debugFrameMessage.reid_time_ms = self.reidSecondsInFrame * 1000.0
        debugFrameMessage.reid_crop_count = self.reidCropCountInFrame
        debugFrameMessage.people_count = len(detectedPeople)
        for detectedPerson in detectedPeople:
            debugPerson = PerceptionDebugPerson()
            debugPerson.id = detectedPerson.trackId
            debugPerson.confidence = detectedPerson.confidence
            debugPerson.x_min = detectedPerson.left
            debugPerson.y_min = detectedPerson.top
            debugPerson.x_max = detectedPerson.right
            debugPerson.y_max = detectedPerson.bottom
            debugPerson.is_leader = detectedPerson.trackId == self.leaderManager.visibleLeaderTrackId
            debugPerson.reid_distance = self.leaderManager.GetRecentDistance(detectedPerson.trackId, imageSeconds)
            personPosition = peoplePositions.get(detectedPerson.trackId)
            debugPerson.position_valid = personPosition is not None
            if personPosition is not None:
                debugPerson.position = personPosition.position
            motionEstimate = peopleEstimates.get(detectedPerson.trackId)
            debugPerson.velocity_uncertainty = math.nan
            if motionEstimate is not None:
                debugPerson.velocity = MakeVelocityVector(motionEstimate)
                debugPerson.velocity_uncertainty = motionEstimate.velocitySigma
            debugFrameMessage.people.append(debugPerson)
        return debugFrameMessage

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


def ConvertQuaternionToHeading(rotation):
    return math.atan2(2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
                      1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z))


def MakeVelocityVector(motionEstimate):
    return Vector3(x=motionEstimate.velocityX, y=motionEstimate.velocityY, z=0.0)


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
