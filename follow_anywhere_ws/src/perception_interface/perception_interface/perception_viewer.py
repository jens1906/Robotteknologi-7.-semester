"""The viewer ROS node: shows what perception sees and decided, and calibrates the leader by click.

Runs on the laptop. Started with `ros2 launch perception_interface viewer.launch.py`.

To include:
- main(): rclpy init, create the node, spin; the OpenCV window is updated from a timer.
- Parameters from config/viewer_params.yaml.
- Subscriptions, all QoS best effort + keep last 1 (a slow WiFi skips frames instead of falling behind):
  - compressed colour image (sensor_msgs/CompressedImage, `.../image_raw/compressed`), decoded with
    cv2.imdecode. Never the raw image: ~27 MB/s over WiFi against ~1-2 MB/s compressed.
  - /perception/tracked_people and /perception/leader_state (small messages).
- No frame cap: shows as many frames as the network and the laptop manage, up to the camera's rate.
- Matching: keep the last image_buffer images by timestamp; a result (stamped with its camera image's
  time) is drawn on the image with the same stamp.
  - exact mode (default, B): only images that have a result are shown -> boxes exactly on people,
    shown at perception's rate.
  - live mode (A): every image, with the newest result -> smooth, boxes trail moving people a little.
  - key 'm' switches mode.
- Click on a person: call /perception/calibrate_leader (PerceptionCalibrateLeader) with that person's
  track id; key 'c': call it with -1 (person nearest the image centre). Show the service's message
  ("enrolling #7" / "nobody there") on screen for a few seconds.
- Keys: m mode, c calibrate centre, q quit.
- Header line: perception state, shown fps, image age (now - stamp), how many images were dropped.
- Warn on screen when no image or no perception result arrived for a while.
- Drawing itself is in overlay.py.
"""

import time
from collections import deque

import cv2
import numpy as np
import rclpy
from follow_anywhere_interfaces.msg import PerceptionLeaderState, PerceptionTrackedPersonArray
from follow_anywhere_interfaces.srv import PerceptionCalibrateLeader
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from perception_interface import overlay

# Parameter defaults (set in config/viewer_params.yaml)
EXACT_MODE = True
IMAGE_BUFFER_SIZE = 30
WINDOW_WIDTH = 960
SHOW_DISTANCES = True

# Window
WINDOW_NAME = 'perception'
WINDOW_EVENT_PERIOD_SECONDS = 0.01
NOTICE_SECONDS = 3.0
CLICK_MARKER_SECONDS = 1.5
SILENCE_SECONDS = 3.0
SILENCE_REPEAT_SECONDS = 1.0
FRAMES_PER_SECOND_SMOOTHING = 0.9
PLACEHOLDER_IMAGE_SHAPE = (480, 640, 3)

# Keys and calibration
QUIT_KEYS = (ord('q'), 27)
TOGGLE_MODE_KEY = ord('m')
CALIBRATE_CENTER_KEY = ord('c')
CENTER_PERSON_TRACK_ID = -1


class PerceptionViewer(Node):
    def __init__(self):
        super().__init__('perception_viewer')

        self.declare_parameter('exactMode', EXACT_MODE)
        self.declare_parameter('imageBufferSize', IMAGE_BUFFER_SIZE)
        self.declare_parameter('windowWidth', WINDOW_WIDTH)
        self.declare_parameter('showDistances', SHOW_DISTANCES)
        self.exactMode = self.get_parameter('exactMode').value
        self.imageBufferSize = self.get_parameter('imageBufferSize').value
        self.windowWidth = self.get_parameter('windowWidth').value
        self.showDistances = self.get_parameter('showDistances').value

        self.imageBuffer = deque(maxlen=self.imageBufferSize)
        self.latestTrackedPeople = None
        self.latestTrackedPeopleArrivalSeconds = 0.0
        self.latestLeaderState = None
        self.shownTrackedPeople = []
        self.lastShownImage = None
        self.lastShownSeconds = time.monotonic()
        self.lastSilenceReportSeconds = 0.0
        self.shownFramesPerSecond = 0.0
        self.windowScale = 1.0
        self.pendingWindowClicks = []
        self.clickMarkerPoint = None
        self.clickMarkerExpirySeconds = 0.0
        self.noticeText = ''
        self.noticeColor = overlay.NOTICE_COLOR
        self.noticeExpirySeconds = 0.0
        self.lastDebugMessage = ''

        # Output topics
        self.debugPublisher = self.create_publisher(String, '/perception_interface/debug', 10)

        # Input topics
        newestMessageQos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=1)
        self.create_subscription(CompressedImage, '/camera/camera/color/image_raw/compressed',
                                 self.HandleCompressedImage, newestMessageQos)
        self.create_subscription(PerceptionTrackedPersonArray, '/perception/tracked_people',
                                 self.HandleTrackedPeople, newestMessageQos)
        self.create_subscription(PerceptionLeaderState, '/perception/leader_state',
                                 self.HandleLeaderState, newestMessageQos)

        # Services
        self.calibrateLeaderClient = self.create_client(PerceptionCalibrateLeader, '/perception/calibrate_leader')

        self.create_timer(WINDOW_EVENT_PERIOD_SECONDS, self.HandleWindowEvents)
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self.HandleMouseEvent)

    def HandleCompressedImage(self, imageMessage):
        arrivalSeconds = time.monotonic()
        stampNanoseconds = ConvertStampToNanoseconds(imageMessage.header.stamp)
        self.imageBuffer.append((stampNanoseconds, arrivalSeconds, imageMessage))
        if not self.exactMode:
            trackedPeopleAgeSeconds = arrivalSeconds - self.latestTrackedPeopleArrivalSeconds
            freshTrackedPeople = self.latestTrackedPeople if trackedPeopleAgeSeconds < SILENCE_SECONDS else None
            self.DrawAndShowImage(imageMessage, freshTrackedPeople, trackedPeopleAgeSeconds)
        elif (self.latestTrackedPeople is not None
              and ConvertStampToNanoseconds(self.latestTrackedPeople.header.stamp) == stampNanoseconds):
            self.DrawAndShowImage(imageMessage, self.latestTrackedPeople, 0.0)

    def HandleTrackedPeople(self, trackedPeopleMessage):
        arrivalSeconds = time.monotonic()
        self.latestTrackedPeople = trackedPeopleMessage
        self.latestTrackedPeopleArrivalSeconds = arrivalSeconds
        if not self.exactMode:
            return
        stampNanoseconds = ConvertStampToNanoseconds(trackedPeopleMessage.header.stamp)
        for imageStampNanoseconds, imageArrivalSeconds, imageMessage in self.imageBuffer:
            if imageStampNanoseconds == stampNanoseconds:
                self.DrawAndShowImage(imageMessage, trackedPeopleMessage, arrivalSeconds - imageArrivalSeconds)
                return

    def HandleLeaderState(self, leaderStateMessage):
        self.latestLeaderState = leaderStateMessage

    def HandleMouseEvent(self, event, windowX, windowY, flags, parameter):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pendingWindowClicks.append((windowX, windowY))

    def HandleCalibrateLeaderAnswer(self, answerFuture):
        answer = answerFuture.result()
        self.SetNotice(answer.message, overlay.LEADER_COLOR if answer.success else overlay.NOTICE_COLOR)

    def HandleWindowEvents(self):
        pressedKey = cv2.waitKey(1) & 0xFF
        if pressedKey in QUIT_KEYS:
            rclpy.shutdown()
            return
        if pressedKey == TOGGLE_MODE_KEY:
            self.exactMode = not self.exactMode
        if pressedKey == CALIBRATE_CENTER_KEY:
            self.CallCalibrateLeader(CENTER_PERSON_TRACK_ID)
        while self.pendingWindowClicks:
            windowX, windowY = self.pendingWindowClicks.pop(0)
            self.HandleWindowClick(windowX, windowY)
        self.ReportNothingToShow(time.monotonic())

    def DrawAndShowImage(self, imageMessage, trackedPeopleMessage, resultDelaySeconds):
        image = cv2.imdecode(np.frombuffer(imageMessage.data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            self.PublishDebugMessage('Image', 'could not decode the compressed image')
            return
        self.shownTrackedPeople = trackedPeopleMessage.people if trackedPeopleMessage is not None else []
        overlay.DrawTrackedPeople(image, self.shownTrackedPeople, self.showDistances)
        overlay.DrawLeaderBanner(image, self.latestLeaderState)
        nowSeconds = time.monotonic()
        self.UpdateShownFramesPerSecond(nowSeconds)
        modeName = 'exact' if self.exactMode else 'live'
        overlay.DrawHeader(image, f'{modeName}  {self.shownFramesPerSecond:.1f} fps  '
                                  f'delay {resultDelaySeconds * 1000.0:.0f} ms  people {len(self.shownTrackedPeople)}')
        self.lastShownImage = image.copy()
        self.ShowFrameInWindow(image, nowSeconds)

    def ShowFrameInWindow(self, image, nowSeconds):
        if self.clickMarkerPoint is not None and nowSeconds < self.clickMarkerExpirySeconds:
            overlay.DrawClickMarker(image, *self.clickMarkerPoint)
        if nowSeconds < self.noticeExpirySeconds:
            overlay.DrawNotice(image, self.noticeText, self.noticeColor)
        self.windowScale = self.windowWidth / image.shape[1]
        cv2.imshow(WINDOW_NAME, cv2.resize(image, None, fx=self.windowScale, fy=self.windowScale))

    def UpdateShownFramesPerSecond(self, nowSeconds):
        secondsSinceLastFrame = nowSeconds - self.lastShownSeconds
        self.lastShownSeconds = nowSeconds
        if secondsSinceLastFrame > 0:
            self.shownFramesPerSecond = (FRAMES_PER_SECOND_SMOOTHING * self.shownFramesPerSecond
                                         + (1 - FRAMES_PER_SECOND_SMOOTHING) / secondsSinceLastFrame)

    def HandleWindowClick(self, windowX, windowY):
        imageX = windowX / self.windowScale
        imageY = windowY / self.windowScale
        self.clickMarkerPoint = (imageX, imageY)
        self.clickMarkerExpirySeconds = time.monotonic() + CLICK_MARKER_SECONDS
        clickedPerson = overlay.FindPersonAtPoint(self.shownTrackedPeople, imageX, imageY)
        if clickedPerson is None:
            self.SetNotice('nobody there: click inside a box', overlay.NOTICE_COLOR)
            return
        self.CallCalibrateLeader(clickedPerson.id)

    def CallCalibrateLeader(self, trackId):
        if not self.calibrateLeaderClient.service_is_ready():
            self.SetNotice(f'{self.calibrateLeaderClient.srv_name} is not available', overlay.NOTICE_COLOR)
            return
        request = PerceptionCalibrateLeader.Request()
        request.track_id = trackId
        self.calibrateLeaderClient.call_async(request).add_done_callback(self.HandleCalibrateLeaderAnswer)

    def SetNotice(self, text, color):
        self.noticeText = text
        self.noticeColor = color
        self.noticeExpirySeconds = time.monotonic() + NOTICE_SECONDS

    def ReportNothingToShow(self, nowSeconds):
        silentSeconds = nowSeconds - self.lastShownSeconds
        if silentSeconds < SILENCE_SECONDS or nowSeconds - self.lastSilenceReportSeconds < SILENCE_REPEAT_SECONDS:
            return
        self.lastSilenceReportSeconds = nowSeconds
        silenceText = f'nothing to show for {silentSeconds:.0f} s: are the camera and perception running?'
        self.PublishDebugMessage('Window', silenceText)
        image = np.zeros(PLACEHOLDER_IMAGE_SHAPE, np.uint8) if self.lastShownImage is None else self.lastShownImage.copy()
        overlay.DrawNotice(image, silenceText, overlay.NOTICE_COLOR)
        self.ShowFrameInWindow(image, nowSeconds)

    def PublishDebugMessage(self, area, whatHappened):
        debugMessage = f'{area}: {whatHappened}'
        if debugMessage == self.lastDebugMessage:
            return
        self.lastDebugMessage = debugMessage
        self.get_logger().warning(debugMessage)
        self.debugPublisher.publish(String(data=debugMessage))


def ConvertStampToNanoseconds(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def main(args=None):
    rclpy.init(args=args)
    viewer = PerceptionViewer()
    try:
        rclpy.spin(viewer)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        viewer.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
