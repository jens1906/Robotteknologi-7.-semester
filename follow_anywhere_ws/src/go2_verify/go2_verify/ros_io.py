# SPDX-License-Identifier: BSD-3-Clause

"""
ROS side of go2_verify: subscribes to the Go2's sensors and keeps the latest data.

Callbacks only copy into buffers under a lock; the GUI thread reads them. Every
subscription has its own callback group, so a slow decoder never blocks the
state topics.
"""

import threading
import time

from go2_verify.h264 import H264Decoder
from go2_verify.rate_monitor import RateMonitor
import numpy as np
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from unitree_api.msg import Response
from unitree_go.msg import Go2FrontVideoData, LidarState, LowState, SportModeState

# Latest sample is all that matters for display; best effort matches the robot's
# reliable publishers and never stalls them.
LATEST_QOS = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        history=QoSHistoryPolicy.KEEP_LAST, depth=1)
# Rate counting needs every sample.
RATE_QOS = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                      history=QoSHistoryPolicy.KEEP_LAST, depth=50)
# H.264 P-frames depend on the previous frame: do not drop any.
VIDEO_QOS = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                       history=QoSHistoryPolicy.KEEP_LAST, depth=20)

VIDEO_SIZE = {360: (640, 360), 720: (1280, 720)}

# Topics only counted, in a child process (see rate_monitor): (topic, type, expected Hz).
# Rates from the bag.
RATE_ONLY = [
    ('/lowstate', 'unitree_go/msg/LowState', 450.0),
    ('/sportmodestate', 'unitree_go/msg/SportModeState', 260.0),
    ('/utlidar/robot_odom', 'nav_msgs/msg/Odometry', 130.0),
    ('/utlidar/imu', 'sensor_msgs/msg/Imu', 224.0),
    ('/utlidar/cloud', 'sensor_msgs/msg/PointCloud2', 14.0),
    ('/utlidar/cloud_deskewed', 'sensor_msgs/msg/PointCloud2', 14.0),
]

# ROS PointCloud2 datatype -> numpy
PC2_DTYPES = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
              5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}

POINT_LIMIT = 60000


class Series:
    """Fixed-size ring buffer of (t, values...)."""

    def __init__(self, n_values: int, size: int = 4000):
        self._buf = np.full((size, n_values + 1), np.nan)
        self._i = 0
        self._full = False

    def add(self, t: float, values) -> None:
        self._buf[self._i, 0] = t
        self._buf[self._i, 1:] = values
        self._i = (self._i + 1) % len(self._buf)
        self._full = self._full or self._i == 0

    def array(self) -> np.ndarray:
        if not self._full:
            return self._buf[:self._i].copy()
        return np.roll(self._buf, -self._i, axis=0)


class Latest:
    """Most recent value plus a sequence number, so the GUI redraws only on change."""

    def __init__(self):
        self.value = None
        self.seq = 0
        self.stamp = 0.0

    def put(self, value) -> None:
        self.value = value
        self.seq += 1
        self.stamp = time.monotonic()


def cloud_xyz(msg: PointCloud2) -> np.ndarray:
    """Return an (N, 3) float32 array of finite x, y, z from a PointCloud2."""
    fields = {f.name: f for f in msg.fields}
    dtype = np.dtype({
        'names': ['x', 'y', 'z'],
        'formats': [PC2_DTYPES[fields[k].datatype] for k in 'xyz'],
        'offsets': [fields[k].offset for k in 'xyz'],
        'itemsize': msg.point_step,
    })
    n = msg.width * msg.height
    pts = np.frombuffer(bytes(msg.data), dtype, count=n)
    xyz = np.stack([pts['x'], pts['y'], pts['z']], axis=1).astype(np.float32)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    return xyz[:POINT_LIMIT]


def image_array(msg: Image):
    """Image message -> ndarray (H, W, 3) uint8 RGB or (H, W) depth in metres."""
    data = np.frombuffer(bytes(msg.data), np.uint8)
    rows = data.reshape(msg.height, msg.step)
    enc = msg.encoding.lower()
    if enc in ('rgb8', 'bgr8'):
        img = rows[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        return img[..., ::-1] if enc == 'bgr8' else img
    if enc in ('rgba8', 'bgra8'):
        img = rows[:, :msg.width * 4].reshape(msg.height, msg.width, 4)[..., :3]
        return img[..., ::-1] if enc == 'bgra8' else img
    if enc in ('mono8', '8uc1'):
        return np.repeat(rows[:, :msg.width, None], 3, axis=2)
    if enc in ('16uc1', 'mono16'):
        return rows.view(np.uint16)[:, :msg.width].astype(np.float32) / 1000.0
    if enc == '32fc1':
        return rows.view(np.float32)[:, :msg.width].copy()
    return None


class Go2VerifyNode(Node):

    def __init__(self):
        super().__init__('go2_verify')
        p = self.declare_parameter
        self.video_height = p('video_height', 360).value
        self.lidar_topic = p('lidar_topic', '/utlidar/cloud_base').value
        self.history_s = float(p('history_s', 10.0).value)
        self.count_high_rate = p('count_high_rate', True).value
        state_topic = p('state_topic', '/lf/sportmodestate').value
        low_topic = p('lowstate_topic', '/lf/lowstate').value
        color_topic = p('color_topic', '/camera/camera/color/image_raw').value
        depth_topic = p('depth_topic', '/camera/camera/depth/image_rect_raw').value

        self.lock = threading.Lock()
        self.t0 = time.monotonic()
        # topic -> [count, last arrival, expected Hz]
        self.rates = {}

        self.head = Latest()
        self.color = Latest()
        self.depth = Latest()
        self.cloud = Latest()
        self.lidar_state = Latest()
        self.response = Latest()
        self.sport_state = Latest()
        self.low_state = Latest()

        self.velocity = Series(3)    # vx, vy, yaw rate
        self.attitude = Series(3)    # roll, pitch, yaw [deg]
        self.height = Series(2)      # body height, position z
        self.foot_force = Series(4)
        self.power = Series(2)       # V, A

        w, h = VIDEO_SIZE.get(self.video_height, (640, 360))
        self.decoder = H264Decoder(self._on_video_frame, w, h)

        self._sub(Go2FrontVideoData, '/frontvideostream', self._video_cb, VIDEO_QOS, 2 * 15.0)
        self._sub(Image, color_topic, self._image_cb(self.color), LATEST_QOS, 30.0)
        self._sub(Image, depth_topic, self._image_cb(self.depth), LATEST_QOS, 30.0)
        self._sub(PointCloud2, self.lidar_topic, self._cloud_cb, LATEST_QOS, 14.0)
        self._sub(SportModeState, state_topic, self._sport_cb, RATE_QOS, 18.0)
        self._sub(LowState, low_topic, self._low_cb, RATE_QOS, 18.0)
        self._sub(LidarState, '/utlidar/lidar_state', self._store(self.lidar_state),
                  RATE_QOS, 4.5)
        self._sub(Response, '/api/sport/response', self._store(self.response), RATE_QOS, 0.0)

        self.rate_monitor = None
        if self.count_high_rate:
            counted = [(t, typ) for t, typ, _ in RATE_ONLY if t not in self.rates]
            for topic, _, hz in RATE_ONLY:
                self.rates.setdefault(topic, [0, 0.0, hz])
            self.rate_monitor = RateMonitor(counted)

        self.get_logger().info(
            f'go2_verify: video {self.video_height}p via {self.decoder.backend}, '
            f'lidar {self.lidar_topic}')

    # -- helpers ---------------------------------------------------------------

    def rate_snapshot(self) -> dict:
        """Return {topic: [count, last arrival, expected Hz]} for all monitored topics."""
        with self.lock:
            rates = {k: list(v) for k, v in self.rates.items()}
        if self.rate_monitor is not None:
            for topic, (count, last) in self.rate_monitor.counts().items():
                rates[topic][:2] = [count, last]
        return rates

    def _sub(self, msg_type, topic, cb, qos, expected_hz):
        self.rates[topic] = [0, 0.0, expected_hz]

        def wrapped(msg):
            with self.lock:
                r = self.rates[topic]
                r[0] += 1
                r[1] = time.monotonic()
            if cb is not None:
                cb(msg)

        self.create_subscription(msg_type, topic, wrapped, qos,
                                 callback_group=MutuallyExclusiveCallbackGroup())

    def _now(self) -> float:
        return time.monotonic() - self.t0

    def _store(self, latest: Latest):
        def cb(msg):
            with self.lock:
                latest.put(msg)
        return cb

    def _image_cb(self, latest: Latest):
        def cb(msg):
            img = image_array(msg)
            if img is None:
                self.get_logger().warn(f'Unsupported image encoding {msg.encoding}',
                                       throttle_duration_sec=5.0)
                return
            with self.lock:
                latest.put(img)
        return cb

    # -- callbacks -------------------------------------------------------------

    def _video_cb(self, msg: Go2FrontVideoData):
        if msg.height == self.video_height:
            self.decoder.push(bytes(msg.data))

    def _on_video_frame(self, frame: np.ndarray):
        with self.lock:
            self.head.put(frame)

    def _cloud_cb(self, msg: PointCloud2):
        xyz = cloud_xyz(msg)
        with self.lock:
            self.cloud.put(xyz)

    def _sport_cb(self, msg: SportModeState):
        t = self._now()
        rpy = np.degrees(np.asarray(msg.imu_state.rpy, dtype=float))
        with self.lock:
            self.sport_state.put(msg)
            self.velocity.add(t, (msg.velocity[0], msg.velocity[1], msg.yaw_speed))
            self.attitude.add(t, rpy)
            self.height.add(t, (msg.body_height, msg.position[2]))

    def _low_cb(self, msg: LowState):
        t = self._now()
        with self.lock:
            self.low_state.put(msg)
            self.foot_force.add(t, msg.foot_force)
            self.power.add(t, (msg.power_v, msg.power_a))

    def close(self):
        self.decoder.close()
        if self.rate_monitor is not None:
            self.rate_monitor.close()
