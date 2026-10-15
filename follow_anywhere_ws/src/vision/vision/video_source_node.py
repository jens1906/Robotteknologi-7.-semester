"""Plays a video file (or a folder of images) as the camera topics, instead of the RealSense.

Publishes the colour image and a synthetic CameraInfo on the RealSense topic names, in real time.
There is no depth, so the detector gives no 3D positions and the motion gate stays out of the way.
"""
import configparser
import glob
import math
import os
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

IMAGE_EXT = ('.jpg', '.jpeg', '.png', '.bmp')
HFOV_DEG = 69.4        # RealSense D4xx colour camera, only used for the synthetic CameraInfo


def resize_to_width(frame, width):
    """Scale a frame to `width` px keeping the aspect ratio (width 0 = leave it alone)."""
    if not width or frame.shape[1] == width:
        return frame
    scale = width / frame.shape[1]
    return cv2.resize(frame, (width, round(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)


class FrameSource:
    """Frames from a video file or from a folder of images (e.g. a MOTChallenge img1 folder)."""

    def __init__(self, path):
        path = os.path.expanduser(path)
        self.files = None
        self.fps = 30.0
        self.index = 0                    # number of the last frame read (MOT: 1-based)
        self.orig_size = None             # (width, height) of the stored frames
        if os.path.isdir(path):
            folder = os.path.join(path, 'img1')
            if not os.path.isdir(folder):
                folder = path
            self.files = sorted(f for f in glob.glob(os.path.join(folder, '*'))
                                if f.lower().endswith(IMAGE_EXT))
            if not self.files:
                raise FileNotFoundError(f'no images in {folder}')
            self.fps = self._seqinfo_fps(path, self.fps)
            self.i = 0
        else:
            if not os.path.isfile(path):
                raise FileNotFoundError(f'{path} (working directory {os.getcwd()})')
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise ValueError(f'cannot open video {path}')
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or self.fps

    @staticmethod
    def _seqinfo_fps(folder, default):
        ini = os.path.join(folder, 'seqinfo.ini')
        if not os.path.isfile(ini):
            return default
        cfg = configparser.ConfigParser()
        cfg.read(ini)
        return cfg.getfloat('Sequence', 'frameRate', fallback=default)

    def read(self):
        """Next BGR frame, or None at the end."""
        if self.files is None:
            ok, frame = self.cap.read()
            if not ok:
                return None
            self.index += 1
        else:
            if self.i >= len(self.files):
                return None
            frame = cv2.imread(self.files[self.i])
            self.i += 1
            stem = os.path.splitext(os.path.basename(self.files[self.i - 1]))[0]
            self.index = int(stem) if stem.isdigit() else self.i      # MOT names are the numbers
        self.orig_size = (frame.shape[1], frame.shape[0])
        return frame

    def rewind(self):
        self.index = 0
        if self.files is None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            self.i = 0


class VideoSourceNode(Node):
    def __init__(self):
        super().__init__('video_source')
        p = self.declare_parameter
        p('video_path', '')               # video file, or a folder of images
        p('loop', False)                  # False: play once, then exit
        p('fps', 0.0)                     # 0 = frame rate of the file
        p('width', 0)                     # px, 0 = keep; else resized keeping the aspect ratio
        p('wait_for_subscriber', True)    # do not start before the detector has loaded and listens
        p('end_delay', 3.0)               # s to stay up after the last frame (pipeline catches up)
        p('color_topic', '/camera/camera/color/image_raw')
        p('info_topic', '/camera/camera/color/camera_info')
        p('frame_id', 'camera_color_optical_frame')
        g = lambda name: self.get_parameter(name).value  # noqa: E731

        if not g('video_path'):
            raise ValueError('set the video_path parameter (video file or image folder)')
        self.source = FrameSource(g('video_path'))
        self.loop = g('loop')
        self.width = g('width')
        self.wait = g('wait_for_subscriber')
        self.end_delay = g('end_delay')
        self.frame_id = g('frame_id')
        self.fps = g('fps') or self.source.fps
        self.bridge = CvBridge()
        self.started = not self.wait
        self.n = 0
        self.end_at = None                # monotonic time at which the node may exit

        self.image_pub = self.create_publisher(Image, g('color_topic'), qos_profile_sensor_data)
        self.info_pub = self.create_publisher(CameraInfo, g('info_topic'), qos_profile_sensor_data)
        self.timer = self.create_timer(1 / self.fps, self.publish_next)
        self.get_logger().info(
            f'Playing {g("video_path")} at {self.fps:.1f} fps on {g("color_topic")}'
            + (', waiting for the detector...' if self.wait else ''))

    @property
    def finished(self):
        return self.end_at is not None and time.monotonic() >= self.end_at

    def camera_info(self, w, h, stamp):
        f = (w / 2) / math.tan(math.radians(HFOV_DEG) / 2)
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self.frame_id
        info.width, info.height = w, h
        info.distortion_model = 'plumb_bob'
        info.d = [0.0] * 5
        info.k = [f, 0.0, w / 2, 0.0, f, h / 2, 0.0, 0.0, 1.0]
        info.p = [f, 0.0, w / 2, 0.0, 0.0, f, h / 2, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def publish_next(self):
        if not self.started:
            if self.image_pub.get_subscription_count() == 0:
                return
            self.started = True
            self.get_logger().info('Detector is listening, starting playback')
        frame = self.source.read()
        if frame is None:
            if self.loop:
                self.source.rewind()
                return
            self.timer.cancel()
            self.end_at = time.monotonic() + self.end_delay
            self.get_logger().info(f'End of video after {self.n} frames')
            return
        frame = resize_to_width(frame, self.width)
        stamp = self.get_clock().now().to_msg()
        msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        self.image_pub.publish(msg)
        self.info_pub.publish(self.camera_info(frame.shape[1], frame.shape[0], stamp))
        self.n += 1


def main():
    rclpy.init()
    node = VideoSourceNode()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
