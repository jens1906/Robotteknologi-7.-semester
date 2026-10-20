"""Frames from a video file or a folder of images (e.g. a MOTChallenge sequence) by index, or live from a ROS topic."""
import configparser
import glob
import os
import time

import cv2

IMAGE_EXT = ('.jpg', '.jpeg', '.png', '.bmp')


class FrameSource:
    def __init__(self, path):
        path = os.path.expanduser(path)
        self.fps = 30.0
        self.files = None
        if os.path.isdir(path):
            folder = os.path.join(path, 'img1')
            folder = folder if os.path.isdir(folder) else path
            self.files = sorted(f for f in glob.glob(os.path.join(folder, '*'))
                                if f.lower().endswith(IMAGE_EXT))
            if not self.files:
                raise FileNotFoundError(f'no images in {folder}')
            ini = os.path.join(path, 'seqinfo.ini')
            if os.path.isfile(ini):
                cfg = configparser.ConfigParser()
                cfg.read(ini)
                self.fps = cfg.getfloat('Sequence', 'frameRate', fallback=self.fps)
            self.count = len(self.files)
        else:
            if not os.path.isfile(path):
                raise FileNotFoundError(f'{path} (working directory {os.getcwd()})')
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise ValueError(f'cannot open video {path}')
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or self.fps
            self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._next = 0

    def frame(self, index):
        """BGR frame number `index` (0-based), or None past the end. In a video the frames are
        read in order, skipped ones are only grabbed."""
        if not 0 <= index < self.count:
            return None
        if self.files is not None:
            return cv2.imread(self.files[index])
        if index < self._next:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self._next = index
        while self._next < index:
            self.cap.grab()
            self._next += 1
        ok, frame = self.cap.read()
        self._next += 1
        return frame if ok else None


class TopicSource:
    """Live camera: the newest image of a ROS topic, e.g. the RealSense driver's colour image.

    Needs rclpy and cv_bridge, so it only works under `ros2 run` (system python). Images the
    pipeline is too slow for are dropped: it always gets the newest one.
    """
    count = 10 ** 9        # a camera has no end
    fps = 30.0

    def __init__(self, topic):
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        rclpy.init()
        self._rclpy, self._bridge, self._msg, self._t0 = rclpy, CvBridge(), None, None
        self._topic, self._seen, self._publishers = topic, 0, 1
        self._node = Node('vision_view')
        self._node.create_subscription(Image, topic, self._store, qos_profile_sensor_data)

    def _store(self, msg):
        self._msg = msg

    def next(self, timeout):
        """(BGR frame, seconds since the first one) of the newest image, or None if none came in `timeout` s."""
        self._rclpy.spin_once(self._node, timeout_sec=timeout)
        msg, self._msg = self._msg, None
        if msg is None:
            return None
        now = time.monotonic()
        self._t0 = self._t0 or now
        self._seen += 1
        if self._seen % 30 == 1:                       # now and then: is somebody else publishing on this topic?
            n = self._node.count_publishers(self._topic)
            if n > 1 and n != self._publishers:
                print(f'WARNING: {n} nodes publish on {self._topic}, their images get mixed '
                      f'(see: ros2 topic info -v {self._topic})', flush=True)
            self._publishers = n
        return self._bridge.imgmsg_to_cv2(msg, 'bgr8'), now - self._t0

    def close(self):
        self._node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()
