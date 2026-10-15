"""RealSense D455 topics, named like realsense2_camera's defaults.

Not in the recorded bag, but mounted on the robot (go2_mounts.yaml: camera_link).

Subscribes:
  /sim/d455/color        (sensor_msgs/Image, rgb8)
  /sim/d455/depth        (sensor_msgs/Image, 32FC1 metres)
  /sim/d455/camera_info  (sensor_msgs/CameraInfo)

Publishes:
  /camera/camera/color/image_raw         rgb8, camera_color_optical_frame
  /camera/camera/color/camera_info
  /camera/camera/depth/image_rect_raw    16UC1 millimetres (0 = no data), camera_depth_optical_frame
  /camera/camera/depth/camera_info

Color and depth share one simulated sensor, so they are already aligned.
"""

import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from go2_robot_emulator.common import SIM_QOS, run

COLOR_FRAME = 'camera_color_optical_frame'
DEPTH_FRAME = 'camera_depth_optical_frame'


class RealSense(Node):

    def __init__(self):
        super().__init__('camera')
        self._info: CameraInfo | None = None
        self.create_subscription(Image, '/sim/d455/color', self._color_cb, SIM_QOS)
        self.create_subscription(Image, '/sim/d455/depth', self._depth_cb, SIM_QOS)
        self.create_subscription(CameraInfo, '/sim/d455/camera_info', self._info_cb, SIM_QOS)
        prefix = '/camera/camera'
        self.color_pub = self.create_publisher(Image, f'{prefix}/color/image_raw', 10)
        self.color_info_pub = self.create_publisher(CameraInfo, f'{prefix}/color/camera_info', 10)
        self.depth_pub = self.create_publisher(Image, f'{prefix}/depth/image_rect_raw', 10)
        self.depth_info_pub = self.create_publisher(CameraInfo, f'{prefix}/depth/camera_info', 10)
        self.get_logger().info('D455 started — /camera/camera/{color,depth}')

    def _info_cb(self, msg: CameraInfo) -> None:
        self._info = msg

    def _publish_info(self, pub, header) -> None:
        if self._info is None:
            return
        self._info.header = header
        pub.publish(self._info)

    def _color_cb(self, msg: Image) -> None:
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = COLOR_FRAME
        self.color_pub.publish(msg)
        self._publish_info(self.color_info_pub, msg.header)

    def _depth_cb(self, msg: Image) -> None:
        if msg.encoding != '32FC1':
            self.get_logger().warn(f'Unexpected depth encoding {msg.encoding}',
                                   throttle_duration_sec=5.0)
            return
        metres = np.frombuffer(bytes(msg.data), np.float32).reshape(msg.height, msg.width)
        mm = np.nan_to_num(metres * 1000.0, nan=0.0, posinf=0.0, neginf=0.0)
        out = Image()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = DEPTH_FRAME
        out.height = msg.height
        out.width = msg.width
        out.encoding = '16UC1'
        out.is_bigendian = 0
        out.step = msg.width * 2
        out.data = np.clip(mm, 0, 65535).astype('<u2').tobytes()
        self.depth_pub.publish(out)
        self._publish_info(self.depth_info_pub, out.header)


def main():
    run(RealSense)


if __name__ == '__main__':
    main()
