"""Head camera stream of the Go2 (/frontvideostream), H.264-encoded like the robot.

Subscribes:
  /sim/head_camera/image  (sensor_msgs/Image, rgb8 1280x720, 15 Hz)

Publishes:
  /frontvideostream  (unitree_go/Go2FrontVideoData)

What the robot sends (bag go2_all_topics_6_10_11_28): every camera frame as two
messages, height 360 and 720, with the same time_frame (units of 10 us, 6667
per frame at 15 fps). `data` is one H.264 Annex-B access unit, High profile,
an IDR with SPS/PPS every 30 frames, no B-frames.
"""

import time
from fractions import Fraction

import av
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from unitree_go.msg import Go2FrontVideoData

from go2_robot_emulator.common import ROBOT_QOS, SIM_QOS, run

FPS = 15
# (width, height, bit rate): average frame sizes on the robot are ~27 kB at 720p
# and ~8.7 kB at 360p, i.e. ~3.2 and ~1.0 Mbit/s.
STREAMS = [(640, 360, 1_000_000), (1280, 720, 3_200_000)]
X264_PARAMS = 'keyint=30:min-keyint=30:scenecut=0:bframes=0:repeat-headers=1'


def make_encoder(width: int, height: int, bit_rate: int) -> av.CodecContext:
    ctx = av.CodecContext.create('libx264', 'w')
    ctx.width = width
    ctx.height = height
    ctx.pix_fmt = 'yuv420p'
    ctx.time_base = Fraction(1, FPS)
    ctx.framerate = Fraction(FPS, 1)
    ctx.bit_rate = bit_rate
    ctx.options = {
        'preset': 'ultrafast',
        'tune': 'zerolatency',
        'profile': 'high',
        'x264-params': X264_PARAMS,
    }
    ctx.open()
    return ctx


class Video(Node):

    def __init__(self):
        super().__init__('frontvideostream')
        self._encoders = [(h, w, make_encoder(w, h, rate)) for w, h, rate in STREAMS]
        self._pts = 0
        self.create_subscription(Image, '/sim/head_camera/image', self._image_cb, SIM_QOS)
        self.pub = self.create_publisher(Go2FrontVideoData, '/frontvideostream', ROBOT_QOS)
        self.get_logger().info('frontvideostream started — 360p + 720p H.264')

    def _image_cb(self, msg: Image) -> None:
        if msg.encoding not in ('rgb8', 'bgr8'):
            self.get_logger().warn(f'Unsupported encoding {msg.encoding}', throttle_duration_sec=5.0)
            return
        pixels = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
        pixels = pixels[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        frame = av.VideoFrame.from_ndarray(pixels, format='rgb24' if msg.encoding == 'rgb8' else 'bgr24')

        time_frame = int(time.monotonic() * 100_000)
        for height, width, encoder in self._encoders:
            scaled = frame.reformat(width=width, height=height, format='yuv420p')
            scaled.pts = self._pts
            data = b''.join(bytes(p) for p in encoder.encode(scaled))
            if not data:
                continue
            out = Go2FrontVideoData()
            out.time_frame = time_frame
            out.height = height
            out.data = data
            self.pub.publish(out)
        self._pts += 1


def main():
    run(Video)


if __name__ == '__main__':
    main()
