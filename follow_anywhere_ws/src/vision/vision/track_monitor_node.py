"""Watches /vision/tracks and reports how stable the track IDs are (see track_metrics).

Runs on the real ROS path, so frame dropping and inference speed are part of the result. On
shutdown (Ctrl-C, or the launch file stopping when the video ends) it prints the summary and
writes the tracks in MOTChallenge format, so they can be scored against ground truth later.
"""
import os
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from vision_interfaces.msg import TrackArray

from vision.track_metrics import TrackStats


class TrackMonitorNode(Node):
    def __init__(self):
        super().__init__('track_monitor')
        p = self.declare_parameter
        p('crowd_size', 10)               # "crowded" means at least this many people in a frame
        p('short_s', 1.0)                 # s, tracks shorter than this count as flicker
        p('report_every', 5.0)            # s, running status line
        p('output', '')                   # MOT txt path, empty = ~/.ros/follow_anywhere/
        g = lambda name: self.get_parameter(name).value  # noqa: E731

        self.crowd_size = g('crowd_size')
        self.short_s = g('short_s')
        self.stats = TrackStats(self.crowd_size, self.short_s)
        self.output = os.path.expanduser(g('output')) or os.path.expanduser(
            time.strftime('~/.ros/follow_anywhere/tracks_%Y%m%d_%H%M%S.txt'))
        self.latency_sum = 0.0
        self.latency_max = 0.0

        self.create_subscription(TrackArray, '/vision/tracks', self.on_tracks, 50)
        self.create_timer(g('report_every'), self.report)
        self.get_logger().info(f'Monitoring /vision/tracks, MOT output: {self.output}')

    def on_tracks(self, msg: TrackArray):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        tracks = [(k.id, (k.x_min, k.y_min, k.x_max, k.y_max), k.confidence)
                  for k in msg.tracks if not k.occluded]       # only people seen this frame
        self.stats.add(t, tracks)
        now = self.get_clock().now().nanoseconds * 1e-9
        self.latency_sum += now - t
        self.latency_max = max(self.latency_max, now - t)

    def lines(self):
        s = self.stats.summary()
        fps = s.frames / s.seconds if s.seconds > 0 else 0.0
        mean_latency = self.latency_sum / s.frames * 1000 if s.frames else 0.0
        return (s.text(self.crowd_size, self.short_s),
                f'processed {fps:.1f} fps | latency camera->tracks mean {mean_latency:.0f} ms '
                f'max {self.latency_max * 1000:.0f} ms')

    def report(self):
        if self.stats.frames:
            self.get_logger().info(' || '.join(self.lines()))

    def finish(self):
        if not self.stats.frames:
            print('track_monitor: no tracks received', flush=True)
            return
        self.stats.write_mot(self.output)
        print('\n=== track_monitor summary ===\n' + '\n'.join(self.lines())
              + f'\nMOT tracks written to {self.output}', flush=True)


def main():
    rclpy.init()
    node = TrackMonitorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.finish()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
