# SPDX-License-Identifier: BSD-3-Clause

"""
go2_verify: live dashboard to confirm the Go2 (robot or sim) is producing data.

Shows the head camera (/frontvideostream, decoded H.264), the RealSense colour
and depth images, the lidar point cloud, state charts and per-topic rates.
rclpy spins in a background thread; the Qt GUI redraws at `refresh_hz`.
"""

import copy
import os
import signal
import threading
import time

from go2_verify.ros_io import Go2VerifyNode
from go2_verify.widgets import ChartPanel, ImageView, LidarView, RateTable
from pyqtgraph.Qt import QtCore, QtWidgets
import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.signals import SignalHandlerOptions

MODES = {0: 'idle', 1: 'balance stand', 2: 'pose', 3: 'locomotion', 5: 'lie down',
         6: 'joint lock', 7: 'damping', 8: 'recovery stand', 9: 'free walk'}


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, node: Go2VerifyNode, use_3d: bool):
        super().__init__()
        self.node = node
        self.setWindowTitle('Go2 verify')

        self.head = ImageView(f'Head camera {node.video_height}p')
        self.color = ImageView('RealSense colour')
        self.depth = ImageView('RealSense depth (0-5 m)', depth_max=5.0)
        self.lidar = LidarView(f'Lidar {node.lidar_topic}', use_3d=use_3d)

        self.chart_panel = ChartPanel([
            ('Body velocity', ['vx', 'vy', 'yaw rate'], 'm/s, rad/s'),
            ('Attitude', ['roll', 'pitch', 'yaw'], 'deg'),
            ('Height', ['body height', 'position z'], 'm'),
            ('Foot force', ['FR', 'FL', 'RR', 'RL'], ''),
            ('Power', ['V', 'A'], ''),
        ], node.history_s)
        self.charts = list(zip(self.chart_panel.charts, [
            node.velocity, node.attitude, node.height, node.foot_force, node.power]))
        self.rates = RateTable(node.rates.keys())

        cams = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        for w in (self.head, self.color, self.depth):
            cams.addWidget(w)

        bottom = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        bottom.addWidget(self.lidar)
        bottom.addWidget(self.chart_panel)
        bottom.addWidget(self.rates)
        bottom.setSizes([500, 600, 320])

        main = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        main.addWidget(cams)
        main.addWidget(bottom)
        main.setSizes([330, 600])
        self.setCentralWidget(main)
        self.status = self.statusBar()
        self.resize(1600, 950)

        self._last_rates = 0.0

    def refresh(self) -> None:
        node = self.node
        now = time.monotonic()
        # Snapshot under the lock (cheap copies), render outside it so callbacks never wait.
        with node.lock:
            head, color, depth, cloud = (copy.copy(x) for x in
                                         (node.head, node.color, node.depth, node.cloud))
            series = [s.array() for _, s in self.charts]
        self.head.update_from(head)
        self.color.update_from(color)
        self.depth.update_from(depth)
        self.lidar.update_from(cloud)
        t = now - node.t0
        for (chart, _), data in zip(self.charts, series):
            chart.update_from(data, t, node.history_s)
        if now - self._last_rates >= 1.0:
            self._last_rates = now
            self.rates.update_from(node.rate_snapshot())
            self.status.showMessage(self._status_text())

    def _status_text(self) -> str:
        node = self.node
        parts = []
        s = node.sport_state.value
        if s is not None:
            parts.append(f'mode {s.mode} ({MODES.get(s.mode, "?")}), gait {s.gait_type}, '
                         f'error {s.error_code}')
        low = node.low_state.value
        if low is not None:
            parts.append(f'battery {low.bms_state.soc}% {low.power_v:.1f} V')
        ls = node.lidar_state.value
        if ls is not None:
            parts.append(f'lidar {ls.cloud_frequency:.1f} Hz, '
                         f'loss {ls.cloud_packet_loss_rate:.2f}, err {ls.error_state}')
        r = node.response.value
        if r is not None:
            parts.append(f'last sport response api {r.header.identity.api_id} '
                         f'code {r.header.status.code}')
        return '   |   '.join(parts) or 'waiting for /lf/sportmodestate, /lf/lowstate ...'


def main(args=None):
    # The Gazebo container runs as root: its Fast DDS shared-memory segments in
    # /dev/shm are not usable by the host user, so samples never arrive although
    # discovery (UDP) works. Use UDP for data too. No effect with CycloneDDS.
    # Set before rclpy.init; the rate monitor child inherits it.
    os.environ.setdefault('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4')
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = Go2VerifyNode()
    use_3d = node.declare_parameter('lidar_3d', True).value
    refresh_hz = float(node.declare_parameter('refresh_hz', 30.0).value)

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    def spin():
        try:
            executor.spin()
        except ExternalShutdownException:
            pass

    threading.Thread(target=spin, daemon=True).start()

    app = QtWidgets.QApplication([])
    window = MainWindow(node, use_3d)
    window.show()

    timer = QtCore.QTimer()
    timer.timeout.connect(window.refresh)
    timer.start(int(1000 / refresh_hz))
    # Ctrl-C in the terminal closes the window (Qt otherwise swallows SIGINT).
    signal.signal(signal.SIGINT, lambda *_: app.quit())

    try:
        app.exec()
    finally:
        executor.shutdown()
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
