"""Top-down window of the 2D sim: the world, the robot and what fake perception reports.

Ground truth (grey/coloured shapes, the robot triangle and its trail) is drawn
from /sim/*; /perception/leader and /perception/obstacles are drawn where perception
puts them, projected with the robot's true pose, so their error is visible.
rclpy spins in a background thread; the Qt GUI redraws at `refresh_hz`.

Keys: F follow the robot, D perception on/off, S lidar scan on/off,
      +/- zoom (while following).
"""

import collections
import math
import os
import signal
import threading
import time

import numpy as np
import pyqtgraph as pg
import rclpy
from follow_anywhere_interfaces.msg import PerceptionLeader, PerceptionObstacleArray
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan
from std_msgs.msg import UInt32

from go2_sim2d.common import yaw_of
from go2_sim2d.geometry import Circle
from go2_sim2d.perception import Params, lidar_limit
from go2_sim2d.world import PERSON_RADIUS, World

PERCEIVED_LEADER_COLOR = (230, 40, 160)
OBSTACLE_COLOR = (60, 120, 230)
LINE_OF_SIGHT_COLOR = (40, 170, 70)
LEADER_STATUS = {PerceptionLeader.NO_LEADER: 'no leader', PerceptionLeader.VISIBLE: 'visible',
                 PerceptionLeader.LOST: 'lost'}
LEADER_COLOR = (240, 120, 0)
PERSON_COLOR = (90, 90, 90)
ROBOT_COLOR = (20, 20, 20)
TRAIL_LENGTH = 2000


class ViewerNode(Node):
    """Keeps the latest message of every topic the viewer draws."""

    def __init__(self):
        super().__init__('sim2d_viewer')
        self.declare_parameter('world', 'open_field')
        self.declare_parameter('refresh_hz', 30.0)
        self.declare_parameter('robot_radius', 0.35)
        # Sensor fields to draw; same names and defaults as fake_perception's.
        defaults = Params()
        for name in ('lidar_fov_deg', 'lidar_range_max', 'lidar_x', 'lidar_height',
                     'lidar_tilt_deg', 'lidar_clear_fov_deg', 'camera_fov_deg',
                     'camera_range_max'):
            self.declare_parameter(name, getattr(defaults, name))
        self.world = World(self.get_parameter('world').value)
        self.lock = threading.Lock()
        self.robot = None
        self.robot_speed = 0.0
        self.people = []
        self.movers = []
        self.leader = None
        self.obstacles = None
        self.scan = None
        self.collisions = 0
        self.create_subscription(Odometry, '/sim/odometry', self._odom_cb, 10)
        self.create_subscription(PoseArray, '/sim/people', self._people_cb, 10)
        self.create_subscription(PoseArray, '/sim/movers', self._movers_cb, 10)
        self.create_subscription(PerceptionLeader, '/perception/leader', self._leader_cb, 10)
        self.create_subscription(PerceptionObstacleArray, '/perception/obstacles',
                                 self._obstacles_cb, 10)
        self.create_subscription(LaserScan, '/perception/scan', self._scan_cb, 10)
        self.create_subscription(UInt32, '/sim/collisions', self._collisions_cb, 10)

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        with self.lock:
            self.robot = (p.x, p.y, yaw_of(msg.pose.pose))
            self.robot_speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)

    def _people_cb(self, msg):
        with self.lock:
            self.people = [(p.position.x, p.position.y, yaw_of(p)) for p in msg.poses]

    def _movers_cb(self, msg):
        with self.lock:
            self.movers = [(p.position.x, p.position.y, yaw_of(p)) for p in msg.poses]

    def _leader_cb(self, msg):
        with self.lock:
            self.leader = msg

    def _obstacles_cb(self, msg):
        with self.lock:
            self.obstacles = msg

    def _scan_cb(self, msg):
        with self.lock:
            self.scan = msg

    def _collisions_cb(self, msg):
        with self.lock:
            self.collisions = msg.data


def _polygon_item(points, color, alpha=255):
    item = QtWidgets.QGraphicsPolygonItem(_polygon(points))
    item.setBrush(pg.mkBrush(*color, alpha))
    item.setPen(pg.mkPen(color=[int(c * 0.6) for c in color], width=1))
    return item


def _polygon(points):
    return QtGui.QPolygonF([QtCore.QPointF(float(x), float(y)) for x, y in points])


def _to_world(robot, p):
    """Point p (base_link) in odom; None if it is NaN."""
    if not (math.isfinite(p.x) and math.isfinite(p.y)):
        return None
    x, y, yaw = robot
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * p.x - s * p.y, y + s * p.x + c * p.y


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, node: ViewerNode):
        super().__init__()
        self.node = node
        world = node.world
        self.setWindowTitle(f'Go2 2D sim: {world.name}')
        self.follow = True
        self.half_width = 8.0
        self.trail = collections.deque(maxlen=TRAIL_LENGTH)
        self.t0 = time.monotonic()

        self.plot = pg.PlotWidget()
        self.plot.setBackground('w')
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel('bottom', 'x', 'm')
        self.plot.setLabel('left', 'y', 'm')
        add = self.plot.addItem

        for obj, shape in zip(world.static, world.static_shapes):
            color = [int(255 * c) for c in obj.get('color', [0.6, 0.6, 0.6])]
            add(_polygon_item(shape.corners(), color))
        self.mover_items = []
        for obj in world.movers:
            item = _polygon_item([(0, 0)], [int(255 * c) for c in obj.get('color', [0.6] * 3)])
            add(item)
            self.mover_items.append(item)

        # Sensor fields, drawn under everything that moves.
        self.lidar_field = pg.PlotDataItem(pen=pg.mkPen((120, 120, 220), width=1,
                                                        style=QtCore.Qt.PenStyle.DashLine))
        self.camera_field = QtWidgets.QGraphicsPolygonItem()
        self.camera_field.setBrush(pg.mkBrush(255, 220, 80, 50))
        self.camera_field.setPen(pg.mkPen((220, 170, 0), width=1))
        self.lidar_clear = pg.PlotDataItem(connect='pairs', pen=pg.mkPen(
            (120, 120, 220), width=1, style=QtCore.Qt.PenStyle.DotLine))
        add(self.lidar_field)
        add(self.lidar_clear)
        add(self.camera_field)

        self.scan_item = pg.ScatterPlotItem(size=3, pen=None, brush=pg.mkBrush(220, 30, 30, 160))
        self.trail_item = pg.PlotDataItem(pen=pg.mkPen((100, 100, 100), width=1))
        self.people_item = pg.ScatterPlotItem(pxMode=False, pen=pg.mkPen('k', width=1))
        self.robot_item = QtWidgets.QGraphicsPolygonItem()
        self.robot_item.setBrush(pg.mkBrush(*ROBOT_COLOR, 220))
        self.robot_item.setPen(pg.mkPen('k', width=1))
        self.robot_body = pg.PlotDataItem(pen=pg.mkPen(ROBOT_COLOR, width=1))
        self.obstacle_items = pg.ScatterPlotItem(
            size=9, pxMode=True, symbol='s', pen=pg.mkPen('k', width=1),
            brush=pg.mkBrush(*OBSTACLE_COLOR, 220))
        self.sight_items = pg.ScatterPlotItem(size=5, pxMode=True, pen=None,
                                              brush=pg.mkBrush(*LINE_OF_SIGHT_COLOR, 200))
        self.leader_item = pg.ScatterPlotItem(size=14, pxMode=True)
        self.velocity_items = pg.PlotDataItem(connect='pairs', pen=pg.mkPen((0, 0, 0), width=2))
        self.perception_items = (self.sight_items, self.velocity_items, self.obstacle_items,
                                 self.leader_item)
        for item in (self.scan_item, self.trail_item, self.people_item, self.robot_body,
                     self.robot_item) + self.perception_items:
            add(item)

        legend = (
            f'<span style="color: rgb{PERCEIVED_LEADER_COLOR}">&#9679; leader</span> '
            '(hollow: last known) '
            f'<span style="color: rgb{OBSTACLE_COLOR}">&#9632; obstacle</span> '
            f'<span style="color: rgb{LINE_OF_SIGHT_COLOR}">&#9679; line of sight</span>')
        self.info = QtWidgets.QLabel()
        help_text = QtWidgets.QLabel(
            f'Perception: {legend}   ·   ground truth: '
            f'<span style="color: rgb{LEADER_COLOR}">&#9679; leader</span> '
            f'<span style="color: rgb{PERSON_COLOR}">&#9679; people</span>   ·   '
            'keys: F follow, D perception, S scan, +/- zoom')
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.plot)
        layout.addWidget(self.info)
        layout.addWidget(help_text)
        central = QtWidgets.QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.resize(1100, 950)

    def keyPressEvent(self, event):
        key = event.key()
        Key = QtCore.Qt.Key
        if key == Key.Key_F:
            self.follow = not self.follow
        elif key == Key.Key_D:
            for item in self.perception_items:
                item.setVisible(not item.isVisible())
        elif key == Key.Key_S:
            self.scan_item.setVisible(not self.scan_item.isVisible())
        elif key in (Key.Key_Plus, Key.Key_Equal):
            self.half_width = max(2.0, self.half_width / 1.25)
        elif key == Key.Key_Minus:
            self.half_width = min(60.0, self.half_width * 1.25)
        else:
            super().keyPressEvent(event)

    def refresh(self):
        node = self.node
        with node.lock:
            robot, speed = node.robot, node.robot_speed
            people, movers = list(node.people), list(node.movers)
            leader_msg, obstacles = node.leader, node.obstacles
            scan, collisions = node.scan, node.collisions
        world = node.world

        for item, shape in zip(self.mover_items, world.mover_shapes(movers)):
            item.setPolygon(_polygon(shape.corners()))

        if people:
            colors = [LEADER_COLOR if i == world.leader_index else PERSON_COLOR
                      for i in range(len(people))]
            self.people_item.setData(
                pos=[(x, y) for x, y, _ in people], size=2 * PERSON_RADIUS,
                brush=[pg.mkBrush(*c, 200) for c in colors])

        if robot is None:
            self.info.setText('waiting for /sim/odometry ...')
            return
        x, y, yaw = robot
        self.trail.append((x, y))
        trail = np.array(self.trail)
        self.trail_item.setData(trail[:, 0], trail[:, 1])
        c, s = math.cos(yaw), math.sin(yaw)
        triangle = [(0.45, 0.0), (-0.3, 0.2), (-0.3, -0.2)]
        self.robot_item.setPolygon(_polygon([(x + c * a - s * b, y + s * a + c * b)
                                             for a, b in triangle]))
        body = Circle(x, y, node.get_parameter('robot_radius').value).corners(36)
        body = np.vstack((body, body[:1]))
        self.robot_body.setData(body[:, 0], body[:, 1])
        self._draw_fields(robot)
        if scan is not None:
            ranges = np.array(scan.ranges)
            bearings = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
            hit = np.isfinite(ranges)
            self.scan_item.setData(x + ranges[hit] * np.cos(yaw + bearings[hit]),
                                   y + ranges[hit] * np.sin(yaw + bearings[hit]))
        self._draw_perception(robot, leader_msg, obstacles)

        if self.follow:
            w = self.half_width
            self.plot.setRange(xRange=(x - w, x + w), yRange=(y - w, y + w), padding=0)

        leader = ''
        if world.leader_index is not None and len(people) > world.leader_index:
            lx, ly, _ = people[world.leader_index]
            leader = f'leader at {math.hypot(lx - x, ly - y):.2f} m   ·   '
        status = 'leader -'
        if leader_msg is not None:
            status = f'leader {LEADER_STATUS.get(leader_msg.status, "?")}'
            if leader_msg.status == PerceptionLeader.LOST:
                status += f' {leader_msg.lost_duration:.1f} s'
        seen = len(obstacles.obstacles) if obstacles is not None else 0
        self.info.setText(
            f't {time.monotonic() - self.t0:6.1f} s   ·   robot ({x:6.2f}, {y:6.2f}) '
            f'{math.degrees(yaw):6.1f} deg, {speed:.2f} m/s   ·   {leader}'
            f'{status}, {seen} obstacles   ·   collisions {collisions}')

    def _draw_fields(self, robot):
        x, y, yaw = robot
        p = self.node.get_parameter

        def wedge(fov_deg, r):
            half = math.radians(fov_deg) / 2
            a = yaw + np.linspace(-half, half, 64)
            arc = np.column_stack((x + r * np.cos(a), y + r * np.sin(a)))
            return arc if fov_deg >= 360 else np.vstack(([(x, y)], arc, [(x, y)]))

        # Lidar: reach per bearing from the sensor, and the edges of the leg-free sector.
        lp = Params(**{n: p(n).value for n in (
            'lidar_fov_deg', 'lidar_range_max', 'lidar_x', 'lidar_height', 'lidar_tilt_deg')})
        sx, sy = x + lp.lidar_x * math.cos(yaw), y + lp.lidar_x * math.sin(yaw)
        half = math.radians(min(lp.lidar_fov_deg, 360.0)) / 2
        b = np.linspace(-half, half, 181)
        r = lidar_limit(lp, b)
        lidar = np.column_stack((sx + r * np.cos(b + yaw), sy + r * np.sin(b + yaw)))
        if lp.lidar_fov_deg >= 360:
            lidar = np.vstack((lidar, lidar[:1]))
        else:
            lidar = np.vstack(([(sx, sy)], lidar, [(sx, sy)]))
        self.lidar_field.setData(lidar[:, 0], lidar[:, 1])
        edge = np.array([-1.0, 1.0]) * math.radians(p('lidar_clear_fov_deg').value) / 2
        er = lidar_limit(lp, edge)
        ends = np.column_stack((sx + er * np.cos(edge + yaw), sy + er * np.sin(edge + yaw)))
        clear = np.array([(sx, sy), ends[0], (sx, sy), ends[1]])
        self.lidar_clear.setData(clear[:, 0], clear[:, 1])
        self.camera_field.setPolygon(_polygon(
            wedge(p('camera_fov_deg').value, p('camera_range_max').value)))

    def _draw_perception(self, robot, leader, obstacles):
        yaw = robot[2]
        c, s = math.cos(yaw), math.sin(yaw)
        arrows = []

        def arrow(start, v):
            # 1 s of motion; velocity is in base_link
            if start is not None and (v.x or v.y):
                arrows.extend([start, (start[0] + c * v.x - s * v.y,
                                       start[1] + s * v.x + c * v.y)])

        if obstacles is not None:
            closest, sight = [], []
            for o in obstacles.obstacles:
                point = _to_world(robot, o.closest_point)
                if point is not None:
                    closest.append(point)
                    arrow(point, o.velocity)
                point = _to_world(robot, o.line_of_sight_point)
                if point is not None:
                    sight.append(point)
            self.obstacle_items.setData(pos=closest)
            self.sight_items.setData(pos=sight)

        spots = []
        if leader is not None:
            if leader.status == PerceptionLeader.VISIBLE:
                point = _to_world(robot, leader.closest_point)
                brush = pg.mkBrush(*PERCEIVED_LEADER_COLOR, 230)
            else:
                point = _to_world(robot, leader.last_known_point)
                brush = pg.mkBrush(0, 0, 0, 0)
            if point is not None:
                spots.append({'pos': point, 'brush': brush,
                              'pen': pg.mkPen(PERCEIVED_LEADER_COLOR, width=2)})
                if leader.status == PerceptionLeader.VISIBLE:
                    arrow(point, leader.velocity)
        self.leader_item.setData(spots)

        if arrows:
            arrows = np.array(arrows)
            self.velocity_items.setData(arrows[:, 0], arrows[:, 1])
        else:
            self.velocity_items.setData([], [])


def main(args=None):
    # The container runs as root: Fast DDS shared memory from it is not readable
    # by a host user. Use UDP for data too (as go2_verify does).
    os.environ.setdefault('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4')
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = ViewerNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    def spin():
        try:
            executor.spin()
        except ExternalShutdownException:
            pass

    threading.Thread(target=spin, daemon=True).start()

    app = QtWidgets.QApplication([])
    window = MainWindow(node)
    window.show()
    timer = QtCore.QTimer()
    timer.timeout.connect(window.refresh)
    timer.start(int(1000 / node.get_parameter('refresh_hz').value))
    # Ctrl-C in the terminal closes the window (Qt otherwise swallows SIGINT).
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    # Let Python run now and then so the SIGINT handler gets called.
    wake = QtCore.QTimer()
    wake.timeout.connect(lambda: None)
    wake.start(200)

    try:
        app.exec()
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
