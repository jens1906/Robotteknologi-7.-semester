"""Unitree L1 lidar topics, from a dense Gazebo gpu_lidar scan.

Subscribes:
  /sim/utlidar/raw   (sensor_msgs/PointCloud2)  dense grid in utlidar_lidar
  /sim/odometry      (nav_msgs/Odometry)        for the odom-frame cloud

Publishes, matching the robot (bag go2_all_topics_6_10_11_28):
  /utlidar/cloud           utlidar_lidar, ~4150 points per frame, ~15 Hz
  /utlidar/cloud_base      base_link, same points minus hits on the robot body
  /utlidar/cloud_deskewed  odom, last `deskewed_frames` frames, x/y/z/intensity
  /utlidar/lidar_state     unitree_go/LidarState, ~4.5 Hz

The L1 scans a non-repetitive pattern, so each frame here is a random subset of
the dense grid. Point layout (32-byte step) is copied from the robot:
  x, y, z, intensity: float32 at 0, 4, 8, 16; ring: uint16 at 20; time: float32 at 24
ring is always 1 and time spans one frame (0 to ~0.068 s), as on the robot.
"""

import collections
import time

import numpy as np
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from unitree_go.msg import LidarState

from go2_robot_emulator.common import ROBOT_QOS, SIM_QOS, load_mount, quat_to_matrix, run

FULL_DTYPE = np.dtype({
    'names': ['x', 'y', 'z', 'intensity', 'ring', 'time'],
    'formats': ['<f4', '<f4', '<f4', '<f4', '<u2', '<f4'],
    'offsets': [0, 4, 8, 16, 20, 24],
    'itemsize': 32,
})
FULL_FIELDS = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='intensity', offset=16, datatype=PointField.FLOAT32, count=1),
    PointField(name='ring', offset=20, datatype=PointField.UINT16, count=1),
    PointField(name='time', offset=24, datatype=PointField.FLOAT32, count=1),
]
DESKEWED_DTYPE = np.dtype({
    'names': ['x', 'y', 'z', 'intensity'],
    'formats': ['<f4', '<f4', '<f4', '<f4'],
    'offsets': [0, 4, 8, 16],
    'itemsize': 32,
})
DESKEWED_FIELDS = FULL_FIELDS[:4]

POINTFIELD_DTYPES = {
    PointField.INT8: 'i1', PointField.UINT8: 'u1', PointField.INT16: '<i2',
    PointField.UINT16: '<u2', PointField.INT32: '<i4', PointField.UINT32: '<u4',
    PointField.FLOAT32: '<f4', PointField.FLOAT64: '<f8',
}

FRAME_DURATION = 0.068


def read_xyz(msg: PointCloud2) -> np.ndarray:
    fields = {f.name: f for f in msg.fields}
    dtype = np.dtype({
        'names': ['x', 'y', 'z'],
        'formats': [POINTFIELD_DTYPES[fields[n].datatype] for n in 'xyz'],
        'offsets': [fields[n].offset for n in 'xyz'],
        'itemsize': msg.point_step,
    })
    raw = np.frombuffer(bytes(msg.data), dtype=dtype, count=msg.width * msg.height)
    return np.stack([raw['x'], raw['y'], raw['z']], axis=1).astype(np.float64)


def make_cloud(points: np.ndarray, frame_id: str, stamp, fields) -> PointCloud2:
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = len(points)
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = points.dtype.itemsize
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = points.tobytes()
    return msg


class Lidar(Node):

    def __init__(self):
        super().__init__('utlidar')
        self.declare_parameter('points_per_frame', 4150)
        self.declare_parameter('max_range', 30.0)
        self.declare_parameter('deskewed_frames', 3)
        # Points inside this base_link box hit the robot itself; the robot
        # drops them from /utlidar/cloud_base.
        self.declare_parameter('body_box_min', [-0.45, -0.2, -0.25])
        self.declare_parameter('body_box_max', [0.45, 0.2, 0.2])

        self._n = self.get_parameter('points_per_frame').value
        self._max_range = self.get_parameter('max_range').value
        self._box_min = np.array(self.get_parameter('body_box_min').value)
        self._box_max = np.array(self.get_parameter('body_box_max').value)
        self._rot, self._trans = load_mount('utlidar_lidar')
        self._rng = np.random.default_rng()
        self._odom: Odometry | None = None
        self._recent = collections.deque(maxlen=self.get_parameter('deskewed_frames').value)
        self._frame_times = collections.deque(maxlen=30)
        self._last_size = 0

        self.create_subscription(PointCloud2, '/sim/utlidar/raw', self._cloud_cb, SIM_QOS)
        self.create_subscription(Odometry, '/sim/odometry', self._odom_cb, SIM_QOS)
        self.cloud_pub = self.create_publisher(PointCloud2, '/utlidar/cloud', ROBOT_QOS)
        self.base_pub = self.create_publisher(PointCloud2, '/utlidar/cloud_base', ROBOT_QOS)
        self.deskewed_pub = self.create_publisher(
            PointCloud2, '/utlidar/cloud_deskewed', ROBOT_QOS)
        self.state_pub = self.create_publisher(LidarState, '/utlidar/lidar_state', ROBOT_QOS)
        self.create_timer(1.0 / 4.5, self._publish_state)

        self.get_logger().info('utlidar started — /utlidar/cloud, /utlidar/cloud_base, '
                               '/utlidar/cloud_deskewed')

    def _odom_cb(self, msg: Odometry) -> None:
        self._odom = msg

    def _cloud_cb(self, msg: PointCloud2) -> None:
        xyz = read_xyz(msg)
        ranges = np.linalg.norm(xyz, axis=1)
        valid = np.isfinite(ranges) & (ranges < self._max_range)
        xyz, ranges = xyz[valid], ranges[valid]
        n = min(len(xyz), int(self._n + self._rng.integers(-50, 51)))
        pick = self._rng.choice(len(xyz), size=n, replace=False)
        xyz, ranges = xyz[pick], ranges[pick]

        points = np.zeros(n, dtype=FULL_DTYPE)
        points['x'], points['y'], points['z'] = xyz.T
        # Gazebo has no reflectivity; fall off with range like a typical return.
        points['intensity'] = np.clip(255.0 - 25.0 * ranges, 10.0, 255.0).round()
        points['ring'] = 1
        points['time'] = np.sort(self._rng.uniform(0.0, FRAME_DURATION, n))

        stamp = self.get_clock().now().to_msg()
        self._frame_times.append(time.monotonic())
        self._last_size = n
        self.cloud_pub.publish(make_cloud(points, 'utlidar_lidar', stamp, FULL_FIELDS))

        in_base = xyz @ self._rot.T + self._trans
        outside = ~np.all((in_base > self._box_min) & (in_base < self._box_max), axis=1)
        base_points = points[outside].copy()
        base_points['x'], base_points['y'], base_points['z'] = in_base[outside].T
        self.base_pub.publish(make_cloud(base_points, 'base_link', stamp, FULL_FIELDS))

        if self._odom is not None:
            pose = self._odom.pose.pose
            q, p = pose.orientation, pose.position
            in_odom = in_base @ quat_to_matrix(q.x, q.y, q.z, q.w).T + (p.x, p.y, p.z)
            frame = np.zeros(n, dtype=DESKEWED_DTYPE)
            frame['x'], frame['y'], frame['z'] = in_odom.T
            frame['intensity'] = points['intensity']
            self._recent.append(frame)
            # np.concatenate would drop the padding to the robot's 32-byte step.
            merged = np.empty(sum(len(f) for f in self._recent), dtype=DESKEWED_DTYPE)
            i = 0
            for f in self._recent:
                merged[i:i + len(f)] = f
                i += len(f)
            self.deskewed_pub.publish(make_cloud(merged, 'odom', stamp, DESKEWED_FIELDS))

    def _publish_state(self) -> None:
        msg = LidarState()
        now = time.time()
        msg.stamp = now
        msg.serial_recv_stamp = now
        msg.software_version = '1.0.0.38'
        msg.sys_rotation_speed = 12922.68
        msg.com_rotation_speed = 259.81
        if len(self._frame_times) > 1:
            span = self._frame_times[-1] - self._frame_times[0]
            msg.cloud_frequency = (len(self._frame_times) - 1) / span if span > 0 else 0.0
        msg.cloud_size = self._last_size * 15
        msg.cloud_scan_num = 210
        msg.imu_frequency = 250.0
        self.state_pub.publish(msg)


def main():
    run(Lidar)


if __name__ == '__main__':
    main()
