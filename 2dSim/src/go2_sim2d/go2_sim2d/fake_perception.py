"""Stands in for the perception stack: leader and obstacles made from the sim's ground truth.

Works only from the ground-truth topics and the scene file (for the static
objects and the movers' sizes), not from sim internals.

Subscribes (ground truth):
  /sim/odometry  (nav_msgs/Odometry)        the robot's true pose
  /sim/people    (geometry_msgs/PoseArray)  scene order; the leader is the one named "leader"
  /sim/movers    (geometry_msgs/PoseArray)  scene order
Publishes (what the real perception stack will publish):
  /perception/leader      (follow_anywhere_interfaces/PerceptionLeader)         base_link, `rate` Hz
  /perception/obstacles   (follow_anywhere_interfaces/PerceptionObstacleArray)  base_link, `rate` Hz
  /perception/scan        (sensor_msgs/LaserScan)  base_link, 360 deg, `scan_rate` Hz

Noise, sensor fields and the other parameters: perception.Params and
config/perception.yaml.
"""

import dataclasses
import math

from follow_anywhere_interfaces.msg import (PerceptionLeader, PerceptionObstacle,
                                            PerceptionObstacleArray)
from geometry_msgs.msg import Point, PoseArray, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from go2_sim2d.common import run, stamp_seconds, yaw_of
from go2_sim2d.perception import (Entity, FakePerception, LeaderTracker, Params,
                                  VelocityTracker)
from go2_sim2d.world import World


class FakePerceptionNode(Node):

    def __init__(self):
        super().__init__('fake_perception')
        self.declare_parameter('world', 'open_field')
        self.declare_parameter('rate', 15.0)
        self.declare_parameter('scan_rate', 15.0)
        self.declare_parameter('seed', -1)
        params = Params()
        for f in dataclasses.fields(Params):
            value = self.declare_parameter(f.name, getattr(params, f.name)).value
            setattr(params, f.name, list(value) if isinstance(value, (list, tuple)) else value)
        seed = self.get_parameter('seed').value

        self.world = World(self.get_parameter('world').value)
        self.perception = FakePerception(params, seed if seed >= 0 else None)
        self.tracker = VelocityTracker()
        self.leader_tracker = LeaderTracker()
        self.leader_id = (self.world.person_id(self.world.leader_index)
                          if self.world.leader_index is not None else None)
        self._robot = None
        self._people = []
        self._movers = []

        self.create_subscription(Odometry, '/sim/odometry', self._odom_cb, 10)
        self.create_subscription(PoseArray, '/sim/people', self._people_cb, 10)
        self.create_subscription(PoseArray, '/sim/movers', self._movers_cb, 10)
        self.leader_pub = self.create_publisher(PerceptionLeader, '/perception/leader', 10)
        self.obstacles_pub = self.create_publisher(PerceptionObstacleArray, '/perception/obstacles', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/perception/scan', 10)
        self.create_timer(1.0 / self.get_parameter('rate').value, self._publish_perception)
        self.create_timer(1.0 / self.get_parameter('scan_rate').value, self._publish_scan)

        self.get_logger().info(
            f'fake_perception started: lidar {params.lidar_fov_deg:.0f} deg '
            f'{params.lidar_range_max:.0f} m, camera {params.camera_fov_deg:.0f} deg '
            f'{params.camera_range_max:.0f} m, noise x{params.noise_scale}')

    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self._robot = (p.x, p.y, yaw_of(msg.pose.pose))

    def _track(self, prefix, msg: PoseArray):
        t = stamp_seconds(msg.header)
        poses = []
        for i, pose in enumerate(msg.poses):
            x, y = pose.position.x, pose.position.y
            poses.append((x, y, yaw_of(pose), self.tracker.update((prefix, i), x, y, t)))
        return poses

    def _people_cb(self, msg: PoseArray) -> None:
        self._people = self._track('person', msg)

    def _movers_cb(self, msg: PoseArray) -> None:
        self._movers = self._track('mover', msg)

    def _entities(self):
        w = self.world
        entities = [Entity(w.static_id(i), 'static', shape)
                    for i, shape in enumerate(w.static_shapes)]
        if len(self._movers) == len(w.movers):
            shapes = w.mover_shapes([(x, y, yaw) for x, y, yaw, _ in self._movers])
            entities += [Entity(w.mover_id(i), 'mover', shape, velocity=m[3])
                         for i, (shape, m) in enumerate(zip(shapes, self._movers))]
        shapes = w.people_shapes(self._people)
        entities += [Entity(w.person_id(i), 'person', shape, leader=i == w.leader_index,
                            velocity=p[3])
                     for i, (shape, p) in enumerate(zip(shapes, self._people))]
        return entities

    def _publish_perception(self) -> None:
        if self._robot is None:
            return
        now = self.get_clock().now()
        t = now.nanoseconds * 1e-9
        detections = self.perception.detect(self._robot, self._entities(), t)
        leader, obstacles = self.leader_tracker.update(detections, self.leader_id, self._robot, t)

        stamp = now.to_msg()
        msg = PerceptionLeader()
        msg.header.stamp = stamp
        msg.header.frame_id = 'base_link'
        msg.status = leader.status
        msg.closest_point = _point(leader.closest_point)
        msg.velocity = _vector(leader.velocity)
        msg.position_uncertainty = float(leader.position_uncertainty)
        msg.velocity_uncertainty = float(leader.velocity_uncertainty)
        msg.lost_duration = float(leader.lost_duration)
        msg.last_known_point = _point(leader.last_known_point)
        self.leader_pub.publish(msg)

        array = PerceptionObstacleArray()
        array.header.stamp = stamp
        array.header.frame_id = 'base_link'
        for o in obstacles:
            out = PerceptionObstacle()
            out.closest_point = _point(o.closest_point)
            out.line_of_sight_point = _point(o.line_of_sight_point)
            out.velocity = _vector(o.velocity)
            out.position_uncertainty = float(o.position_uncertainty)
            out.velocity_uncertainty = float(o.velocity_uncertainty)
            array.obstacles.append(out)
        self.obstacles_pub.publish(array)

    def _publish_scan(self) -> None:
        if self._robot is None:
            return
        p = self.perception.p
        shapes = [e.shape for e in self._entities()]
        bearings, ranges = self.perception.scan(self._robot, shapes)
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.angle_min = float(bearings[0])
        msg.angle_increment = 2 * math.pi / p.scan_rays
        msg.angle_max = float(bearings[-1])
        msg.scan_time = 1.0 / self.get_parameter('scan_rate').value
        msg.range_min = float(p.lidar_range_min)
        msg.range_max = float(p.lidar_range_max)
        msg.ranges = ranges.astype(float).tolist()
        self.scan_pub.publish(msg)


def _point(xy):
    return Point(x=float(xy[0]), y=float(xy[1]), z=0.0)


def _vector(xy):
    return Vector3(x=float(xy[0]), y=float(xy[1]), z=0.0)


def main():
    run(FakePerceptionNode)


if __name__ == '__main__':
    main()
