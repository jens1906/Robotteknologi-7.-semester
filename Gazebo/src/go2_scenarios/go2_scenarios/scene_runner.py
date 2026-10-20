"""Moves the scene's movers and publishes where the people are (sim only).

Runs on sim time (/clock), since Gazebo animates the actors on sim time.
Publishes (sim-internal, robot code must not use them; for evaluation only):
  /sim/leader/pose  (geometry_msgs/PoseStamped)  the leader on the ground, frame odom
  /sim/people       (geometry_msgs/PoseArray)    every person, in scene order, frame odom
Both are stamped with wall-clock time like the emulator's topics, so they line
up with /utlidar/robot_odom in a bag. The Go2 spawns at the origin, so the
world frame equals odom.

Movers are kinematic: their pose is set through /world/go2_world/set_pose
(bridged by ros_gz_bridge) every cycle.
"""

import math

import rclpy
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose

from go2_scenarios import scene


def _pose(x, y, z, yaw):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = x, y, z
    pose.orientation.z = math.sin(yaw / 2)
    pose.orientation.w = math.cos(yaw / 2)
    return pose


class SceneRunner(Node):

    def __init__(self):
        super().__init__('scene_runner')
        self.declare_parameter('world', 'empty')
        self.declare_parameter('rate', 20.0)
        world = self.get_parameter('world').value
        loaded = scene.load(world)
        self._wall = Clock(clock_type=ClockType.SYSTEM_TIME)

        self._people = [(p['name'], scene.timeline(p), float(p.get('start_delay', 0.0)))
                        for p in loaded['people']]
        self._movers = [(m['name'], scene.timeline(m), float(m.get('start_delay', 0.0)),
                         scene.height(m) / 2 + 0.01, float(m.get('yaw', 0.0)))
                        for m in loaded['movers']]
        self._has_leader = any(name == 'leader' for name, _, _ in self._people)
        self._pending = {}

        self._leader_pub = self.create_publisher(PoseStamped, '/sim/leader/pose', 10)
        self._people_pub = self.create_publisher(PoseArray, '/sim/people', 10)
        self._set_pose = self.create_client(SetEntityPose, f'/world/{scene.WORLD_NAME}/set_pose')

        if not self._people and not self._movers:
            self.get_logger().info(f"World '{world}' has no people or movers, nothing to do")
            return
        self.get_logger().info(
            f"World '{world}': {len(self._people)} people "
            f"({'with' if self._has_leader else 'without'} leader), {len(self._movers)} movers")
        self.create_timer(1.0 / self.get_parameter('rate').value, self._tick)

    def _tick(self):
        t = self.get_clock().now().nanoseconds * 1e-9
        if t == 0.0:
            return  # no /clock yet
        stamp = self._wall.now().to_msg()

        people = PoseArray()
        people.header.frame_id = 'odom'
        people.header.stamp = stamp
        for name, keys, delay in self._people:
            x, y, yaw = scene.pose_at(keys, t - delay)
            pose = _pose(x, y, 0.0, yaw)
            people.poses.append(pose)
            if name == 'leader':
                leader = PoseStamped()
                leader.header = people.header
                leader.pose = pose
                self._leader_pub.publish(leader)
        if self._people:
            self._people_pub.publish(people)

        if self._movers and self._set_pose.service_is_ready():
            for name, keys, delay, z, yaw in self._movers:
                if self._pending.get(name) is not None and not self._pending[name].done():
                    continue  # previous request still in flight; don't queue up
                x, y, _ = scene.pose_at(keys, t - delay)
                request = SetEntityPose.Request()
                request.entity.name = name
                request.entity.type = Entity.MODEL
                request.pose = _pose(x, y, z, yaw)
                self._pending[name] = self._set_pose.call_async(request)


def main():
    rclpy.init()
    node = SceneRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
