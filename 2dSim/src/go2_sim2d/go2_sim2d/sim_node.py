"""The 2D sim: a kinematic Go2 in a go2_scenarios world, with the robot's ROS interface.

No physics: the robot moves exactly as commanded through the sport API (after
the clamp and an optional first-order lag), and stops in front of static
objects and movers instead of pushing them. People walk through it, like the
Gazebo actors. Runs and stamps on wall-clock time, like the robot.

Subscribes:
  /api/sport/request   (unitree_api/Request)     same api_ids as the Gazebo sim
Publishes, robot interface (what robot code may use):
  /api/sport/response  (unitree_api/Response)    one per request
  /utlidar/robot_odom  (nav_msgs/Odometry)       130 Hz, odom -> base_link, twist in base_link
  /utlidar/robot_pose  (geometry_msgs/PoseStamped) 17 Hz, odom
  /sportmodestate, /lf/sportmodestate (unitree_go/SportModeState) 260 Hz, 18 Hz
Publishes, ground truth (sim only, like Gazebo's /sim/*; robot code must not use it):
  /sim/odometry        (nav_msgs/Odometry)       the true pose, odom
  /sim/leader/pose     (geometry_msgs/PoseStamped)
  /sim/people          (geometry_msgs/PoseArray) every person, scene order
  /sim/movers          (geometry_msgs/PoseArray) every mover, scene order
  /sim/collisions      (std_msgs/UInt32)         contacts with objects or people so far

The robot starts at the origin facing +x, so odom equals the world frame
(odometry drift aside, see odom_drift).
"""

import json
import math
import time

import numpy as np
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import UInt32
from unitree_api.msg import Request, Response
from unitree_go.msg import SportModeState

from go2_sim2d import geometry
from go2_sim2d.common import ROBOT_QOS, run
from go2_sim2d.world import World

API_DAMP = 1001
API_BALANCE_STAND = 1002
API_STOP_MOVE = 1003
API_STAND_UP = 1004
API_STAND_DOWN = 1005
API_RECOVERY_STAND = 1006
API_MOVE = 1008

DOWN_APIS = {API_STAND_DOWN, API_DAMP}
UP_APIS = {API_STAND_UP, API_RECOVERY_STAND, API_BALANCE_STAND}

# SportModeState.mode values the robot reports.
MODE_BALANCE_STAND = 1
MODE_LOCOMOTION = 3
MODE_LIE_DOWN = 5

STAND_HEIGHT = 0.32
CONTACT_MARGIN = 0.02
DOWN_HEIGHT = 0.14


def pose_msg(x, y, z, yaw):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = float(x), float(y), float(z)
    pose.orientation.z = math.sin(yaw / 2)
    pose.orientation.w = math.cos(yaw / 2)
    return pose


class Sim2d(Node):

    def __init__(self):
        super().__init__('sim2d')
        self.declare_parameter('world', 'open_field')
        self.declare_parameter('rate', 100.0)
        self.declare_parameter('odom_rate', 130.0)
        self.declare_parameter('pose_rate', 17.0)
        self.declare_parameter('sportmodestate_rate', 260.0)
        self.declare_parameter('lf_rate', 18.0)
        self.declare_parameter('ground_truth_rate', 50.0)
        # Sport API, as in Gazebo's go2_robot_emulator/sport_api.py; the robot
        # reaches a new speed in ~0.1 s.
        self.declare_parameter('cmd_timeout', 0.5)
        self.declare_parameter('response_time', 0.1)
        self.declare_parameter('max_vx', 0.6)
        self.declare_parameter('max_vy', 0.4)
        self.declare_parameter('max_wz', 1.6)
        self.declare_parameter('robot_radius', 0.35)
        # Odometry error: std of the relative error of each velocity component
        # integrated into /utlidar/robot_odom (0 = odometry equals ground truth).
        self.declare_parameter('odom_drift', 0.0)

        def param(name):
            return self.get_parameter(name).value

        self.world = World(param('world'))
        self._dt = 1.0 / param('rate')
        self._cmd_timeout = param('cmd_timeout')
        self._tau = param('response_time')
        self._limits = [param('max_vx'), param('max_vy'), param('max_wz')]
        self._robot_radius = param('robot_radius')
        self._drift = param('odom_drift')
        self._rng = np.random.default_rng()

        self._start = time.monotonic()
        self._last_tick = self._start
        self._pose = [0.0, 0.0, 0.0]     # true x, y, yaw
        self._odom = [0.0, 0.0, 0.0]     # what odometry reports
        self._target = [0.0, 0.0, 0.0]   # commanded body velocity
        self._velocity = [0.0, 0.0, 0.0]  # actual body velocity
        self._last_move: float | None = None
        self._down = False
        self._collisions = 0
        self._in_contact = False
        self._people = self.world.people_poses(0.0)
        self._movers = self.world.mover_poses(0.0)

        self.create_subscription(Request, '/api/sport/request', self._request_callback, 10)
        self.response_pub = self.create_publisher(Response, '/api/sport/response', 10)
        self.odom_pub = self.create_publisher(Odometry, '/utlidar/robot_odom', ROBOT_QOS)
        self.pose_pub = self.create_publisher(PoseStamped, '/utlidar/robot_pose', ROBOT_QOS)
        self.sportstate_pub = self.create_publisher(SportModeState, '/sportmodestate', ROBOT_QOS)
        self.lf_sportstate_pub = self.create_publisher(
            SportModeState, '/lf/sportmodestate', ROBOT_QOS)
        self.truth_pub = self.create_publisher(Odometry, '/sim/odometry', 10)
        self.leader_pub = self.create_publisher(PoseStamped, '/sim/leader/pose', 10)
        self.people_pub = self.create_publisher(PoseArray, '/sim/people', 10)
        self.movers_pub = self.create_publisher(PoseArray, '/sim/movers', 10)
        self.collisions_pub = self.create_publisher(UInt32, '/sim/collisions', 10)

        def every(rate_param, callback):
            self.create_timer(1.0 / param(rate_param), callback)

        self.create_timer(self._dt, self._tick)
        every('odom_rate', self._publish_odom)
        every('pose_rate', self._publish_pose)
        every('sportmodestate_rate', lambda: self.sportstate_pub.publish(self._sportstate()))
        every('lf_rate', lambda: self.lf_sportstate_pub.publish(self._sportstate()))
        every('ground_truth_rate', self._publish_ground_truth)

        self.get_logger().info(
            f"World '{self.world.name}': {len(self.world.static)} objects, "
            f"{len(self.world.movers)} movers, {len(self.world.people)} people "
            f"({'with' if self.world.leader_index is not None else 'without'} leader)")

    # ---- Sport API ----

    def _request_callback(self, msg: Request) -> None:
        api_id = msg.header.identity.api_id
        if api_id == API_MOVE:
            self._move(msg.parameter)
        elif api_id == API_STOP_MOVE:
            self._last_move = None
            self._target = [0.0, 0.0, 0.0]
        elif api_id in DOWN_APIS:
            self._down = True
            self._last_move = None
            self._target = [0.0, 0.0, 0.0]
        elif api_id in UP_APIS:
            self._down = False

        if not msg.header.policy.noreply:
            response = Response()
            response.header.identity = msg.header.identity
            response.header.status.code = 0
            self.response_pub.publish(response)

    def _move(self, parameter: str) -> None:
        try:
            p = json.loads(parameter)
            cmd = [float(p.get('x', 0.0)), float(p.get('y', 0.0)), float(p.get('z', 0.0))]
        except (ValueError, TypeError, AttributeError):
            self.get_logger().warning(f'Bad Move parameter: {parameter!r}', throttle_duration_sec=2.0)
            return
        if self._down:
            self.get_logger().info('Move ignored while lying down', throttle_duration_sec=2.0)
            return
        self._target = [max(-lim, min(lim, v)) for v, lim in zip(cmd, self._limits)]
        self._last_move = time.monotonic()

    # ---- Simulation ----

    def _tick(self) -> None:
        now = time.monotonic()
        dt, self._last_tick = now - self._last_tick, now
        t = now - self._start
        self._people = self.world.people_poses(t)
        self._movers = self.world.mover_poses(t)

        if self._last_move is not None and now - self._last_move > self._cmd_timeout:
            self._last_move = None
            self._target = [0.0, 0.0, 0.0]
        alpha = 1.0 - math.exp(-dt / self._tau) if self._tau > 0 else 1.0
        self._velocity = [v + alpha * (g - v) for v, g in zip(self._velocity, self._target)]
        if self._last_move is None and all(abs(v) < 1e-3 for v in self._velocity):
            self._velocity = [0.0, 0.0, 0.0]

        vx, vy, wz = self._velocity
        x, y, yaw = self._pose
        c, s = math.cos(yaw), math.sin(yaw)
        moved = [x + (c * vx - s * vy) * dt, y + (s * vx + c * vy) * dt,
                 geometry.wrap(yaw + wz * dt)]

        blocking = self.world.static_shapes + self.world.mover_shapes(self._movers)
        r = self._robot_radius
        # Blocked if the step overlaps an object and doesn't move out of it,
        # so a robot that got touched by a mover can still drive away.
        before = min((geometry.distance(b, x, y) for b in blocking), default=math.inf)
        after = min((geometry.distance(b, moved[0], moved[1]) for b in blocking), default=math.inf)
        blocked = after < r and after < before
        if blocked:
            moved[:2] = x, y
            self._velocity[:2] = [0.0, 0.0]
        self._pose = moved

        # A contact lasts until the robot is CONTACT_MARGIN clear again, so
        # creeping along an object counts as one collision.
        clear = r + CONTACT_MARGIN
        near = min((geometry.distance(b, moved[0], moved[1]) for b in blocking),
                   default=math.inf) < clear
        touching_person = any(geometry.distance(p, moved[0], moved[1]) < clear
                              for p in self.world.people_shapes(self._people))
        contact = blocked or near or touching_person
        if contact and not self._in_contact:
            self._collisions += 1
            self.get_logger().warning(f'Collision #{self._collisions} at '
                                   f'({moved[0]:.2f}, {moved[1]:.2f})')
        self._in_contact = contact

        # Odometry integrates the actual motion, with a relative error per component.
        ovx, ovy, owz = self._velocity if not blocked else (0.0, 0.0, wz)
        if self._drift > 0:
            ovx, ovy, owz = (v * (1 + self._rng.normal(0.0, self._drift)) for v in (ovx, ovy, owz))
        ox, oy, oyaw = self._odom
        c, s = math.cos(oyaw), math.sin(oyaw)
        self._odom = [ox + (c * ovx - s * ovy) * dt, oy + (s * ovx + c * ovy) * dt,
                      geometry.wrap(oyaw + owz * dt)]

    # ---- Publishing ----

    def _height(self):
        return DOWN_HEIGHT if self._down else STAND_HEIGHT

    def _odometry(self, pose, frame='odom'):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame
        msg.child_frame_id = 'base_link'
        msg.pose.pose = pose_msg(*pose[:2], self._height(), pose[2])
        msg.twist.twist.linear.x, msg.twist.twist.linear.y = self._velocity[:2]
        msg.twist.twist.angular.z = self._velocity[2]
        return msg

    def _publish_odom(self) -> None:
        self.odom_pub.publish(self._odometry(self._odom))

    def _publish_pose(self) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.pose = pose_msg(*self._odom[:2], self._height(), self._odom[2])
        self.pose_pub.publish(msg)

    def _sportstate(self) -> SportModeState:
        msg = SportModeState()
        now = time.time_ns()
        msg.stamp.sec = now // 1_000_000_000
        msg.stamp.nanosec = now % 1_000_000_000
        if self._down:
            msg.mode = MODE_LIE_DOWN
        elif any(self._velocity):
            msg.mode = MODE_LOCOMOTION
        else:
            msg.mode = MODE_BALANCE_STAND
        x, y, yaw = self._odom
        msg.position[:] = (x, y, self._height())
        msg.body_height = self._height()
        msg.velocity[:] = (self._velocity[0], self._velocity[1], 0.0)
        msg.yaw_speed = float(self._velocity[2])
        msg.imu_state.rpy[:] = (0.0, 0.0, yaw)
        msg.imu_state.quaternion[:] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        msg.imu_state.gyroscope[:] = (0.0, 0.0, self._velocity[2])
        msg.imu_state.accelerometer[:] = (0.0, 0.0, 9.81)
        return msg

    def _publish_ground_truth(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self.truth_pub.publish(self._odometry(self._pose))

        people = PoseArray()
        people.header.stamp = stamp
        people.header.frame_id = 'odom'
        people.poses = [pose_msg(x, y, 0.0, yaw) for x, y, yaw in self._people]
        self.people_pub.publish(people)
        if self.world.leader_index is not None:
            leader = PoseStamped()
            leader.header = people.header
            leader.pose = people.poses[self.world.leader_index]
            self.leader_pub.publish(leader)

        movers = PoseArray()
        movers.header = people.header
        movers.poses = [pose_msg(x, y, 0.0, yaw) for x, y, yaw in self._movers]
        self.movers_pub.publish(movers)
        self.collisions_pub.publish(UInt32(data=self._collisions))


def main():
    run(Sim2d)


if __name__ == '__main__':
    main()
