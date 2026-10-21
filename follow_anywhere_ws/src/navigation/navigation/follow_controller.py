import json
import math
import time

import cvxpy as cp
import numpy as np
import rclpy
from rclpy.node import Node

from perception_msgs.msg import DetectionArray
from unitree_api.msg import Request

API_STOP_MOVE = 1003
API_MOVE = 1008

CLASS_LEADER = 3

class Command:
    def __init__(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0):
        self.vx = vx
        self.vy = vy
        self.wz = wz

    def is_zero(self) -> bool:
        return self.vx == 0.0 and self.vy == 0.0 and self.wz == 0.0


def clamp(v: float, limit: float) -> float:
    return max(-limit, min(limit, v))


class FollowLeader(Node):
    def __init__(self):
        super().__init__('follow_leader')
        self.has_leader = False
        self.leader_id = None
        self.last_seen = None
        self.leader = None
        self.moving = False
        self.speed = [0.0,0.0,0.0]

        self.obstacles = []

        self.pub = self.create_publisher(Request, '/api/sport/request', 10)
        self.create_subscription(DetectionArray, '/perception/detections', self.on_detections, 10)
        rate = self.declare_parameter('rate', 10.0).value
        self.dt = 1.0 / rate
        self.create_timer(self.dt, self.tick)

    def on_detections(self, msg: DetectionArray) -> None:
        now = time.monotonic()
        
        obstacles = []
        for i in msg.detections:
            if i.class_id == CLASS_LEADER:
                self.leader = i
                self.last_seen = now
                self.has_leader = True
                self.leader_id = i.id
                
            elif self.has_leader and i.id == self.leader_id:
                self.leader = i
                self.last_seen = now
                
            else:
                obstacles.append((i.nearest_range, i.nearest_bearing))

        self.obstacles = obstacles
                

    def tick(self) -> None:
        cmd = self.compute()

        if cmd is not None and not cmd.is_zero():
            self.send(API_MOVE, json.dumps({'x': cmd.vx, 'y': cmd.vy, 'z': cmd.wz}))
            self.moving = True
        elif self.moving:
            self.stop()

    def send(self, api_id: int, parameter: str = '') -> None:
        msg = Request()
        msg.header.identity.id = time.monotonic_ns() // 1000
        msg.header.identity.api_id = api_id
        msg.parameter = parameter
        self.pub.publish(msg)

    def stop(self) -> None:
        self.send(API_STOP_MOVE)
        self.moving = False

    def compute(self) -> Command:
        now = time.monotonic()
        if not self.has_leader:
            return Command()

        if self.last_seen is None or now - self.last_seen > 1.0:
            self.has_leader = False
            return Command()

        if self.leader is None:
            return Command()

        dt = self.dt          # your control period
        margin = 1.5  # safety margin for obstacles

        A = np.asarray(self.A())
        B = np.asarray(self.B())

        Q = np.diag([10.0, 1.0])
        R = np.diag([5.0, 0.5, 0.5])

        u = cp.Variable(3)

        # current state and reference
        x0 = np.array([self.leader.range - 2.0, self.leader.bearing])

        # prediction
        x_next = x0 + dt * (B @ u)

        # obstacles: position after one step must stay on the free side
        C = []
        if self.obstacles:
            obs = np.asarray(self.obstacles)            # (N, 2): [d, theta]
            a = np.column_stack([np.cos(obs[:, 1]), np.sin(obs[:, 1])])
            b = obs[:, 0] - margin
            C = [a @ (u[:2] * dt) <= b]

        objective = cp.Minimize(cp.quad_form(x_next, Q) + cp.quad_form(u, R))
        problem = cp.Problem(objective, C)
        problem.solve()

        cmd = Command()

        if problem.status in ("optimal", "optimal_inaccurate"):
            vx, vy, w = u.value
            cmd.vx = clamp(vx, 1.5)
            cmd.vy = clamp(vy, 1.5)
            cmd.wz = clamp(w, 2.0)
        return cmd




    def A(self) -> [[float, float], [float, float]]:
        if self.leader is None:
            return [0.0, 0.0, 0.0, 0.0]
        t = self.leader.bearing
        d = self.leader.range
        vx = self.speed[0]
        vy = self.speed[1]
        vsin = vx*math.sin(t) - vy*math.cos(t)
        vcos = vx*math.cos(t) + vy*math.sin(t)

        return [[0.0,vsin],[vsin/d,vcos/d]]

    def B(self) -> [[float, float, float], [float, float, float]]:
        if self.leader is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        t = self.leader.bearing
        d = self.leader.range

        return [[-math.cos(t), -math.sin(t), 0.0], [math.sin(t)/d, -math.cos(t)/d, -1.0]]


        




def main(args=None):
    rclpy.init(args=args)
    node = FollowLeader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.stop()
            rclpy.spin_once(node, timeout_sec=0.1)
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
