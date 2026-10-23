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


class FollowLeader(Node):
    def __init__(self):
        super().__init__('follow_leader')
        self.has_leader = False
        self.leader_id = None
        self.last_seen = None
        self.leader = None
        self.last_path = None
        self.moving = False
        self.speed = [0.0,0.0,0.0]
        self.last_time = 0

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
                speed_triplet = (0.0, 0.0, 0.0)
                if i.moving:
                    speed_triplet = (i.velocity.x, i.velocity.y, i.velocity_std)
                
                obstacles.append((i.nearest_range, i.nearest_bearing, speed_triplet))

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
            self.last_path = None
            return Command()

        if self.leader is None:
            return Command()

        dt = self.dt          # your control period
        margin = 1.0  # safety margin for obstacles
        N = 7  # prediction horizon

        #A = np.asarray(self.A())
        B = np.asarray(self.B())

        Q = np.diag([8.0, 0.5])
        R = np.diag([7.5, 7.5, 0.5])

        u = cp.Variable((3, N))
        x = cp.Variable((3, N + 1))
        constraints = []

        # current state and reference
        constraints += [x[:, 0] == np.array([0.0, 0.0, 0.0])]  # initial state is at the origin
        cost = 0

        # accumulate the displacement over the prediction horizon
        vp = np.zeros(2)
        p = 0
        t = 0


        # convert obs to Cartesian coordinates
        obstacles = np.asarray([[r * math.cos(b), r * math.sin(b), s[0], s[1]] for r, b, s in self.obstacles])
        obstacle_std = np.asarray([s[2] for r, b, s in self.obstacles])  # extract the std deviations
        if len(obstacles) > 0:
            obstacle_speed = np.linalg.norm(obstacles[:, 2:], axis=1)  # compute the speed of each obstacle
            obstacles[obstacle_speed < obstacle_std * 2, 2: ] = 0 # set the velocity to zero for obstacles with low speed relative to their uncertainty
        leader = np.array([self.leader.range * math.cos(self.leader.bearing), self.leader.range * math.sin(self.leader.bearing), self.leader.velocity.x, self.leader.velocity.y])
        leader_std = np.array([self.leader.velocity_std])
        leader_speed = np.linalg.norm(leader[2:])  # compute the speed of the leader
        if leader_speed < leader_std * 2:
            leader[2:] = 0  # set the velocity to zero for the leader if its speed is low relative to its uncertainty


        for k in range(N):
            constraints += [x[:, k + 1] == x[:, k] + dt * (B @ u[:, k])]

            for (i, max_vel) in enumerate([1.0, 1.0, 1.0]):
                constraints += [u[i, k] <=  max_vel]
                constraints += [u[i, k] >= -max_vel]

            error = x[:, k + 1] - np.array([leader[0], leader[1], 0.0])  # desired state is the leader's position with zero orientation
            cost += cp.quad_form(x[:, k + 1], Q) + cp.quad_form(u[:, k], R)
            if len(obstacles) > 0:
                cos_t = math.cos(t)
                sin_t = math.sin(t)
                Rk = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                # accumulate the displacement over the prediction horizon
                if self.last_path is not None:
                    idx = min(k+1, self.last_path.shape[1] - 1)
                    t += self.last_path[2, idx] * dt  # accumulate the change in orientation over the prediction horizon
                    cos_t = math.cos(t)
                    sin_t = math.sin(t)
                    vp += np.array([self.last_path[0, idx] * dt * cos_t - self.last_path[1, idx] * dt * sin_t,
                                    self.last_path[0, idx] * dt * sin_t + self.last_path[1, idx] * dt * cos_t])  # accumulate the displacement over the prediction horizon
                p = p + dt * (Rk @ u[:2, k])
                shifted_obstacles = obstacles[:,:2] + obstacles[:,2:]*(k+1)*dt - vp # shift the obstacles by the estimated displacement
                moved_obstacles = obstacles[:,:2] + obstacles[:,2:]*(k+1)*dt  # shift the obstacles by their estimated displacement

                distances = np.linalg.norm(shifted_obstacles, axis=1)  # compute the distances to the shifted obstacles
                directions = shifted_obstacles / distances[:, np.newaxis]  # compute the direction vectors to the shifted obstacles
                for i in range(len(distances)):
                    constraints += [directions[i] @ (moved_obstacles[i,:2] - p) >= margin]  # ensure the distance to each obstacle is greater than the safety margin

        objective = cp.Minimize(cost)
        problem = cp.Problem(objective, constraints)
        problem.solve()

        cmd = Command()


        if problem.status in ("optimal", "optimal_inaccurate"):
            self.last_path = u.value  # store the last computed path for future use
            cmd.vx = u.value[0, 0]
            cmd.vy = u.value[1, 0]
            cmd.wz = u.value[2, 0]
            self.speed = [cmd.vx, cmd.vy, cmd.wz]
        else:
            self.last_path = None  # reset the last path if the problem is infeasible
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
