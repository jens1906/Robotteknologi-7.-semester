#!/usr/bin/env python3
"""Replay the real robot's sport commands into the sim and compare the motion.

Sends every /api/sport/request from a robot bag to the running sim at its
original timing in sim time (/clock), so the result does not depend on how fast
this machine runs the sim. Records the sim's /sportmodestate, and compares body-frame
velocity (vx, vy, yaw rate) with what the real robot did in the bag:
RMSE per axis, mean time to reach 63% of a commanded step, and distance
travelled. Writes both time series to a CSV (open in PlotJuggler or similar).

    python3 tools/replay_commands.py /repo/go2_all_topics_6_10_11_28 [--csv /tmp/replay.csv]

Run it inside the sim container while the sim is running, with the robot
standing still.
"""

import argparse
import csv
import json
import math
import sys
import time

import numpy as np
import rclpy
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rosgraph_msgs.msg import Clock
from unitree_api.msg import Request
from unitree_go.msg import SportModeState

from bag_utils import read_messages

API_MOVE = 1008
LEAD_IN = 1.0     # s of recording before the first command
LEAD_OUT = 3.0    # s of recording after the last command
GRID_DT = 0.02


def body_velocity(state):
    """vx, vy and yaw rate from a SportModeState (velocity is body-frame)."""
    return state.velocity[0], state.velocity[1], state.yaw_speed


def command(request):
    if request.header.identity.api_id != API_MOVE:
        return 0.0, 0.0, 0.0
    p = json.loads(request.parameter)
    return p.get('x', 0.0), p.get('y', 0.0), p.get('z', 0.0)


def load_bag(path):
    requests, states = [], []
    for topic, stamp, msg in read_messages(path, ['/api/sport/request', '/sportmodestate']):
        t = stamp / 1e9
        if topic == '/api/sport/request':
            requests.append((t, msg))
        else:
            states.append((t, *body_velocity(msg), msg.position[0], msg.position[1]))
    if not requests:
        raise SystemExit('No /api/sport/request in the bag')
    t0 = requests[0][0] - LEAD_IN
    requests = [(t - t0, m) for t, m in requests]
    end = requests[-1][0] + LEAD_OUT
    states = np.array([(t - t0, *rest) for t, *rest in states if 0 <= t - t0 <= end])
    return requests, states, end


def replay(requests, end):
    """Publish the requests on the sim's clock; return sim /sportmodestate on that clock."""
    rclpy.init()
    node = rclpy.create_node('replay_commands')
    pub = node.create_publisher(Request, '/api/sport/request', 10)
    qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                     history=QoSHistoryPolicy.KEEP_LAST, depth=50)
    states = []
    sim_now = [None]       # latest /clock, seconds
    start = [None]         # /clock at replay start
    wall = []

    def on_clock(msg):
        sim_now[0] = msg.clock.sec + msg.clock.nanosec / 1e9
        if start[0] is not None:
            wall.append((time.monotonic(), sim_now[0]))

    def on_state(msg):
        if start[0] is not None and sim_now[0] is not None:
            states.append((sim_now[0] - start[0], *body_velocity(msg),
                           msg.position[0], msg.position[1]))

    node.create_subscription(SportModeState, '/sportmodestate', on_state, qos)
    node.create_subscription(Clock, '/clock', on_clock, qos)

    deadline = time.monotonic() + 5.0
    while ((pub.get_subscription_count() == 0 or sim_now[0] is None)
           and time.monotonic() < deadline):
        rclpy.spin_once(node, timeout_sec=0.1)
    if pub.get_subscription_count() == 0 or sim_now[0] is None:
        raise SystemExit('No /api/sport/request subscriber or no /clock: is the sim running?')

    print(f'Replaying {len(requests)} requests over {end:.0f} s of sim time...')
    start[0] = sim_now[0]
    pending = list(requests)
    while sim_now[0] - start[0] < end:
        now = sim_now[0] - start[0]
        while pending and pending[0][0] <= now:
            pub.publish(pending.pop(0)[1])
        rclpy.spin_once(node, timeout_sec=0.002)

    node.destroy_node()
    rclpy.shutdown()
    if len(wall) > 1:
        rtf = (wall[-1][1] - wall[0][1]) / (wall[-1][0] - wall[0][0])
        print(f'Sim real-time factor during replay: {rtf:.2f} (timing follows sim time)')
    if not states:
        raise SystemExit('No /sportmodestate received from the sim')
    return np.array(states)


def resample(states, grid):
    return np.stack([np.interp(grid, states[:, 0], states[:, i]) for i in range(1, 6)], axis=1)


def step_lags(grid, cmd, vel):
    """Time from each 0 -> nonzero command step until the velocity reaches 63% of it."""
    lags = []
    for i in range(1, len(grid)):
        if cmd[i - 1] == 0.0 and abs(cmd[i]) > 0.1:
            target = 0.63 * cmd[i]
            for j in range(i, min(i + int(1.5 / GRID_DT), len(grid))):
                if cmd[j] != cmd[i]:
                    break
                if (vel[j] - target) * np.sign(cmd[i]) >= 0:
                    lags.append(grid[j] - grid[i])
                    break
    return lags


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag', help='bag directory or .mcap file from the real robot')
    parser.add_argument('--csv', default='replay.csv', help='output CSV path')
    args = parser.parse_args()

    requests, real, end = load_bag(args.bag)
    sim = replay(requests, end)

    grid = np.arange(0.0, end, GRID_DT)
    real_g, sim_g = resample(real, grid), resample(sim, grid)
    cmds = np.zeros((len(grid), 3))
    for t, msg in requests:
        cmds[grid >= t] = command(msg)
    # The robot stops when Moves stop arriving; the bridge times out after 0.5 s.
    last_move = -1e9
    times = [t for t, m in requests]
    for i, t in enumerate(grid):
        k = np.searchsorted(times, t, side='right') - 1
        if k >= 0 and requests[k][1].header.identity.api_id == API_MOVE:
            last_move = times[k]
        if t - last_move > 0.5:
            cmds[i] = 0.0

    print(f'\n{"axis":8} {"RMSE real-sim":>14} {"lag real":>10} {"lag sim":>10}')
    for i, name in enumerate(['vx m/s', 'vy m/s', 'wz rad/s']):
        rmse = math.sqrt(np.mean((real_g[:, i] - sim_g[:, i]) ** 2))
        lag_real = step_lags(grid, cmds[:, i], real_g[:, i])
        lag_sim = step_lags(grid, cmds[:, i], sim_g[:, i])

        def fmt(lags):
            return f'{np.mean(lags):.2f} s' if lags else '-'
        print(f'{name:8} {rmse:14.3f} {fmt(lag_real):>10} {fmt(lag_sim):>10}')

    def distance(xy):
        return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    print(f'\ndistance travelled: real {distance(real_g[:, 3:5]):.2f} m, '
          f'sim {distance(sim_g[:, 3:5]):.2f} m')

    with open(args.csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'cmd_x', 'cmd_y', 'cmd_z', 'real_vx', 'real_vy', 'real_wz',
                    'sim_vx', 'sim_vy', 'sim_wz'])
        for row in zip(grid, cmds, real_g, sim_g):
            w.writerow([f'{row[0]:.3f}', *(f'{v:.4f}' for v in row[1]),
                        *(f'{v:.4f}' for v in row[2][:3]), *(f'{v:.4f}' for v in row[3][:3])])
    print(f'wrote {args.csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
