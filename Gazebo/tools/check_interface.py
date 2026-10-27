#!/usr/bin/env python3
"""Check that the running sim publishes the same ROS interface as the real Go2.

Reads a bag recorded on the robot, then listens to the live sim and prints one
row per topic: present, same type, rate close to the robot's, same frame_id,
same PointCloud2 layout. Exits non-zero if any emulated topic fails.

    python3 tools/check_interface.py /repo/go2_all_topics_6_10_11_28 [--duration 10]

Run it inside the sim container (it needs unitree_go / unitree_api), while
`ros2 launch go2_sim go2_sim.launch.py` is running.
"""

import argparse
import collections
import sys
import time

import rclpy
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rosgraph_msgs.msg import Clock
from rosidl_runtime_py.utilities import get_message

from bag_utils import open_bag, read_messages

# On the robot but deliberately not emulated (app/firmware services and
# derived maps our code does not use, and the robot's internal /lowcmd).
NOT_EMULATED = {
    '/lowcmd', '/audiosender', '/audiohub/player/state', '/rosout',
    '/utlidar/range_info', '/utlidar/height_map_array', '/utlidar/voxel_map_compressed',
    '/api/obstacles_avoid/request', '/api/obstacles_avoid/response',
    '/api/uwbswitch/request', '/api/uwbswitch/response',
    '/api/vui/request', '/api/vui/response',
    '/lf/battery_alarm', '/rtc/state', '/gnss', '/multiplestate', '/gpt_state',
    # Request side is the user's code, not the robot.
    '/api/sport/request',
}
# Answered per request, so it only has a rate while something sends requests.
EVENT_TOPICS = {'/api/sport/response'}
# Published as Gazebo sensors produce data, so they run in sim time: their
# wall-clock rate scales with the sim's real-time factor on this machine.
# Everything else is published on wall-clock timers, like the robot.
SIM_TIME_TOPICS = {'/utlidar/cloud', '/utlidar/cloud_base', '/utlidar/cloud_deskewed',
                   '/utlidar/imu', '/frontvideostream'}
RATE_TOLERANCE = 0.3


def describe(msg):
    """frame_id and PointCloud2 layout of a message, where it has them."""
    info = {}
    header = getattr(msg, 'header', None)
    if header is not None and hasattr(header, 'frame_id'):
        info['frame'] = header.frame_id
    elif isinstance(getattr(msg, 'frame_id', None), str):
        info['frame'] = msg.frame_id
    if hasattr(msg, 'fields') and hasattr(msg, 'point_step'):
        info['layout'] = (msg.point_step,
                          tuple((f.name, f.offset, f.datatype) for f in msg.fields))
    if hasattr(msg, 'child_frame_id'):
        info['child'] = msg.child_frame_id
    return info


def read_bag(path):
    reader, types = open_bag(path)
    meta = reader.get_metadata()
    duration = meta.duration.nanoseconds / 1e9
    counts = {t.topic_metadata.name: t.message_count for t in meta.topics_with_message_count}
    topics = [t for t, c in counts.items() if c > 0]
    first = {}
    for topic, _, msg in read_messages(path, topics):
        if topic not in first:
            first[topic] = describe(msg)
        if len(first) == len(topics):
            break
    return {t: {'type': types[t], 'rate': counts[t] / duration, **first.get(t, {})}
            for t in topics}


def listen(expected, duration):
    rclpy.init()
    node = rclpy.create_node('check_interface')
    # Best effort matches both reliable and best-effort publishers.
    qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                     history=QoSHistoryPolicy.KEEP_LAST, depth=50)
    stamps = collections.defaultdict(list)
    first = {}

    def callback(topic):
        def cb(msg):
            stamps[topic].append(time.monotonic())
            if topic not in first:
                first[topic] = describe(msg)
        return cb

    for topic, spec in expected.items():
        node.create_subscription(get_message(spec['type']), topic, callback(topic), qos)
    clock = []
    node.create_subscription(
        Clock, '/clock',
        lambda m: clock.append((time.monotonic(), m.clock.sec + m.clock.nanosec / 1e9)), qos)

    end = time.monotonic() + duration
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)

    # Read the graph after listening, so discovery has had time to finish.
    live_types = {name: types for name, types in node.get_topic_names_and_types()}
    tf_publishers = node.count_publishers('/tf') + node.count_publishers('/tf_static')
    rtf = ((clock[-1][1] - clock[0][1]) / (clock[-1][0] - clock[0][0])
           if len(clock) > 1 else 1.0)
    node.destroy_node()
    rclpy.shutdown()

    live = {}
    for topic in expected:
        s = stamps[topic]
        rate = (len(s) - 1) / (s[-1] - s[0]) if len(s) > 1 and s[-1] > s[0] else 0.0
        live[topic] = {'types': live_types.get(topic, []), 'rate': rate, **first.get(topic, {})}
    return live, tf_publishers, rtf


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag', help='bag directory or .mcap file from the real robot')
    parser.add_argument('--duration', type=float, default=10.0, help='seconds to listen')
    args = parser.parse_args()

    robot = read_bag(args.bag)
    expected = {t: s for t, s in robot.items() if t not in NOT_EMULATED}
    print(f'Listening to {len(expected)} topics for {args.duration:.0f} s...')
    live, tf_publishers, rtf = listen(expected, args.duration)
    print(f'Sim real-time factor: {rtf:.2f}. Rates of Gazebo-sensor topics (*) are '
          'compared at that speed.')

    failed = 0
    print(f'\n{"topic":32} {"result":6} {"rate robot/sim":>16}  notes')
    for topic in sorted(expected):
        want, got = expected[topic], live[topic]
        notes = []
        if not got['types']:
            notes.append('not published')
        elif want['type'] not in got['types']:
            notes.append(f'type {got["types"]} != {want["type"]}')
        if got['types'] and topic not in EVENT_TOPICS:
            if got['rate'] == 0.0:
                notes.append('no messages')
            else:
                want_rate = want['rate'] * (rtf if topic in SIM_TIME_TOPICS else 1.0)
                if abs(got['rate'] / want_rate - 1.0) > RATE_TOLERANCE:
                    notes.append(f'rate off by {100 * (got["rate"] / want_rate - 1):+.0f}%')
        for key in ('frame', 'child', 'layout'):
            if key in want and key in got and want[key] != got[key]:
                notes.append(f'{key} {got[key]!r} != {want[key]!r}')
        ok = not notes
        failed += not ok
        rates = f'{want["rate"]:6.1f}/{got["rate"]:6.1f}'
        name = topic + (' *' if topic in SIM_TIME_TOPICS else '')
        print(f'{name:32} {"ok" if ok else "FAIL":6} {rates:>16}  {"; ".join(notes)}')

    skipped = sorted(t for t in robot if t in NOT_EMULATED)
    print(f'\nNot emulated (by design): {", ".join(skipped)}')
    if tf_publishers:
        failed += 1
        print(f'FAIL: {tf_publishers} publisher(s) on /tf or /tf_static; the robot has none')
    else:
        print('ok: nothing on /tf or /tf_static, like the robot')
    print(f'\n{failed} failure(s)')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
