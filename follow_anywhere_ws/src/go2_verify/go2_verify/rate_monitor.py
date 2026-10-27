# SPDX-License-Identifier: BSD-3-Clause

"""
Counts high-rate topics (/lowstate 450 Hz, /sportmodestate 260 Hz, ...) in a child process.

A Python callback per message at these rates holds the GIL often enough to starve
the GUI, so the counting runs in its own interpreter. Messages are received raw
(no deserialization). Results go to a shared array: [count, last arrival] per topic,
arrival in time.monotonic() (system-wide clock on Linux).
"""

import importlib
import multiprocessing as mp
import time


def _run(topics, shared, stop):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from rclpy.signals import SignalHandlerOptions

    qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                     history=QoSHistoryPolicy.KEEP_LAST, depth=50)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node('go2_verify_rates')
    for i, (topic, type_name) in enumerate(topics):
        module, name = type_name.rsplit('/', 1)
        msg_type = getattr(importlib.import_module(module.replace('/', '.')), name)

        def cb(_, i=i):
            shared[2 * i] += 1
            shared[2 * i + 1] = time.monotonic()

        node.create_subscription(msg_type, topic, cb, qos, raw=True)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    while not stop.is_set():
        executor.spin_once(timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()


class RateMonitor:
    """topics: list of (topic, 'pkg/msg/Type')."""

    def __init__(self, topics):
        ctx = mp.get_context('spawn')
        self.topics = [t for t, _ in topics]
        # lock=False: one writer, and a torn read only shifts one second's estimate.
        self._shared = ctx.Array('d', 2 * len(topics), lock=False)
        self._stop = ctx.Event()
        self._proc = ctx.Process(target=_run, args=(topics, self._shared, self._stop),
                                 daemon=True)
        self._proc.start()

    def counts(self):
        """Return {topic: (count, last arrival)}."""
        return {t: (int(self._shared[2 * i]), self._shared[2 * i + 1])
                for i, t in enumerate(self.topics)}

    def close(self):
        self._stop.set()
        self._proc.join(timeout=2.0)
        if self._proc.is_alive():
            self._proc.terminate()
