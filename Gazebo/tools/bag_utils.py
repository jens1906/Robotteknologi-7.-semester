"""Reading Go2 bags for the sim test tools."""

import glob
import os

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def open_bag(path, topics=None):
    """Open a bag dir or .mcap file. Returns (reader, {topic: type})."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, '*.mcap')))
        if not files:
            raise SystemExit(f'No .mcap file in {path}')
        # Open the file itself: metadata.yaml may name a file that was renamed.
        path = files[0]
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id='mcap'),
                rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topics is not None:
        reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    return reader, types


def read_messages(path, topics):
    """Yield (topic, bag_time_ns, message) for the given topics."""
    reader, types = open_bag(path, topics)
    classes = {t: get_message(types[t]) for t in topics if t in types}
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        yield topic, stamp, deserialize_message(data, classes[topic])
