# SPDX-License-Identifier: BSD-3-Clause

"""
H.264 decoding of the Go2 head camera (/frontvideostream).

Each message holds one Annex-B access unit. PyAV decodes in-process; without it
an ffmpeg subprocess does the work. That needs ffmpeg's native h264 decoder:
Fedora's ffmpeg-free only has OpenH264, which cannot decode the robot's High
profile stream, so install PyAV (requirements.txt).

Frames are dropped until the first SPS (sent with every IDR), so joining
mid-stream gives no grey smear.
"""

import subprocess
import threading

import numpy as np

try:
    import av
except ImportError:  # pragma: no cover - depends on the machine
    av = None

NAL_SPS = 7


def has_sps(data: bytes) -> bool:
    """Return True if the Annex-B access unit contains an SPS NAL unit."""
    i = data.find(b'\x00\x00\x01')
    while i >= 0 and i + 3 < len(data):
        if data[i + 3] & 0x1F == NAL_SPS:
            return True
        i = data.find(b'\x00\x00\x01', i + 3)
    return False


class H264Decoder:
    """Feed access units with `push`; decoded RGB frames go to `on_frame(ndarray)`."""

    def __init__(self, on_frame, width: int, height: int):
        self._on_frame = on_frame
        self._synced = False
        self.backend = 'pyav' if av is not None else 'ffmpeg'
        if av is not None:
            self._codec = av.CodecContext.create('h264', 'r')
        else:
            self._size = (width, height)
            self._proc = subprocess.Popen(
                ['ffmpeg', '-loglevel', 'error', '-fflags', 'nobuffer', '-flags', 'low_delay',
                 '-probesize', '32', '-analyzeduration', '0',
                 '-f', 'h264', '-i', 'pipe:', '-vf', f'scale={width}:{height}',
                 '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            threading.Thread(target=self._read_loop, daemon=True).start()

    def push(self, data: bytes) -> None:
        if not self._synced:
            if not has_sps(data):
                return
            self._synced = True
        if av is not None:
            try:
                for frame in self._codec.decode(av.Packet(data)):
                    self._on_frame(frame.to_ndarray(format='rgb24'))
            except av.error.FFmpegError:
                self._synced = False
        else:
            try:
                self._proc.stdin.write(data)
                self._proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                pass

    def _read_loop(self) -> None:
        width, height = self._size
        n = width * height * 3
        while True:
            buf = self._proc.stdout.read(n)
            if len(buf) < n:
                return
            self._on_frame(np.frombuffer(buf, np.uint8).reshape(height, width, 3))

    def close(self) -> None:
        if av is None and self._proc.poll() is None:
            self._proc.kill()
