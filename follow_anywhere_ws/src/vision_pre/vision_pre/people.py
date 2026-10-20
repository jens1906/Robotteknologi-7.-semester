"""People in a frame: YOLO pose detections with a tracker id (BoT-SORT) and COCO-17 keypoints."""
import os
from dataclasses import dataclass

import numpy as np

MIN_BOX = 24     # px, smaller detections are ignored


@dataclass
class Person:
    id: int
    box: tuple              # x1, y1, x2, y2 in image pixels
    conf: float
    kps: np.ndarray         # (17, 3) x, y, confidence in image pixels

    @property
    def area(self):
        return max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1])


def default_tracker():
    """The tuned BoT-SORT config of this package (source tree or ROS share dir), else ByteTrack."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'config', 'botsort_tuned.yaml')
    if os.path.isfile(path):
        return path
    try:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory('vision_pre'), 'config', 'botsort_tuned.yaml')
        if os.path.isfile(path):
            return path
    except Exception:
        pass
    return 'bytetrack.yaml'


class PersonTracker:
    def __init__(self, model='yolov8n-pose.pt', imgsz=640, tracker=None, device=None):
        # ultralytics would pip-install missing packages into the environment (it once pulled in
        # numpy 2 and broke everything), so keep it from doing that
        os.environ.setdefault('YOLO_AUTOINSTALL', 'False')
        from ultralytics import YOLO
        self._load = lambda: YOLO(model)
        self.yolo = self._load()
        self.imgsz, self.tracker, self.device = imgsz, tracker or default_tracker(), device

    def reset(self):
        """Forget all tracks, e.g. when a video loops. Track ids keep counting up, they are not reused."""
        self.yolo = self._load()

    def __call__(self, frame_bgr):
        """Tracked people of this frame. Call it on consecutive frames: the tracker keeps state."""
        res = self.yolo.track(frame_bgr, persist=True, tracker=self.tracker, classes=[0],
                              imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if res.boxes is None or res.boxes.id is None:
            return []
        kps = res.keypoints.data.cpu().numpy()
        return [Person(i, tuple(b), c, k) for i, b, c, k in
                zip(res.boxes.id.int().tolist(), res.boxes.xyxy.tolist(),
                    res.boxes.conf.tolist(), kps)
                if b[2] - b[0] >= MIN_BOX and b[3] - b[1] >= MIN_BOX]
