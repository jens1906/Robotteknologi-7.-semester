"""Finds people in the colour image and keeps their ids from frame to frame. Plain python, no ROS.

To include:
- YOLO pose model (yolov8n-pose.pt) + BoT-SORT (config/botsort.yaml) through ultralytics `track`.
- Returns per person: tracker id, box (pixels), confidence, 17 COCO keypoints (x, y, confidence).
- Ignore boxes smaller than detector_min_box.
- imgsz 640 for the RealSense (its own resolution).
- reset(): forget all tracks (e.g. a replayed rosbag starts over). Ultralytics ignores `persist` after
  the first call, so reset means reloading the model.
- Set YOLO_AUTOINSTALL=False: ultralytics would otherwise pip-install packages into the environment.
- The ids only hold for seconds (until a person is hidden or leaves the image). Recognising the leader
  again under a new id is leader_manager's job.
"""

from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO

# Parameter defaults (set in config/perception_params.yaml)
DETECTOR_MODEL_PATH = 'yolov8n-pose.pt'
TRACKER_CONFIG_PATH = 'botsort.yaml'
DETECTOR_IMAGE_SIZE = 640
MINIMUM_BOX_SIZE_PIXELS = 24
REQUESTED_DEVICE = 'auto'

# Detection
PERSON_CLASS_ID = 0
HALF_PRECISION = 16
NO_CUDA_GPU_WARNING = 'no CUDA GPU found, the detector runs on the CPU which is slow'


@dataclass
class DetectedPerson:
    trackId: int
    left: float
    top: float
    right: float
    bottom: float
    confidence: float
    bodyKeypoints: np.ndarray


def SelectInferenceDevice(requestedDevice):
    if requestedDevice not in ('auto', 'cuda'):
        return requestedDevice, ''
    if torch.cuda.is_available():
        return 'cuda:0', ''
    return 'cpu', NO_CUDA_GPU_WARNING


class PersonDetector:
    def __init__(self, detectorModelPath=DETECTOR_MODEL_PATH, trackerConfigPath=TRACKER_CONFIG_PATH,
                 detectorImageSize=DETECTOR_IMAGE_SIZE, minimumBoxSizePixels=MINIMUM_BOX_SIZE_PIXELS,
                 requestedDevice=REQUESTED_DEVICE):
        self.trackerConfigPath = trackerConfigPath
        self.detectorImageSize = detectorImageSize
        self.minimumBoxSizePixels = minimumBoxSizePixels
        self.inferenceDevice, self.inferenceDeviceWarning = SelectInferenceDevice(requestedDevice)
        self.inferencePrecision = HALF_PRECISION if self.inferenceDevice.startswith('cuda') else None
        self.model = YOLO(detectorModelPath)

    def DetectAndTrackPeople(self, imageBgr):
        trackingResult = self.model.track(imageBgr, persist=True, tracker=self.trackerConfigPath,
                                          classes=[PERSON_CLASS_ID], imgsz=self.detectorImageSize,
                                          device=self.inferenceDevice, quantize=self.inferencePrecision,
                                          verbose=False)[0]
        if trackingResult.boxes is None or trackingResult.boxes.id is None:
            return []
        trackIds = trackingResult.boxes.id.int().tolist()
        allBoundingBoxValues = trackingResult.boxes.xyxy.tolist()
        confidences = trackingResult.boxes.conf.tolist()
        allBodyKeypoints = trackingResult.keypoints.data.cpu().numpy()
        detectedPeople = []
        for trackId, boundingBoxValues, confidence, bodyKeypoints in zip(trackIds, allBoundingBoxValues, confidences,
                                                                          allBodyKeypoints):
            left, top, right, bottom = boundingBoxValues
            if right - left < self.minimumBoxSizePixels or bottom - top < self.minimumBoxSizePixels:
                continue
            detectedPeople.append(DetectedPerson(trackId, left, top, right, bottom, confidence, bodyKeypoints))
        return detectedPeople
