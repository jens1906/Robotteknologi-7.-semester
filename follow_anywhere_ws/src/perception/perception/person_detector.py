"""Finds people in the colour image and keeps their ids from frame to frame. Plain python, no ROS.

To include:
- YOLO pose model (yolo26n-pose.pt) + BoT-SORT (config/botsort.yaml) through ultralytics `track`.
- Returns per person: tracker id, box (pixels), confidence, 17 COCO keypoints (x, y, confidence).
- Perception re-evaluation: the task comes from the model file, so a model without keypoints (e.g. yolo26n-seg)
  gives all keypoints with confidence 0 (= nothing visible, what the rest does without keypoints).
  trackerConfigFile none = detection only, ids are only the order within the frame (detector recall runs).
  odometryGmc: BoT-SORT's camera motion compensation (gmc.apply, sparse optical flow) is replaced by the image shift
  the robot's own turn gives (focal length x heading change since the last frame, from /utlidar/robot_odom).
- Ignore boxes smaller than detector_min_box.
- imgsz 640 for the RealSense (its own resolution).
- A TensorRT engine (detectorModelFile ending in .engine, made on the Orin by scripts/fetch_models.py --engine) is
  loaded the same way; its precision and image size were fixed when it was built.
- reset(): forget all tracks (e.g. a replayed rosbag starts over). Ultralytics ignores `persist` after
  the first call, so reset means reloading the model.
- Set YOLO_AUTOINSTALL=False: ultralytics would otherwise pip-install packages into the environment.
- The ids only hold for seconds (until a person is hidden or leaves the image). Recognising the leader
  again under a new id is leader_manager's job.
"""

import math
import os
from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO

# Parameter defaults (set in config/perception_params.yaml)
DETECTOR_MODEL_PATH = 'yolo26n-pose.pt'
TRACKER_CONFIG_PATH = 'botsort.yaml'
DETECTOR_IMAGE_SIZE = 640
MINIMUM_BOX_SIZE_PIXELS = 24
REQUESTED_DEVICE = 'auto'

# Detection
PERSON_CLASS_ID = 0
KEYPOINT_COUNT = 17
NO_TRACKER = 'none'
HALF_PRECISION = 16
TENSORRT_ENGINE_EXTENSION = '.engine'
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
                 requestedDevice=REQUESTED_DEVICE, odometryGmc=False):
        self.trackerConfigPath = trackerConfigPath
        self.odometryGmc = odometryGmc
        self.previousHeading = None
        self.odometryWarp = np.eye(2, 3)
        self.detectorImageSize = detectorImageSize
        self.minimumBoxSizePixels = minimumBoxSizePixels
        self.inferenceDevice, self.inferenceDeviceWarning = SelectInferenceDevice(requestedDevice)
        isTensorrtEngine = detectorModelPath.endswith(TENSORRT_ENGINE_EXTENSION)
        self.inferencePrecision = HALF_PRECISION if self.inferenceDevice.startswith('cuda') and not isTensorrtEngine else None
        self.model = YOLO(detectorModelPath)

    def DetectAndTrackPeople(self, imageBgr, robotHeading=None, focalLengthPixels=None):
        if self.odometryGmc:
            self.PrepareOdometryWarp(robotHeading, focalLengthPixels)
        if os.path.basename(self.trackerConfigPath) == NO_TRACKER:
            trackingResult = self.model.predict(imageBgr, classes=[PERSON_CLASS_ID], imgsz=self.detectorImageSize,
                                                device=self.inferenceDevice, quantize=self.inferencePrecision,
                                                verbose=False)[0]
            trackIds = list(range(len(trackingResult.boxes)))
        else:
            trackingResult = self.model.track(imageBgr, persist=True, tracker=self.trackerConfigPath,
                                              classes=[PERSON_CLASS_ID], imgsz=self.detectorImageSize,
                                              device=self.inferenceDevice, quantize=self.inferencePrecision,
                                              verbose=False)[0]
            if trackingResult.boxes is None or trackingResult.boxes.id is None:
                return []
            trackIds = trackingResult.boxes.id.int().tolist()
        allBoundingBoxValues = trackingResult.boxes.xyxy.tolist()
        confidences = trackingResult.boxes.conf.tolist()
        allBodyKeypoints = trackingResult.keypoints.data.cpu().numpy() if trackingResult.keypoints is not None \
            else np.zeros((len(trackIds), KEYPOINT_COUNT, 3), np.float32)
        detectedPeople = []
        for trackId, boundingBoxValues, confidence, bodyKeypoints in zip(trackIds, allBoundingBoxValues, confidences,
                                                                          allBodyKeypoints):
            left, top, right, bottom = boundingBoxValues
            if right - left < self.minimumBoxSizePixels or bottom - top < self.minimumBoxSizePixels:
                continue
            detectedPeople.append(DetectedPerson(trackId, left, top, right, bottom, confidence, bodyKeypoints))
        return detectedPeople

    # shortcut: yaw only (no translation, no pitch from the 30 deg camera tilt), upgrade if odometry GMC is kept
    def PrepareOdometryWarp(self, robotHeading, focalLengthPixels):
        self.odometryWarp = np.eye(2, 3)
        if robotHeading is not None and focalLengthPixels and self.previousHeading is not None:
            headingChange = math.atan2(math.sin(robotHeading - self.previousHeading),
                                       math.cos(robotHeading - self.previousHeading))
            self.odometryWarp[0, 2] = focalLengthPixels * math.tan(headingChange)
        self.previousHeading = robotHeading
        for tracker in getattr(self.model.predictor, 'trackers', None) or []:
            tracker.gmc.apply = lambda rawFrame, detections=None: self.odometryWarp
