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
