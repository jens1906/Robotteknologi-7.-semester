"""Pixel + depth -> 3D position in base_link. Plain python, no ROS.

To include:
- Camera intrinsics from camera_info (fx, fy, cx, cy): pixel (u, v) + depth z -> point in the camera
  optical frame.
- Camera optical frame -> base_link with the mounting transform (from TF, given by perception_node).
- A person's depth: median of the depth around the torso keypoints (shoulders, hips), not the box
  centre (that often hits the background). Ignore 0 / out-of-range depth (depth_min..depth_max);
  no usable depth -> position_valid = false.
- distance (horizontal) and angle (0 = straight ahead, positive = left) for the leader message.
- Depth image -> points for obstacle_mapper's fallback when there is no lidar.
"""
