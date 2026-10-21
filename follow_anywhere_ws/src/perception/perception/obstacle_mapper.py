"""Builds the obstacle list for navigation: static from the lidar, dynamic from the camera. Plain python, no ROS.

To include:
- Static obstacles: lidar /utlidar/cloud_base (already in base_link). Remove the ground
  (ground_height), ignore points beyond obstacle_max_range, reduce to obstacles with position + size,
  no id, no velocity, source = SOURCE_LIDAR.
- Fallback without lidar (use_lidar off or no cloud): the same from the depth image points
  (coordinate_conversion), lidar_used = false.
- Dynamic obstacles: tracked people from velocity_tracker (id, position, size from the box + depth,
  velocity, flags), source = SOURCE_CAMERA, class_name = person.
- Points that belong to a tracked person should not also appear as a static obstacle.
- static / dynamic is UNKNOWN for anything that cannot be decided (e.g. the robot moves without
  odometry).
- Later: clustering + tracking of lidar objects that move (out of the camera's view).
"""
