"""ROS subscribers for every input. Hands the newest data to perception_node; nothing else here uses ROS.

To include:
- Colour image + aligned depth image + camera_info from the RealSense, paired by timestamp
  (message_filters). Keep only the newest pair: when processing is slower than the camera, older
  frames are dropped, never queued.
- Lidar: latest /utlidar/cloud_base (PointCloud2, already in base_link). Report "no lidar" when none
  arrived for a while, so obstacle_mapper falls back to the depth image (use_lidar).
- Odometry: /utlidar/robot_odom (nav_msgs/Odometry, odom -> base_link). Keep a short history so the
  robot pose at an image's timestamp can be looked up (nearest sample), not just the newest one.
- Camera mounting: TF base_link -> camera optical frame (tf2), published by the launch file + driver.
- Warnings: a topic that is silent, and a topic with more than one publisher (a leftover test
  publisher once mixed its video into the camera feed).
- QoS: sensor data (best effort) for the camera.
"""
