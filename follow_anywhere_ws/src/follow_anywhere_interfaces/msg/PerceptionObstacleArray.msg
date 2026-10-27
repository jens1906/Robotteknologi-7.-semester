# All obstacles of one processed frame (an empty list is published too).
# header.stamp: time of the sensor data they were computed from. header.frame_id: base_link.
std_msgs/Header header
bool lidar_used                       # false: lidar missing or switched off, depth image only
PerceptionObstacle[] obstacles
