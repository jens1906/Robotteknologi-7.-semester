# All obstacles of one camera frame or one lidar cloud (an empty list is published too).
# header.stamp: time of the camera image or lidar cloud they were computed from. header.frame_id: base_link.

std_msgs/Header header
PerceptionObstacle[] obstacles
