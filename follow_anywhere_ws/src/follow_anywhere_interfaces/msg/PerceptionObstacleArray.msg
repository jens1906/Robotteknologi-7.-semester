# All obstacles of one camera frame (an empty list is published too).
# header.stamp: time of the camera image they were computed from. header.frame_id: base_link.

std_msgs/Header header
PerceptionObstacle[] obstacles
