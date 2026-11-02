# All objects seen in one perception frame. header.frame_id is base_link and
# header.stamp the time the robot was at the pose the detections are relative to.
std_msgs/Header header
Detection[] detections
