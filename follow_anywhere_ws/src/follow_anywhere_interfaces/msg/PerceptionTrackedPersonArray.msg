# All tracked people of one processed frame (an empty list is published too).
# header.stamp: time of the camera image. header.frame_id: base_link.
std_msgs/Header header
PerceptionTrackedPerson[] people
