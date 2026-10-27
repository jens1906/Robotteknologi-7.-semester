# The leader. Frame: base_link (x forward, y left), z = 0. Units: m, m/s, s.
# header.stamp is the time of the camera image this was computed from.

uint8 NO_LEADER = 0                        # nobody calibrated yet (or calibration still running)
uint8 VISIBLE = 1                          # identified in this image
uint8 LOST = 2                             # known, but not identified in this image

std_msgs/Header header
uint8 status
geometry_msgs/Point closest_point          # nearest point of the leader to the robot, this frame; NaN when not VISIBLE or no depth this frame
geometry_msgs/Vector3 velocity             # over the ground (the robot's own motion removed); 0 when standing still; while LOST the velocity of the last sighting
float32 position_uncertainty               # m, 1 sigma; while LOST it grows, and is the uncertainty around last_known_point + velocity * lost_duration
float32 velocity_uncertainty               # m/s, 1 sigma
float32 lost_duration                      # s since last identified, 0 while VISIBLE, NaN if never seen since start
geometry_msgs/Point last_known_point       # last valid closest_point, kept in place on the floor while the robot moves (also the end of the line of sight while LOST); NaN if never seen
