# Where the leader is, published for every processed camera frame.
# Frame: base_link (x forward, y left, z up). Units: m, m/s, rad.
# header.stamp is the time of the camera image this was computed from, not the publish time.

uint8 IDLE = 0        # no leader enrolled
uint8 ENROLLING = 1   # collecting views of the chosen person
uint8 FOLLOWING = 2   # leader identified in this frame
uint8 SEARCHING = 3   # leader known but not identified right now (lost recently)
uint8 LOST = 4        # not identified for a long time

std_msgs/Header header
uint8 state
int32 track_id                        # tracker id of the leader, -1 if not identified

# valid while FOLLOWING
float32 distance                      # m, horizontal distance to the leader
float32 angle                         # rad, 0 = straight ahead, positive = to the left
geometry_msgs/Point position

# NaN while velocity_valid is false (the use_velocity parameter is off)
geometry_msgs/Vector3 velocity
bool velocity_valid
bool velocity_compensated             # true: the robot's own motion is removed (odometry used)

# where and when the leader was last identified (for SEARCHING / LOST)
builtin_interfaces/Time last_seen_time
geometry_msgs/Point last_seen_position
