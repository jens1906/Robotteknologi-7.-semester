# One tracked person in one camera frame, for the viewer and debugging.

int32 id                                   # tracker id, kept while the tracker follows the person
float32 confidence                         # detector score, 0..1

# bounding box in image pixels
float32 x_min
float32 y_min
float32 x_max
float32 y_max

bool is_leader
float32 reid_distance                      # distance to the leader's stored views, NaN = not compared recently

geometry_msgs/Point position               # base_link, m (measured front of the torso)
bool position_valid                        # false when there was no usable depth
geometry_msgs/Vector3 velocity             # base_link, m/s, over the ground (this person's own filter, also for the leader)
float32 velocity_uncertainty               # m/s, 1 sigma; NaN when there was no position
