# One tracked person in one camera frame. For debugging and rosbags: shows what the tracker and the
# leader logic saw and decided.

int32 id                              # tracker id, kept while the tracker follows the person
float32 confidence                    # detector score, 0..1

# bounding box in image pixels
float32 x_min
float32 y_min
float32 x_max
float32 y_max

bool is_leader
float32 reid_distance                 # distance to the leader's stored views, NaN = not compared this frame

geometry_msgs/Point position          # base_link, m
bool position_valid                   # false when there was no usable depth
