# What perception saw and decided in one camera frame. For the viewer, PlotJuggler and rosbags; not an output
# (the output is /perception/leader and /perception/obstacles).
# header.stamp: time of the camera image. header.frame_id: base_link (positions); boxes are in image pixels.

uint8 IDLE = 0                             # no leader enrolled
uint8 ENROLLING = 1                        # collecting views of the chosen person
uint8 FOLLOWING = 2                        # leader identified (or missing shorter than graceTime)
uint8 SEARCHING = 3                        # leader known but not identified right now
uint8 LOST = 4                             # not identified for longer than lostTime

std_msgs/Header header

# leader logic
uint8 leader_state
int32 leader_track_id                      # tracker id of the leader in this frame, -1 = not identified
float32 seconds_since_seen                 # 0 while identified, then grows; NaN if never seen
float32 leader_distance                    # m, horizontal; NaN when not identified or no depth
geometry_msgs/Point last_seen_position     # where the leader was last identified; NaN if never seen

# stored leader views per group (calibration + learned)
uint8 front_views
uint8 back_views
uint8 side_left_views
uint8 side_right_views
uint8 partly_hidden_views

# cost of this frame
float32 detector_time_ms                   # person detector + tracker
float32 reid_time_ms                       # OSNet, all crops of this frame
int32 reid_crop_count                      # crops OSNet embedded this frame
int32 people_count                         # people tracked this frame

PerceptionDebugPerson[] people
