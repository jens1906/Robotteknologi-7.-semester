# Live metrics of one evaluation run (web viewer record button = one run). For rosbags and PlotJuggler.
# header.stamp: newest perception stamp. NaN = not measured yet in this run.

std_msgs/Header header
float32 run_time                           # s since the first perception message of the run

# distance to the leader (perception's belief, VISIBLE frames only)
bool leader_visible
float32 leader_distance                    # m, this frame
float32 distance_error                     # m, leader_distance - desired distance
float32 average_distance                   # m
float32 average_absolute_error             # m, mean |distance - desired distance|
float32 distance_rmse                      # m, to the desired distance
float32 minimum_distance                   # m
float32 maximum_distance                   # m

# line of sight losses (leader_state FOLLOWING -> SEARCHING / LOST)
int32 loss_count
float32 loss_time_ratio                    # share of the time since first FOLLOWING spent searching / lost
float32 current_loss_duration              # s, 0 while FOLLOWING
int32 recovery_count
float32 average_recovery_time              # s
float32 last_recovery_time                 # s
int32 suspected_switch_count               # leader track id changed without a loss, or recovered far from where it was lost

# obstacles (/perception/obstacles_combined), hints for collisions
float32 obstacle_clearance                 # m, nearest obstacle this message
float32 minimum_obstacle_clearance         # m
int32 near_collision_count                 # times the clearance went below the near collision distance

# manual marks from the web viewer buttons
int32 collision_mark_count
int32 wrong_leader_mark_count

# runtime
float32 perception_rate                    # Hz of /perception/leader, last rate window
float32 navigation_rate                    # Hz of /api/sport/request, last rate window
float32 frame_time_ms                      # perception frame callback, this frame
