"""Positions and velocities of tracked people (and the leader) over time. Plain python, no ROS.

To include:
- One kalman_filter per tracker id, fed with that person's 3D position from coordinate_conversion.
- A new tracker id gets a fresh filter (an id change would otherwise look like a jump in velocity);
  a filter without measurements for track_timeout is dropped.
- use_odometry true: shift every filter by the robot's motion between the two frames (robot pose at
  each image's timestamp, from sensor_input) -> velocity_compensated = true.
  use_odometry false: no shift -> relative velocity, includes the robot's own motion.
- use_velocity false: still filter (it smooths positions), but report velocity_valid = false.
- Gives the smoothed position, velocity and uncertainty per id to leader_state and obstacle_mapper.
"""
