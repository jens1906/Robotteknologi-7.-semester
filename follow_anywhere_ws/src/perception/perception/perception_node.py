"""The perception ROS node: start-up, threads, publishing. The only file besides sensor_input.py that uses ROS.

To include:
- main(): rclpy init, create the node, spin (entry point `perception_node` in setup.py).
- Read every parameter from config/perception_params.yaml and create the parts with them:
  SensorInput, PersonDetector, ReidModel, LeaderManager, CoordinateConversion, VelocityTracker,
  ObstacleMapper.
- Device: pick the GPU once ('auto' / 'cuda' / 'cpu'); warn loudly when it falls back to the CPU.
- Python packages: the node runs with the system python (plain `colcon build`); append the workspace
  .venv site-packages at the END of sys.path so only missing packages (ultralytics, onnxruntime, torch)
  come from the venv and the system numpy / cv_bridge keep priority.
- Two threads, so re-identification never stalls the video:
  1. frames: newest frame -> person_detector -> coordinate_conversion -> velocity_tracker ->
     obstacle_mapper -> publish tracked_people + obstacles.
  2. leader: re-identification for leader_manager (enrol / verify / search), using the newest people;
     its result is merged into the next published leader_state.
- Publish (all stamped with the camera image's time, frame base_link):
  /perception/leader_state, /perception/obstacles, /perception/tracked_people.
- Service /perception/calibrate_leader (follow_anywhere_interfaces/PerceptionCalibrateLeader): enrol the
  person with the given track id, -1 = the person nearest the image centre. Called by the viewer
  (perception_interface, click), the terminal, later a remote-control button.
- Headless: no window here. The viewer runs in perception_interface on a laptop and reads the
  compressed image the RealSense driver publishes (raw images never leave the robot).
- Fill velocity fields with NaN and velocity_valid = false while use_velocity is off.
"""
