"""The Kalman filter maths, separate so it can be tuned and tested on its own. Plain python (numpy), no ROS.

To include:
- Constant-velocity model, state [x, y, vx, vy] in base_link (z is not needed for the robot).
- predict(dt): move the state ahead; process noise from kalman_process_noise.
- predict with ego-motion: before predicting, shift the state by the inverse of the robot's own motion
  since the last update (rotation + translation from odometry). Without odometry this shift is skipped.
- update(measured position): measurement noise from kalman_measurement_noise.
- Read-out: position, velocity, their uncertainty (1 sigma).
- Later: predicting further ahead (the 10 s idea) uses the same predict, with growing uncertainty.
"""
