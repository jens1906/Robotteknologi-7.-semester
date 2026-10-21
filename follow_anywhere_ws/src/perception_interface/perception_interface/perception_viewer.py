"""The viewer ROS node: shows what perception sees and decided, and calibrates the leader by click.

Runs on the laptop. Started with `ros2 launch perception_interface viewer.launch.py`.

To include:
- main(): rclpy init, create the node, spin; the OpenCV window is updated from a timer.
- Parameters from config/viewer_params.yaml.
- Subscriptions, all QoS best effort + keep last 1 (a slow WiFi skips frames instead of falling behind):
  - compressed colour image (sensor_msgs/CompressedImage, `.../image_raw/compressed`), decoded with
    cv2.imdecode. Never the raw image: ~27 MB/s over WiFi against ~1-2 MB/s compressed.
  - /perception/tracked_people and /perception/leader_state (small messages).
- No frame cap: shows as many frames as the network and the laptop manage, up to the camera's rate.
- Matching: keep the last image_buffer images by timestamp; a result (stamped with its camera image's
  time) is drawn on the image with the same stamp.
  - exact mode (default, B): only images that have a result are shown -> boxes exactly on people,
    shown at perception's rate.
  - live mode (A): every image, with the newest result -> smooth, boxes trail moving people a little.
  - key 'm' switches mode.
- Click on a person: call /perception/calibrate_leader (PerceptionCalibrateLeader) with that person's
  track id; key 'c': call it with -1 (person nearest the image centre). Show the service's message
  ("enrolling #7" / "nobody there") on screen for a few seconds.
- Keys: m mode, c calibrate centre, q quit.
- Header line: perception state, shown fps, image age (now - stamp), how many images were dropped.
- Warn on screen when no image or no perception result arrived for a while.
- Drawing itself is in overlay.py.
"""
