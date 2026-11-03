# follow_anywhere_ws

ROS2 (Lyrical) workspace for the follow-anywhere project on the physical Unitree Go2.

| Package | Purpose |
|---|---|
| `unitree_api` | Go2 request/response messages (`/api/sport/request`, ...). Backbone. |
| `unitree_go` | Go2 state/command messages (`/sportmodestate`, `/lowstate`, ...). Backbone. |
| `manual_controller` | WASD/arrow-key manual control. |
| `navigation`, `vision` | Placeholders. Add `<depend>unitree_go</depend>` to use the Go2 messages. |
| `perception_msgs` | `Detection` / `DetectionArray`: the leader, people and obstacles in polar coordinates around `base_link`, with class, stds and estimated velocity. Output of `vision`, input of `navigation`. `2dSim`'s `fake_perception` publishes it on `/perception/detections`. |
| `go2_verify` | Live dashboard to check the Go2 works: head camera, RealSense colour/depth, lidar point cloud, state charts and per-topic rates. See [Verify](#verify). |
| `web_viewer` | C++ web server (Boost.Beast) to view any topic, lidar clouds and camera images in a browser over WebSockets. See [Web viewer](#web-viewer). |
| `tmp_lidar` | C++ lidar part of perception: obstacle clusters from `/utlidar/cloud_base`, fused with the camera's leader and people (lidar positions, duplicates removed). Publishes `/perception/leader`, `/perception/obstacles`, `/perception/scan`. `ros2 launch tmp_lidar perception_fused.launch.py`. See [src/tmp_lidar/README.md](src/tmp_lidar/README.md). |
| `go2_bringup` | TF for the Go2 (the robot publishes none): sensor mounts from `config/go2_mounts.yaml` and `odom -> base_link` from `/utlidar/robot_odom`. `ros2 launch go2_bringup go2_tf.launch.py`. Same on the real robot and the sim. |

`unitree_api` and `unitree_go` are the Lyrical-patched copies from `go2_real_ws/src`. Keep their names, since the type names must match what the robot publishes. `Go2FrontVideoData` uses the layout the robot actually sends (`time_frame`, `height`, `data`), not Unitree's repo definition, and `VoxelMapCompressed` is added; both were measured from the bag `go2_all_topics_6_10_11_28`. The Gazebo sim builds these same packages.

## Build

```bash
cd follow_anywhere_ws
source /opt/ros/lyrical/setup.bash
colcon build
```

## Every new terminal

```bash
cd follow_anywhere_ws
source /opt/ros/lyrical/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="YOUR_NIC" priority="default" multicast="default"/></Interfaces></General></Domain></CycloneDDS>'
```

Replace `YOUR_NIC` with the interface connected to the Go2 (see `go2_real_ws/README.md` for the network setup).

## Manual control

```bash
scripts/drive_go2.sh [NIC]
```

The script sources ROS and the workspace (building it first if needed), sets up CycloneDDS on the Go2 interface, warns if the Go2 does not answer ping, and starts the teleop. Without `NIC` it uses `$GO2_NIC`, then the interface with a `192.168.123.x` address, then the first `en*` link. Manual equivalent after the steps above: `ros2 run manual_controller keyboard_teleop`.

W/S forward/back, A/D strafe, left/right arrows turn, up/down arrows speed, U stand up, J stand down, R recovery stand, space stop, Q quit.

## Web viewer

`web_viewer` serves a page on port 8080 that lists every topic. Click one to open a panel:

- `sensor_msgs/Image` (rgb8, bgr8, mono8, yuv422, 16UC1/32FC1 depth with a colour map) is sent as JPEG; `CompressedImage` jpeg/png is passed through as is.
- `PointCloud2` and `LaserScan` are drawn as a 3D orbit view (drag to orbit, right-drag to pan, wheel to zoom, `c` toggles height/intensity colour, `t` top view).
- Any other type is decoded through introspection and shown as YAML-like text.

The node only subscribes to a topic while a browser has it open, and drops frames for a slow client instead of queueing them. Open panels are remembered per browser; the first time it opens `default_topics`.

```bash
ros2 run web_viewer web_viewer
# then browse to http://<host>:8080
ros2 run web_viewer web_viewer --ros-args -p port:=9090 -p image_hz:=30.0 -p max_points:=80000
```

Parameters: `address` (`0.0.0.0`), `port` (8080), `image_hz`/`cloud_hz`/`json_hz` (15/10/10, max rate sent to the browser, doubles), `max_points` (40000, clouds are thinned to this), `max_array` (64, longer arrays in JSON are cut), `jpeg_quality` (70), `max_width` (960, images are scaled down to this), `depth_max_m` (6.0, top of the depth colour map), `default_topics`, `www_root` (serve the page from another directory, e.g. `src/web_viewer/www` while editing it), `bag_directory` (the repo's `RosBag/`), `record_exclude_regex` (raw images, theora/zstd streams and `/config_change_status`).

**Recording bags:** the ▶ button in the top right runs `ros2 bag record -a --exclude-regex <record_exclude_regex>` on the machine running web_viewer. While it records, the button shows ⏸ and a timer. Type a title and press ⏸ to stop: the bag is finalised and renamed to `<title>_MMDD_HHMM`, using the start time, in `bag_directory`. It records under `recording_MMDD_HHMM` until it is saved; a bag still named that way was not stopped from the page. The recorder inherits web_viewer's environment, so start web_viewer from a terminal with `RMW_IMPLEMENTATION`/`CYCLONEDDS_URI` set and `install/setup.bash` sourced. Errors are shown next to the button, logged, and published on `/web_viewer/debug`.

`/frontvideostream` (H.264) is not decoded; use the RealSense topics or `go2_verify` for the head camera.

## Verify

`go2_verify` opens a window showing what the Go2 (or the sim, or a bag) publishes: the head camera (`/frontvideostream`, decoded H.264), the RealSense colour and depth images, the lidar cloud (3D, mouse to orbit/zoom), charts of velocity, attitude, height, foot force and power, and a table of measured vs. expected topic rates (green ok, yellow slow, red missing).

Python deps once (into the user site of the interpreter ROS uses):

```bash
uv pip install --python /usr/bin/python3 --target ~/.local/lib/python3.14/site-packages \
    -r src/go2_verify/requirements.txt
```

Run in a terminal set up as above:

```bash
ros2 run go2_verify go2_verify
ros2 run go2_verify go2_verify --ros-args -p video_height:=720 -p lidar_3d:=false
```

Parameters: `video_height` (360/720), `lidar_topic` (`/utlidar/cloud_base`; `/utlidar/cloud_deskewed` is in `odom`), `lidar_3d` (false = 2D top-down), `history_s`, `refresh_hz`, `state_topic`/`lowstate_topic` (`/lf/*`, 18 Hz), `color_topic`, `depth_topic`, `count_high_rate` (rate-count `/lowstate` etc. in a child process).

Without the robot: `ros2 bag play ../go2_all_topics_6_10_11_28 --loop`. Use CycloneDDS for that (`export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`): Fast DDS fails to replay `/sportmodestate` from that bag.

With the Gazebo sim, `go2_verify` sets `FASTDDS_BUILTIN_TRANSPORTS=UDPv4` itself. The sim container runs as root, and its Fast DDS shared-memory segments in `/dev/shm` can't be read by the host user. So other host tools (`ros2 topic echo`, your own nodes) see the topics but receive no data. Export the same variable in their terminal.

## Camera calibration

`ros2 run perception calibrate_camera` measures where the RealSense sits on the Go2 (`base_link -> camera_link`) by aligning its depth points onto the Go2 lidar (`/utlidar/cloud_base`, already in `base_link`). It saves the result as `cameraX..cameraYaw` in `src/perception/config/perception_params.yaml`, and `perception.launch.py` publishes those values as the static TF at every start, so the camera and the lidar share the same 0 0 0. Settings: the `camera_calibration` section of the same file. The sim keeps its own estimate in `go2_bringup/config/go2_mounts.yaml`.

**Scene.** The camera is tilted ~30° up, so it never sees the floor; it and the lidar only overlap on the walls, from about camera height up to roughly 0.3 m (at 2 m) to 0.8 m (at 4 m) above it. Nothing about the layout is hardcoded: it only needs surfaces both sensors see, facing enough different directions.

- Robot standing still (or lying down, `J`) about 3 m from a room corner, pointing at it. Nobody in front during the ~15 s capture.
- A flat, matte board (≥ 1 × 1 m: plywood, a table top on its side, a big cardboard sheet) leaned against a wall at ~45–60°, 2.5–3 m ahead and within ~40° of straight ahead, its top ~1 m above the floor. Walls only fix x, y and the angles; the board fixes the height.
- Optional: a cabinet or box stack ~1 m tall, 2 m ahead and to the side.
- Brick, plaster and painted walls are fine. Keep windows and glass doors (closed curtains are fine), mirrors, shiny metal and direct sunlight out of the camera's view.

**Run** (on the Jetson, terminals set up as above):

```bash
ros2 launch perception perception.launch.py          # terminal 1: RealSense driver + TF
ros2 run perception calibrate_camera                 # terminal 2: ~15-30 s, then Ctrl+C
ros2 run perception calibrate_camera --ros-args -p paramsFile:=/tmp/copy.yaml   # read and write only that file
```

It first tries the saved pitch ±45° and yaw ±15° in 5° steps (the tilt is the hardest part to tape), then aligns from the best tries. It prints the overlap (depth pixels used / beyond `depthMax`, camera points above the lidar's view), the best tried tilt, the saved and the new mount, the residual, the matched points and the 4×4 matrix. It saves only when the residual is small, enough camera points lie on lidar surfaces, the scene holds the camera in every direction, and the camera position is within 0.15 m of the saved one; otherwise it says why (e.g. which direction is not held, or that camera and lidar see no common surface). Keep a rough tape measurement of the position in the file (±5 cm); check the tilt with a phone level if you can.

With the camera tilted up, most of its view is ceiling and upper wall that the lidar never sees, so the walls must be close: a room corner at 2.5-3 m, not a big hall (depth beyond `depthMax`, 4.5 m, is not used).

To send a calibration scene for debugging (~300 MB for 15 s, robot still):

```bash
ros2 bag record -o calib_scene /utlidar/cloud_base /camera/camera/aligned_depth_to_color/image_raw \
  /camera/camera/color/camera_info /camera/camera/color/image_raw/compressed /tf_static
```

**Check**: in the web viewer, open `/perception/calibration_cloud` and press `c`: lidar points have intensity 0, camera points moved with the new mount intensity 1; they should lie on the same walls. Restart `perception.launch.py` to use the new mount, repeat from 2-3 spots (results should agree within ~1 cm / 0.5°), and commit `perception_params.yaml` from the Jetson.

Don't run `go2_tf.launch.py` next to `perception.launch.py` on the robot: it publishes a second `base_link -> camera_link` (the sim estimate) and optical frames that clash with the RealSense driver's.
