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

Parameters: `address` (`0.0.0.0`), `port` (8080), `image_hz`/`cloud_hz`/`json_hz` (15/10/10, max rate sent to the browser, doubles), `max_points` (40000, clouds are thinned to this), `max_array` (64, longer arrays in JSON are cut), `jpeg_quality` (70), `max_width` (960, images are scaled down to this), `depth_max_m` (6.0, top of the depth colour map), `default_topics`, `www_root` (serve the page from another directory, e.g. `src/web_viewer/www` while editing it).

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
