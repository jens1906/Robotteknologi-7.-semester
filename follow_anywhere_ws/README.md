# follow_anywhere_ws

ROS2 (Lyrical) workspace for the follow-anywhere project on the physical Unitree Go2.

| Package | Purpose |
|---|---|
| `unitree_api` | Go2 request/response messages (`/api/sport/request`, ...). Backbone. |
| `unitree_go` | Go2 state/command messages (`/sportmodestate`, `/lowstate`, ...). Backbone. |
| `manual_controller` | WASD/arrow-key manual control. |
| `navigation`, `vision` | Placeholders. Add `<depend>unitree_go</depend>` to use the Go2 messages. |
| `go2_verify` | Live dashboard to check the Go2 works: head camera, RealSense colour/depth, lidar point cloud, state charts and per-topic rates. See [Verify](#verify). |
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
