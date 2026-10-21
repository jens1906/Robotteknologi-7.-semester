# follow_anywhere_ws

ROS2 (Lyrical) workspace for the follow-anywhere project on the physical Unitree Go2.

| Package | Purpose |
|---|---|
| `unitree_api` | Go2 request/response messages (`/api/sport/request`, ...). Backbone. |
| `unitree_go` | Go2 state/command messages (`/sportmodestate`, `/lowstate`, ...). Backbone. |
| `manual_controller` | WASD/arrow-key manual control. |
| `follow_anywhere_interfaces` | Messages of all parts, named after the owning part (`Perception...`, later `Navigation...`). |
| `perception` | People detection (YOLO pose + BoT-SORT) and leader identification (OSNet) on the RealSense colour image: `/perception/tracked_people`, `/perception/leader_state`. Depth, obstacles come in later steps. Run: `ros2 launch perception perception.launch.py`. |
| `perception_interface` | Viewer for perception, runs on the laptop: compressed camera feed with boxes, track ids and the leader; click a person (or `c`) to make them the leader, `r` forgets the leader. Run: `ros2 launch perception_interface viewer.launch.py`. |
| `navigation` | Placeholder. Add `<depend>unitree_go</depend>` to use the Go2 messages. |

`unitree_api` and `unitree_go` are the Lyrical-patched copies from `go2_real_ws/src`. Keep their names, since the type names must match what the robot publishes.

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

## Models for perception

The models are not in git. After a pull, fetch them once and build again (the files are installed at build time):

```bash
python3 scripts/fetch_models.py      # YOLO pose + OSNet into src/perception/resource/models/
colcon build --symlink-install
```

The saved leader lives in `~/.ros/follow_anywhere/perception_leader.npz` (loaded at start, deleted by `ros2 service call /perception/clear_leader std_srvs/srv/Trigger`).
