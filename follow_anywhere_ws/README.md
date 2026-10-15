# follow_anywhere_ws

ROS2 (Lyrical) workspace for the follow-anywhere project on the physical Unitree Go2.

| Package | Purpose |
|---|---|
| `unitree_api` | Go2 request/response messages (`/api/sport/request`, ...). Backbone. |
| `unitree_go` | Go2 state/command messages (`/sportmodestate`, `/lowstate`, ...). Backbone. |
| `manual_controller` | WASD/arrow-key manual control. |
| `navigation`, `vision` | Placeholders. Add `<depend>unitree_go</depend>` to use the Go2 messages. |
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
