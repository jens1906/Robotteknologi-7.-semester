# follow_anywhere_ws

ROS2 (Jazzy) workspace for the follow-anywhere project on the physical Unitree Go2.

| Package | Purpose |
|---|---|
| `unitree_api` | Go2 request/response messages (`/api/sport/request`, ...). Backbone. |
| `unitree_go` | Go2 state/command messages (`/sportmodestate`, `/lowstate`, ...). Backbone. |
| `manual_controller` | WASD/arrow-key manual control. |
| `navigation`, `vision` | Placeholders. Add `<depend>unitree_go</depend>` to use the Go2 messages. |

`unitree_api` and `unitree_go` are the Jazzy-patched copies from `go2_real_ws/src`. Keep their names, since the type names must match what the robot publishes.

## Build

```bash
cd follow_anywhere_ws
source /opt/ros/jazzy/setup.bash
colcon build
```

## Every new terminal

```bash
cd follow_anywhere_ws
source /opt/ros/jazzy/setup.bash
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
