# Robotteknologi-7.-semester

## Driving the Go2 with the keyboard

1. Connect the laptop to the Go2's Ethernet port and set that interface to manual IPv4 `192.168.123.99`, netmask `255.255.255.0` (`ping 192.168.123.161` should answer). Details in [go2_real_ws/README.md](go2_real_ws/README.md).
2. Install once: `sudo apt install ros-lyrical-rmw-cyclonedds-cpp`
3. Run:

```bash
follow_anywhere_ws/scripts/drive_go2.sh            # auto-detects the interface
follow_anywhere_ws/scripts/drive_go2.sh enx...     # or name it explicitly
```

The first run builds the workspace. Keys: W/S forward/back, A/D strafe, left/right arrows turn, up/down arrows speed, U stand up, J stand down, R recovery stand, space stop, Q quit. The workspace is documented in [follow_anywhere_ws/README.md](follow_anywhere_ws/README.md).
















# go2_real_ws

Standalone ROS2 (Lyrical) workspace for the **physical** Unitree Go2. It only holds the Go2 message packages (`unitree_go`, `unitree_api`). Build and source it to read topics and rosbags. It is independent of `Gazebo/`.

`src/unitree_go` and `src/unitree_api` are copies of the packages in `Gazebo/upstream/unitree_ros2/cyclonedds_ws/src/unitree/`, with the Connext DDS IDL block removed from their `CMakeLists.txt`. That block does not build on Lyrical and is not needed.

## One-time setup

```bash
sudo apt install ros-lyrical-rmw-cyclonedds-cpp ros-lyrical-rosbag2-storage-mcap
```

Connect the laptop to the Go2's Ethernet port (use a USB-Ethernet adapter if needed). Set that interface to manual IPv4 `192.168.123.99`, netmask `255.255.255.0`, then check:

```bash
ip -br link                 # find the interface name, e.g. enp3s0 / enx...
ping 192.168.123.161        # the Go2
```

The phone app keeps working over Wi-Fi/Bluetooth as usual.

## Build (once)

```bash
cd go2_real_ws
source /opt/ros/lyrical/setup.bash
colcon build
```

## Every new terminal

```bash
cd go2_real_ws
source /opt/ros/lyrical/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="YOUR_NIC" priority="default" multicast="default"/></Interfaces></General></Domain></CycloneDDS>'
```

Replace `YOUR_NIC` with the interface connected to the Go2.

## Use

```bash
ros2 topic list
ros2 topic echo /sportmodestate --once

# record while you drive with the app
ros2 bag record -s mcap -o bags/run1 /sportmodestate /lowstate
ros2 bag info bags/run1
ros2 bag play bags/run1
```

`ros2 bag info/play` and `ros2 topic echo` need the workspace sourced, otherwise you get "package 'unitree_go' not found". MCAP bags can also be opened in Foxglove or PlotJuggler without it.

## Notes

- The Go2's camera video is not a ROS topic, so it cannot be recorded this way.
- Lidar topics (e.g. `/utlidar/cloud`) make large bags.
- Lyrical's CycloneDDS is newer than the Go2's (0.10.2). If `ros2 topic list` shows no Unitree topics, build the same workspace on Humble.
