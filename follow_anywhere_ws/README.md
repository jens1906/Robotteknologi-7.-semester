# follow_anywhere_ws

ROS2 (Lyrical) workspace for the follow-anywhere project on the physical Unitree Go2.

| Package | Purpose |
|---|---|
| `unitree_api` | Go2 request/response messages (`/api/sport/request`, ...). Backbone. |
| `unitree_go` | Go2 state/command messages (`/sportmodestate`, `/lowstate`, ...). Backbone. |
| `manual_controller` | WASD/arrow-key manual control. |
| `kpr` | KPR person re-identification, inference only (library, no nodes). Licence: see `src/kpr/LICENSE`. |
| `vision_pre` | People tracking + leader identification (KPR, OSNet as an opt-in baseline). `vision_pre.view` shows it on a video. |
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

## Vision: watch the leader tracking

Needs the KPR weights `kpr_occ_pt_IN_*.pth.tar` in `models/` (download from the Google Drive folder in the
[KPR README](https://github.com/VlSomers/keypoint_promptable_reidentification#download-the-pre-trained-models);
not in git) and `timm` in the venv. Test videos: `scripts/fetch_mot17.py` fills `data/`.

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select kpr vision_pre     # once; plain colcon, not the venv python
source install/setup.bash
ros2 run vision_pre view data/mot17_09
```

Without building: `PYTHONPATH=src/vision_pre:src/kpr .venv/bin/python -m vision_pre.view data/mot17_09`.

Two panels, OSNet (left) and KPR (right), same video and same people. **Click a person** to make them the
leader in both panels (or press `c` for the person nearest the centre); `r` forgets the leader, space pauses,
`q` quits. Green box = identified leader, the number under a box is the distance to the leader's stored
views (lower = more alike).

- **The leader is saved** to `~/.ros/follow_anywhere/leader_<embedder>.npz` when enrolment finishes and loaded
  at the next start (the panel then says SEARCHING until they are found). `--fresh` ignores the file, `r`
  only forgets it for this session. Another person's views are never mixed in: a new click replaces the file.
- **`--loop`** plays the clip again and again from `--start`: each lap restarts the tracker (new ids) and the
  leader is searched again with the views it has, so every lap tests re-identification.
- Other options: `--embedders kpr` (one panel), `--realtime` (drop frames like the robot would),
  `--out file.mp4`, `--no-show --click FRAME,X,Y` (headless), `--start N`, `--imgsz 1280` (finds more
  far-away people). It uses the GPU when there is one and warns when it falls back to the CPU. The Qt
  "Cannot find font directory" lines are harmless.

`view` is not a ROS node (it does not use rclpy). Installed scripts run with the system python, which has no
`timm` or `onnxruntime`; `vision_pre/__init__.py` appends the `.venv` site-packages at the END of `sys.path`, so
only those come from the venv and the system numpy / OpenCV keep priority (the venv has numpy 2 and OpenCV 5,
which break `cv_bridge` in real ROS nodes). Do not build with `.venv/bin/python -m colcon`.
Tests: `PYTHONPATH=src/vision_pre:src/kpr .venv/bin/python -m pytest src/vision_pre/test/test_leader.py src/kpr/test`.
