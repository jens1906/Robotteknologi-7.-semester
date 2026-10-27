# perception_msgs

The interface between the perception stack (`vision`) and the navigation stack (`navigation`). Perception publishes everything it sees around the robot; navigation subscribes and does not care whether it comes from the real robot, Gazebo or `2dSim`'s `fake_perception`.

## Topics

| Topic | Type | Frame | Rate | Notes |
|---|---|---|---|---|
| `/perception/detections` | `perception_msgs/DetectionArray` | `base_link` | ~15 Hz | All objects seen in one frame. |
| `/perception/scan` | `sensor_msgs/LaserScan` | `base_link` | ~15 Hz | 360° range scan, for free-space checks between detections. |

## DetectionArray

| Field | Type | Meaning |
|---|---|---|
| `header.stamp` | `time` | When the robot was at the pose the detections are relative to. Use it to look up `odom -> base_link` if you need the detections in `odom`. |
| `header.frame_id` | `string` | Always `base_link`. |
| `detections` | `Detection[]` | One per object. Empty means nothing seen, not "no data". |

## Detection

Polar coordinates around `base_link` (x forward, y left, bearing counter-clockwise from +x, in rad).

| Field | Unit | Meaning |
|---|---|---|
| `id` | | Track id, stable while the object is tracked. A new id means a new track, not necessarily a new object. |
| `class_id` | `CLASS_*` | What the object is (see below). |
| `confidence` | 0..1 | Confidence in `class_id`. 0 for `CLASS_UNKNOWN`. |
| `sensors` | `SENSOR_*` bitmask | Which sensors saw it this frame. |
| `range`, `bearing` | m, rad | Centre of the object. |
| `range_std`, `bearing_std` | m, rad | 1σ of the centre. |
| `nearest_range`, `nearest_bearing` | m, rad | Closest point of the object's outline. Use this for clearance: for walls and big boxes the centre is far from the part you can hit. |
| `radius` | m | Radius of a circle around the centre containing the footprint. |
| `velocity` | m/s | Velocity over the ground (not relative to the robot), expressed in `base_link`. `z` unused. |
| `velocity_std` | m/s | 1σ per axis. |
| `moving` | bool | `|velocity|` above the stack's motion threshold. |

### Constants

| Class | Value | Meaning |
|---|---|---|
| `CLASS_UNKNOWN` | 0 | Seen by the lidar only: geometry, no class. |
| `CLASS_OBJECT` | 1 | Seen by the camera, not a person. |
| `CLASS_PERSON` | 2 | A person who is not the leader. |
| `CLASS_LEADER` | 3 | The person the robot follows. At most one per frame. |

| Sensor | Bit | Field of view (robot / `2dSim` defaults) |
|---|---|---|
| `SENSOR_LIDAR` | 1 | 360°, 0.1–15 m. Geometry only. |
| `SENSOR_CAMERA` | 2 | ±45° in front, 0.3–8 m. Classifies. |

Only the camera classifies, so anything behind or beside the robot is `CLASS_UNKNOWN`. The leader can drop to `CLASS_PERSON` or `CLASS_UNKNOWN` and back while keeping its `id`: follow the id, not just the class.

## Example

```python
import math
from perception_msgs.msg import Detection, DetectionArray

def on_detections(msg: DetectionArray):
    for d in msg.detections:
        x = d.range * math.cos(d.bearing)          # base_link
        y = d.range * math.sin(d.bearing)
        clearance = d.nearest_range
        if d.class_id == Detection.CLASS_LEADER:
            ...

node.create_subscription(DetectionArray, '/perception/detections', on_detections, 10)
```

## Producers

- `vision` on the robot (to do).
- `2dSim`'s `fake_perception`: made from the sim's ground truth with configurable noise, sensor fields, dropouts and occlusion. See [2dSim/USAGE.txt](../../../2dSim/USAGE.txt).
