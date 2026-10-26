# One object seen by the perception stack, in local polar coordinates around
# base_link (x forward, y left). Published in a DetectionArray.

uint8 CLASS_UNKNOWN=0   # seen by the lidar only: geometry, no class
uint8 CLASS_OBJECT=1    # seen by the camera, not a person
uint8 CLASS_PERSON=2
uint8 CLASS_LEADER=3

uint8 SENSOR_LIDAR=1
uint8 SENSOR_CAMERA=2

uint32 id               # track id, stable while the object is tracked
uint8 class_id          # CLASS_*
float32 confidence      # of class_id, 0..1
uint8 sensors           # bitmask of SENSOR_* that saw it this frame

# Centre of the object.
float32 range           # m
float32 bearing         # rad, counter-clockwise from +x of base_link
float32 range_std       # m, 1 sigma
float32 bearing_std     # rad, 1 sigma

# Closest point of the object's outline (walls and large boxes have their
# centre far from the part that matters).
float32 nearest_range
float32 nearest_bearing

float32 radius          # m, radius of a circle around the centre containing the footprint

# Estimated velocity over the ground (not relative to the robot), expressed
# in base_link. z is unused.
geometry_msgs/Vector3 velocity
float32 velocity_std    # m/s, 1 sigma per axis
bool moving             # |velocity| above the stack's motion threshold
