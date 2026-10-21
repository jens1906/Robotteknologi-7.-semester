# One obstacle. Frame: base_link (x forward, y left, z up). Units: m, m/s.
# Static obstacles come from the lidar (or the depth image as fallback), dynamic ones are tracked
# people (and other moving classes) from the camera.

uint8 STATIC = 0
uint8 DYNAMIC = 1
uint8 UNKNOWN = 2      # e.g. static/dynamic cannot be told while the robot moves without odometry

uint8 SOURCE_LIDAR = 0
uint8 SOURCE_CAMERA = 1

int32 id                              # tracker id for dynamic obstacles, -1 for static
uint8 kind                            # STATIC, DYNAMIC or UNKNOWN
uint8 source                          # SOURCE_LIDAR or SOURCE_CAMERA
string class_name                     # e.g. person; empty for lidar obstacles

geometry_msgs/Point position          # centre
geometry_msgs/Vector3 size            # extent in x, y, z
float32 position_uncertainty          # m, 1 sigma

# NaN while velocity_valid is false
geometry_msgs/Vector3 velocity
bool velocity_valid
bool velocity_compensated             # true: the robot's own motion is removed (odometry used)
