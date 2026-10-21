# Messages of every part of the follow-anywhere system, in one package.
# Rules: a message's name starts with the part that owns it (Perception..., Navigation...), CamelCase,
# no underscores (ROS 2 requires that), all files directly in msg/ (ROS 2 allows no subfolders).
# Each part only edits its own files. Add a new .msg / .srv to the list below.
cmake_minimum_required(VERSION 3.8)
project(follow_anywhere_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  # perception
  "msg/PerceptionFrameStats.msg"
  "msg/PerceptionLeaderState.msg"
  "msg/PerceptionObstacle.msg"
  "msg/PerceptionObstacleArray.msg"
  "msg/PerceptionTrackedPerson.msg"
  "msg/PerceptionTrackedPersonArray.msg"
  "srv/PerceptionCalibrateLeader.srv"
  # navigation: add here
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs
)

ament_package()
