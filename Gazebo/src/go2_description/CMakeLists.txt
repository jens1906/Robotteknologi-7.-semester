cmake_minimum_required(VERSION 3.8)
project(go2_description)

find_package(ament_cmake REQUIRED)

# Path to the upstream go2_description directory.
# Set by Dockerfile: -DUPSTREAM_PATH=/upstream/go2_description
# Default falls back to relative path for local builds inside Gazebo/
if(NOT DEFINED UPSTREAM_PATH)
  set(UPSTREAM_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../upstream/unitree_ros/robots/go2_description")
endif()

# Install the upstream URDF (used by go2_gazebo.urdf.xacro via xacro:include)
install(
  FILES "${UPSTREAM_PATH}/urdf/go2_description.urdf"
  DESTINATION share/${PROJECT_NAME}/urdf/upstream
)

# Install upstream dae meshes — URDF references package://go2_description/dae/*.dae
install(
  DIRECTORY "${UPSTREAM_PATH}/dae/"
  DESTINATION share/${PROJECT_NAME}/dae
)

# Install our Gazebo+CHAMP xacro
install(
  DIRECTORY urdf
  DESTINATION share/${PROJECT_NAME}
)

# Install config (gait.yaml, ros2_controllers.yaml)
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
