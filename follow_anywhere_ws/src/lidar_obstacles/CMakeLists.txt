# Build file of the lidar_obstacles package.
cmake_minimum_required(VERSION 3.16)
project(lidar_obstacles)

# C++17, built optimised (Release) unless told otherwise, with extra compiler warnings.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

# Libraries the code uses (ROS messages and Eigen for the Kalman filter matrices).
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(follow_anywhere_interfaces REQUIRED)
find_package(Eigen3 REQUIRED)

# Core: no ROS, so it can be unit tested on its own.
add_library(lidar_obstacles_core STATIC
  src/cloud_filter.cpp
  src/clustering.cpp
  src/cluster_tracker.cpp
  src/fusion.cpp)
target_include_directories(lidar_obstacles_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(lidar_obstacles_core PUBLIC Eigen3::Eigen)
set_target_properties(lidar_obstacles_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The ROS node: lidar_obstacles_node.cpp, linked with the core library and the ROS message libraries.
add_executable(lidar_obstacles_node src/lidar_obstacles_node.cpp)
target_link_libraries(lidar_obstacles_node lidar_obstacles_core rclcpp::rclcpp
  ${sensor_msgs_TARGETS} ${nav_msgs_TARGETS} ${geometry_msgs_TARGETS}
  ${std_msgs_TARGETS} ${visualization_msgs_TARGETS} ${follow_anywhere_interfaces_TARGETS})

# Install the node, and the config and launch folders so `ros2 launch lidar_obstacles ...` finds them.
install(TARGETS lidar_obstacles_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

# Unit tests (test/test_core.cpp) against the core library; run with `colcon test`.
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_core test/test_core.cpp)
  target_link_libraries(test_core lidar_obstacles_core)
endif()

ament_package()
