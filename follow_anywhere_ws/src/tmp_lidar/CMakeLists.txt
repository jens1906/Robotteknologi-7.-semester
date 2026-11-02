cmake_minimum_required(VERSION 3.16)
project(tmp_lidar)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(follow_anywhere_interfaces REQUIRED)

# Core: no ROS, so it can be unit tested on its own.
add_library(tmp_lidar_core STATIC
  src/cloud_filter.cpp
  src/clustering.cpp
  src/cluster_tracker.cpp
  src/fusion.cpp
  src/hit_grid.cpp
  src/scan_builder.cpp)
target_include_directories(tmp_lidar_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
set_target_properties(tmp_lidar_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(tmp_lidar_node src/tmp_lidar_node.cpp)
target_link_libraries(tmp_lidar_node tmp_lidar_core rclcpp::rclcpp
  ${sensor_msgs_TARGETS} ${nav_msgs_TARGETS} ${geometry_msgs_TARGETS}
  ${visualization_msgs_TARGETS} ${follow_anywhere_interfaces_TARGETS})

install(TARGETS tmp_lidar_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_core test/test_core.cpp)
  target_link_libraries(test_core tmp_lidar_core)
endif()

ament_package()
