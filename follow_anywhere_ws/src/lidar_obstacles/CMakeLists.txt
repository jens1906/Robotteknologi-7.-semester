cmake_minimum_required(VERSION 3.16)
project(lidar_obstacles)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(follow_anywhere_interfaces REQUIRED)

add_executable(lidar_obstacles_node src/lidar_obstacles_node.cpp)
target_link_libraries(lidar_obstacles_node rclcpp::rclcpp ${sensor_msgs_TARGETS} ${std_msgs_TARGETS}
  ${geometry_msgs_TARGETS} ${nav_msgs_TARGETS} ${follow_anywhere_interfaces_TARGETS})

install(TARGETS lidar_obstacles_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()
