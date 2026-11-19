cmake_minimum_required(VERSION 3.16)
project(leader_search)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

find_package(ament_cmake REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)
find_package(yaml-cpp REQUIRED)
if(TARGET yaml-cpp::yaml-cpp)
  set(YAML_TARGET yaml-cpp::yaml-cpp)
else()
  set(YAML_TARGET yaml-cpp)
endif()

# Core: plain C++17, no ROS (spec: Code layout).
add_library(leader_search_core STATIC src/search_filter.cpp)
target_include_directories(leader_search_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(leader_search_core PUBLIC ${YAML_TARGET})
set_target_properties(leader_search_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ROS node (thin wrapper), as a component and an executable.
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(follow_anywhere_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
add_library(leader_search_component SHARED src/leader_search_node.cpp)
target_link_libraries(leader_search_component leader_search_core rclcpp::rclcpp
  rclcpp_components::component ament_index_cpp::ament_index_cpp
  ${follow_anywhere_interfaces_TARGETS} ${geometry_msgs_TARGETS} ${nav_msgs_TARGETS}
  ${sensor_msgs_TARGETS} ${std_msgs_TARGETS} ${visualization_msgs_TARGETS})
rclcpp_components_register_node(leader_search_component
  PLUGIN "leader_search::LeaderSearchNode" EXECUTABLE leader_search_node)
install(TARGETS leader_search_component
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)

# p_d / cam_range calibration from a rosbag (spec: Calibrating p_d and range).
find_package(rosbag2_cpp REQUIRED)
add_executable(calibrate_pd src/calibrate_pd.cpp)
target_link_libraries(calibrate_pd rclcpp::rclcpp rosbag2_cpp::rosbag2_cpp
  ${follow_anywhere_interfaces_TARGETS} ${nav_msgs_TARGETS} ${sensor_msgs_TARGETS})
install(TARGETS calibrate_pd DESTINATION lib/${PROJECT_NAME})

add_executable(bench_step test/bench_step.cpp)
target_link_libraries(bench_step leader_search_core)
install(TARGETS bench_step DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_search_filter test/test_search_filter.cpp)
  target_link_libraries(test_search_filter leader_search_core)
  target_compile_definitions(test_search_filter PRIVATE
    LEADER_SEARCH_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/config/leader_search.yaml")
endif()

install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_package()
