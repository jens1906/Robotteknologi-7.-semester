cmake_minimum_required(VERSION 3.16)
project(web_viewer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(follow_anywhere_interfaces REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rosidl_typesupport_introspection_cpp REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Threads REQUIRED)

add_executable(web_viewer
  src/main.cpp
  src/server.cpp
  src/msg_json.cpp
  src/encoders.cpp)
target_include_directories(web_viewer PRIVATE ${OpenCV_INCLUDE_DIRS})
target_compile_definitions(web_viewer PRIVATE
  WEB_VIEWER_BAG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../RosBag")
target_link_libraries(web_viewer
  rclcpp::rclcpp
  ${sensor_msgs_TARGETS}
  ${std_msgs_TARGETS}
  ${std_srvs_TARGETS}
  ${follow_anywhere_interfaces_TARGETS}
  ament_index_cpp::ament_index_cpp
  ${rosidl_typesupport_introspection_cpp_TARGETS}
  Boost::headers
  ${OpenCV_LIBS}
  Threads::Threads)

install(TARGETS web_viewer DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY www DESTINATION share/${PROJECT_NAME})

ament_package()
