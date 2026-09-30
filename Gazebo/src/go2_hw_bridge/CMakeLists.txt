cmake_minimum_required(VERSION 3.8)
project(go2_hw_bridge)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  go2_hw_bridge/lowstate_publisher.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME lowstate_publisher
)

ament_package()
