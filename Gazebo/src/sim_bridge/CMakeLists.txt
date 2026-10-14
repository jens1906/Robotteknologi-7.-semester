cmake_minimum_required(VERSION 3.8)
project(sim_bridge)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  sim_bridge/sport_request_bridge.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME sport_request_bridge
)

ament_package()
