cmake_minimum_required(VERSION 3.16)
project(navigation_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(perception_msgs REQUIRED)
find_package(unitree_api REQUIRED)
find_package(Eigen3 REQUIRED)

# OSQP: use an installed one if there is one, otherwise build v1.0.0 statically.
find_package(osqp QUIET)
if(NOT osqp_FOUND)
  include(FetchContent)
  set(OSQP_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
  set(OSQP_BUILD_DEMO_EXE OFF CACHE BOOL "" FORCE)
  set(OSQP_BUILD_UNITTESTS OFF CACHE BOOL "" FORCE)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  FetchContent_Declare(osqp
    GIT_REPOSITORY https://github.com/osqp/osqp.git
    GIT_TAG v1.0.0
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(osqp)
endif()
if(TARGET osqp::osqpstatic)
  set(OSQP_TARGET osqp::osqpstatic)
elseif(TARGET osqpstatic)
  set(OSQP_TARGET osqpstatic)
else()
  set(OSQP_TARGET osqp::osqp)
endif()

# The MPC itself, without ROS, so it can be tested on its own.
add_library(follow_mpc STATIC src/mpc.cpp)
target_include_directories(follow_mpc PUBLIC src)
target_link_libraries(follow_mpc PUBLIC Eigen3::Eigen ${OSQP_TARGET})

add_executable(follow_leader_cpp src/follow_controller.cpp)
target_link_libraries(follow_leader_cpp follow_mpc rclcpp::rclcpp
  ${perception_msgs_TARGETS} ${unitree_api_TARGETS})

install(TARGETS follow_leader_cpp DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_follow_controller test/test_follow_controller.cpp)
  target_link_libraries(test_follow_controller follow_mpc)
endif()

ament_package()
