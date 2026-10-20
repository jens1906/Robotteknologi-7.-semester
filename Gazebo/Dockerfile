FROM ros:lyrical-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# ros-lyrical-ros-gz pulls Gazebo Jetty + bridge + spawn tools via ROS repos.
# No separate OSRF Gazebo apt source needed.
RUN apt-get update && apt-get install -y \
    ros-lyrical-ros-gz \
    ros-lyrical-ros2-control \
    ros-lyrical-gz-ros2-control \
    ros-lyrical-xacro \
    ros-lyrical-robot-state-publisher \
    ros-lyrical-joint-state-publisher \
    ros-lyrical-joint-state-broadcaster \
    ros-lyrical-joint-trajectory-controller \
    ros-lyrical-nav-msgs \
    ros-lyrical-nav2-bringup \
    ros-lyrical-robot-localization \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-av \
    python3-numpy \
    python3-yaml \
    && rm -rf /var/lib/apt/lists/*

# Let EGL (headless rendering) use the NVIDIA driver that nvidia-container-toolkit
# mounts with docker-compose.gpu.yml; the toolkit adds the library but not this
# glvnd vendor file. Without an NVIDIA GPU, glvnd skips it and uses Mesa.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /ros2_ws

# ---- unitree_go / unitree_api messages + shared Go2 config ----
# Build context is the repo root (see docker-compose.yml), so the sim builds
# the same message definitions and sensor mounts as follow_anywhere_ws, which
# match what the real robot sends.
COPY follow_anywhere_ws/src/unitree_go/  src/unitree_go/
COPY follow_anywhere_ws/src/unitree_api/ src/unitree_api/
COPY follow_anywhere_ws/src/go2_bringup/ src/go2_bringup/

# ---- CHAMP gait controller (not in Lyrical binary repos — build from source) ----
# Only copy the packages we actually use: champ (core lib), champ_base (controller),
# champ_msgs (message types). Skip champ_description/navigation/bringup/config/gazebo
# as they pull in unavailable deps (velodyne_gazebo_plugins, ecl_threads, etc.).
COPY Gazebo/upstream/champ/champ/      src/champ/
COPY Gazebo/upstream/champ/champ_base/ src/champ_base/
COPY Gazebo/upstream/champ/champ_msgs/ src/champ_msgs/

# Port CHAMP to Lyrical:
# - ament_target_dependencies() was removed from ament_cmake, but champ_base
#   still uses it. Inject a minimal stand-in (only defined when the real one
#   is missing) right after find_package(ament_cmake).
# - The deprecated .h headers of tf2 / tf2_ros / message_filters / urdf were
#   removed; rewrite those includes to the .hpp versions.
RUN python3 - <<'PYEOF'
import pathlib, re

include_re = re.compile(
    r'(#include\s*[<"](?:tf2|tf2_ros|message_filters|urdf)/[^>"]*)\.h([>"])')
for pkg in ('champ', 'champ_base'):
    for src in pathlib.Path(f'src/{pkg}').rglob('*'):
        if src.suffix in ('.h', '.hpp', '.cpp') and src.is_file():
            text = src.read_text()
            new = include_re.sub(r'\1.hpp\2', text)
            if new != text:
                src.write_text(new)
print('CHAMP includes rewritten to .hpp')

# message_filters::Subscriber::subscribe() now requires an explicit QoS.
# rclcpp::QoS(10) equals the old default (reliable, keep last 10).
src = pathlib.Path('src/champ_base/src/state_estimation.cpp')
text = src.read_text()
new = re.sub(r'(_subscriber_\.subscribe\([^;]*?"\w+")\);',
             r'\1, rclcpp::QoS(10));', text)
assert new != text
src.write_text(new)
print('state_estimation.cpp subscribe() calls patched')

shim = '''
if(NOT COMMAND ament_target_dependencies)
  function(ament_target_dependencies target)
    foreach(dep ${ARGN})
      if(NOT ${dep}_FOUND)
        message(FATAL_ERROR "ament_target_dependencies(): '${dep}' not found")
      endif()
      if(${dep}_TARGETS)
        target_link_libraries(${target} ${${dep}_TARGETS})
      else()
        target_include_directories(${target} PUBLIC ${${dep}_INCLUDE_DIRS})
        target_link_libraries(${target} ${${dep}_LIBRARIES})
      endif()
      target_compile_definitions(${target} PUBLIC ${${dep}_DEFINITIONS})
    endforeach()
  endfunction()
endif()
'''

cmake = pathlib.Path('src/champ_base/CMakeLists.txt')
text = cmake.read_text()
anchor = 'find_package(ament_cmake REQUIRED)'
assert anchor in text
cmake.write_text(text.replace(anchor, anchor + '\n' + shim, 1))
print('champ_base CMakeLists.txt patched successfully')
PYEOF

# ---- Our ROS2 packages ----
COPY Gazebo/upstream/unitree_ros/robots/go2_description/ /upstream/go2_description/
COPY Gazebo/src/go2_description/    src/go2_description/
COPY Gazebo/src/go2_sim/            src/go2_sim/
COPY Gazebo/src/go2_robot_emulator/ src/go2_robot_emulator/
COPY Gazebo/src/go2_scenarios/      src/go2_scenarios/
COPY Gazebo/tools/                  /ros2_ws/tools/

# ---- Dependencies ----
# apt-get update is needed again because rm -rf /var/lib/apt/lists/* cleared the cache above.
# --skip-keys: skip packages not available in Lyrical repos or handled above manually.
RUN apt-get update && \
    rosdep init || true && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y \
        --skip-keys "velodyne_gazebo_plugins ecl_threads" && \
    rm -rf /var/lib/apt/lists/*

# ---- Build ----
RUN . /opt/ros/lyrical/setup.sh && \
    colcon build --symlink-install \
        --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DUPSTREAM_PATH=/upstream/go2_description

# ---- Runtime environment ----
RUN echo ". /opt/ros/lyrical/setup.sh" >> /root/.bashrc && \
    echo ". /ros2_ws/install/setup.sh" >> /root/.bashrc

CMD ["/bin/bash", "-c", \
     ". /opt/ros/lyrical/setup.sh && \
      . /ros2_ws/install/setup.sh && \
      ros2 launch go2_sim go2_sim.launch.py world:=${GO2_WORLD:-empty}"]
