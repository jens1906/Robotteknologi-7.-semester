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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ros2_ws

# ---- unitree_go / unitree_api messages ----
# Copy only the message packages from the vendored unitree_ros2 repo.
# Path inside the vendored repo: cyclonedds_ws/src/unitree/
# unitree_api is needed by sim_bridge (/api/sport/request).
COPY upstream/unitree_ros2/cyclonedds_ws/src/unitree/unitree_go/  src/unitree_go/
COPY upstream/unitree_ros2/cyclonedds_ws/src/unitree/unitree_api/ src/unitree_api/

# Patch both CMakeLists.txt to make rosidl_generator_dds_idl optional.
# The DDS IDL generation requires RTI Connext which is not available here.
# We guard it so the standard rosidl_generate_interfaces() call still works.
RUN python3 - <<'PYEOF'
import re, pathlib

for pkg in ('unitree_go', 'unitree_api'):
    cmake = pathlib.Path(f'src/{pkg}/CMakeLists.txt')
    text = cmake.read_text()

    # Make the find_package QUIET (won't error if not found)
    text = text.replace(
        'find_package(rosidl_generator_dds_idl REQUIRED)',
        'find_package(rosidl_generator_dds_idl QUIET)'
    )

    # Wrap the DDS IDL generation block in a found-guard
    dds_block = re.search(
        r'rosidl_generate_dds_interfaces\(.*?\)',
        text,
        re.DOTALL,
    )
    if dds_block:
        old = dds_block.group(0)
        new = 'if(rosidl_generator_dds_idl_FOUND)\n' + old + '\nendif()'
        text = text.replace(old, new)

    # Wrap the add_dependencies DDS block
    add_dep = re.search(
        r'add_dependencies\(\s*\$\{PROJECT_NAME\}\s*\$\{PROJECT_NAME\}__dds_connext_idl\s*\)',
        text,
        re.DOTALL,
    )
    if add_dep:
        old = add_dep.group(0)
        new = 'if(rosidl_generator_dds_idl_FOUND)\n' + old + '\nendif()'
        text = text.replace(old, new)

    cmake.write_text(text)
    print(f'{pkg} CMakeLists.txt patched successfully')
PYEOF

# ---- CHAMP gait controller (not in Lyrical binary repos — build from source) ----
# Only copy the packages we actually use: champ (core lib), champ_base (controller),
# champ_msgs (message types). Skip champ_description/navigation/bringup/config/gazebo
# as they pull in unavailable deps (velodyne_gazebo_plugins, ecl_threads, etc.).
COPY upstream/champ/champ/      src/champ/
COPY upstream/champ/champ_base/ src/champ_base/
COPY upstream/champ/champ_msgs/ src/champ_msgs/

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
COPY upstream/unitree_ros/robots/go2_description/ /upstream/go2_description/
COPY src/go2_description/ src/go2_description/
COPY src/go2_sim/         src/go2_sim/
COPY src/go2_hw_bridge/   src/go2_hw_bridge/
COPY src/sim_bridge/      src/sim_bridge/

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
      ros2 launch go2_sim go2_sim.launch.py"]
