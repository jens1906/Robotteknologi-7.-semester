FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# ros-jazzy-ros-gz pulls Gazebo Harmonic + bridge + spawn tools via ROS repos.
# No separate OSRF Gazebo apt source needed.
RUN apt-get update && apt-get install -y \
    ros-jazzy-ros-gz \
    ros-jazzy-ros2-control \
    ros-jazzy-gz-ros2-control \
    ros-jazzy-xacro \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-joint-state-publisher \
    ros-jazzy-joint-state-broadcaster \
    ros-jazzy-joint-trajectory-controller \
    ros-jazzy-nav-msgs \
    ros-jazzy-nav2-bringup \
    ros-jazzy-robot-localization \
    python3-colcon-common-extensions \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ros2_ws

# ---- unitree_go messages ----
# Copy only the message package from the unitree_ros2 submodule.
# Path inside the submodule: cyclonedds_ws/src/unitree/unitree_go/
COPY upstream/unitree_ros2/cyclonedds_ws/src/unitree/unitree_go/ src/unitree_go/

# Patch unitree_go/CMakeLists.txt to make rosidl_generator_dds_idl optional.
# The DDS IDL generation requires RTI Connext which is not available here.
# We guard it so the standard rosidl_generate_interfaces() call still works.
RUN python3 - <<'PYEOF'
import re, pathlib

cmake = pathlib.Path('src/unitree_go/CMakeLists.txt')
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
print('unitree_go CMakeLists.txt patched successfully')
PYEOF

# ---- CHAMP gait controller (not in Jazzy binary repos — build from source) ----
# Only copy the packages we actually use: champ (core lib), champ_base (controller),
# champ_msgs (message types). Skip champ_description/navigation/bringup/config/gazebo
# as they pull in unavailable deps (velodyne_gazebo_plugins, ecl_threads, etc.).
COPY upstream/champ/champ/      src/champ/
COPY upstream/champ/champ_base/ src/champ_base/
COPY upstream/champ/champ_msgs/ src/champ_msgs/

# ---- Our ROS2 packages ----
COPY upstream/unitree_ros/robots/go2_description/ /upstream/go2_description/
COPY src/go2_description/ src/go2_description/
COPY src/go2_sim/         src/go2_sim/
COPY src/go2_hw_bridge/   src/go2_hw_bridge/

# ---- Dependencies ----
# apt-get update is needed again because rm -rf /var/lib/apt/lists/* cleared the cache above.
# --skip-keys: skip packages not available in Jazzy repos or handled above manually.
RUN apt-get update && \
    rosdep init || true && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y \
        --skip-keys "velodyne_gazebo_plugins ecl_threads" && \
    rm -rf /var/lib/apt/lists/*

# ---- Build ----
RUN . /opt/ros/jazzy/setup.sh && \
    colcon build --symlink-install \
        --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DUPSTREAM_PATH=/upstream/go2_description

# ---- Runtime environment ----
RUN echo ". /opt/ros/jazzy/setup.sh" >> /root/.bashrc && \
    echo ". /ros2_ws/install/setup.sh" >> /root/.bashrc

CMD ["/bin/bash", "-c", \
     ". /opt/ros/jazzy/setup.sh && \
      . /ros2_ws/install/setup.sh && \
      ros2 launch go2_sim go2_sim.launch.py"]
