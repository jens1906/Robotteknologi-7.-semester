FROM osrf/ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# ros-jazzy-ros-gz pulls Gazebo Harmonic + bridge + spawn tools via ROS repos.
# No separate OSRF Gazebo apt source needed.
RUN apt-get update && apt-get install -y \
    ros-jazzy-ros-gz \
    ros-jazzy-ros2-control \
    ros-jazzy-gz-ros2-control \
    ros-jazzy-xacro \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-joint-state-broadcaster \
    ros-jazzy-joint-trajectory-controller \
    ros-jazzy-nav-msgs \
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
COPY upstream/champ/ src/champ_upstream/
# CHAMP ros2 branch contains subdirs: champ, champ_base, champ_bringup,
# champ_config, champ_description, champ_gazebo, champ_msgs, champ_navigation.
# We copy all into src/ so colcon finds them.
RUN cp -r src/champ_upstream/champ             src/ && \
    cp -r src/champ_upstream/champ_base        src/ && \
    cp -r src/champ_upstream/champ_bringup     src/ && \
    cp -r src/champ_upstream/champ_config      src/ && \
    cp -r src/champ_upstream/champ_description src/ && \
    cp -r src/champ_upstream/champ_msgs        src/ && \
    cp -r src/champ_upstream/champ_navigation  src/ && \
    rm -rf src/champ_upstream

# ---- Our ROS2 packages ----
COPY upstream/unitree_ros/robots/go2_description/ /upstream/go2_description/
COPY src/go2_description/ src/go2_description/
COPY src/go2_sim/         src/go2_sim/
COPY src/go2_hw_bridge/   src/go2_hw_bridge/

# ---- Dependencies ----
RUN rosdep init || true && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y

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
