FROM ros:lyrical-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# No Gazebo: the 2D sim is plain Python. pyqtgraph + Qt for the viewer window.
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-numpy \
    python3-yaml \
    python3-pyqtgraph \
    python3-pyqt6 \
    python3-pytest \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ros2_ws

# ---- Messages + shared Go2 config from follow_anywhere_ws ----
# Build context is the repo root (see docker-compose.yml), so the sim builds
# the same message definitions the robot code uses.
COPY follow_anywhere_ws/src/unitree_go/      src/unitree_go/
COPY follow_anywhere_ws/src/unitree_api/     src/unitree_api/
COPY follow_anywhere_ws/src/go2_bringup/     src/go2_bringup/
COPY follow_anywhere_ws/src/perception_msgs/ src/perception_msgs/

# ---- Worlds: the same scenes as the Gazebo sim ----
COPY Gazebo/src/go2_scenarios/ src/go2_scenarios/

# ---- The 2D sim ----
COPY 2dSim/src/go2_sim2d/ src/go2_sim2d/

# ---- Dependencies ----
RUN apt-get update && \
    rosdep init || true && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    rm -rf /var/lib/apt/lists/*

# ---- Build ----
RUN . /opt/ros/lyrical/setup.sh && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# ---- Runtime environment ----
RUN echo ". /opt/ros/lyrical/setup.sh" >> /root/.bashrc && \
    echo ". /ros2_ws/install/setup.sh" >> /root/.bashrc

CMD ["/bin/bash", "-c", \
     ". /opt/ros/lyrical/setup.sh && \
      . /ros2_ws/install/setup.sh && \
      ros2 launch go2_sim2d sim2d.launch.py world:=${GO2_WORLD:-open_field} \
        viewer:=${SIM2D_VIEWER:-true} noise_scale:=${SIM2D_NOISE:-1.0}"]
