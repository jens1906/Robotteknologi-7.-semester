#!/usr/bin/env python3
"""
Converts Gazebo joint states and IMU data to unitree_go/msg/LowState.
Publishes /lowstate matching the topic format of the physical Unitree Go2.

Subscribes:
  /joint_states  (sensor_msgs/msg/JointState)  from ros_gz_bridge
  /imu/data      (sensor_msgs/msg/Imu)          from ros_gz_bridge

Publishes:
  /lowstate  (unitree_go/msg/LowState)

Joint order in LowState.motor_state[0..11] — matches Unitree SDK convention:
  0  FR_hip_joint    3  FL_hip_joint    6  RR_hip_joint    9  RL_hip_joint
  1  FR_thigh_joint  4  FL_thigh_joint  7  RR_thigh_joint  10 RL_thigh_joint
  2  FR_calf_joint   5  FL_calf_joint   8  RR_calf_joint   11 RL_calf_joint
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import JointState, Imu
from unitree_go.msg import LowState, LowCmd

JOINT_ORDER = [
    'FR_hip_joint',    # motor_state[0]
    'FR_thigh_joint',  # motor_state[1]
    'FR_calf_joint',   # motor_state[2]
    'FL_hip_joint',    # motor_state[3]
    'FL_thigh_joint',  # motor_state[4]
    'FL_calf_joint',   # motor_state[5]
    'RR_hip_joint',    # motor_state[6]
    'RR_thigh_joint',  # motor_state[7]
    'RR_calf_joint',   # motor_state[8]
    'RL_hip_joint',    # motor_state[9]
    'RL_thigh_joint',  # motor_state[10]
    'RL_calf_joint',   # motor_state[11]
]

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class LowStatePublisher(Node):

    def __init__(self):
        super().__init__('go2_hw_bridge')

        self._imu_msg: Imu | None = None

        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_callback, SENSOR_QOS)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self._imu_callback, SENSOR_QOS)

        self.lowstate_pub = self.create_publisher(LowState, '/lowstate', 10)

        # Stub: subscribe to /lowcmd for future joint position/torque control.
        # When a Gazebo joint effort/position controller is wired up, convert
        # LowCmd.motor_cmd[i].q/dq/kp/kd/tau to controller commands here.
        self.lowcmd_sub = self.create_subscription(
            LowCmd, '/lowcmd', self._lowcmd_callback, 10)

        self.get_logger().info('go2_hw_bridge started — publishing /lowstate')

    def _imu_callback(self, msg: Imu) -> None:
        self._imu_msg = msg

    def _joint_callback(self, msg: JointState) -> None:
        lowstate = LowState()

        name_to_idx = {name: i for i, name in enumerate(msg.name)}

        for motor_idx, joint_name in enumerate(JOINT_ORDER):
            if joint_name not in name_to_idx:
                self.get_logger().debug(
                    f'Joint {joint_name} not in /joint_states',
                    throttle_duration_sec=10.0,
                )
                continue
            src = name_to_idx[joint_name]
            m = lowstate.motor_state[motor_idx]
            m.mode = 1
            if src < len(msg.position):
                m.q = float(msg.position[src])
            if src < len(msg.velocity):
                m.dq = float(msg.velocity[src])
            if src < len(msg.effort):
                m.tau_est = float(msg.effort[src])

        if self._imu_msg is not None:
            imu = self._imu_msg
            s = lowstate.imu_state
            # Unitree quaternion order: [w, x, y, z]
            s.quaternion[0] = imu.orientation.w
            s.quaternion[1] = imu.orientation.x
            s.quaternion[2] = imu.orientation.y
            s.quaternion[3] = imu.orientation.z
            s.gyroscope[0] = imu.angular_velocity.x
            s.gyroscope[1] = imu.angular_velocity.y
            s.gyroscope[2] = imu.angular_velocity.z
            s.accelerometer[0] = imu.linear_acceleration.x
            s.accelerometer[1] = imu.linear_acceleration.y
            s.accelerometer[2] = imu.linear_acceleration.z

        stamp_ns = (msg.header.stamp.sec * 1_000_000_000
                    + msg.header.stamp.nanosec)
        lowstate.tick = stamp_ns & 0xFFFFFFFF

        self.lowstate_pub.publish(lowstate)

    def _lowcmd_callback(self, msg: LowCmd) -> None:
        # Future: convert LowCmd motor commands to Gazebo joint position targets.
        # For now: stub — command received but not applied.
        self.get_logger().debug('Received /lowcmd (stub — not applied yet)',
                                throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = LowStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
