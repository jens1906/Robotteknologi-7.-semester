"""Robot state topics of the physical Go2, from Gazebo joint states, IMUs and odometry.

Subscribes (ros_gz_bridge):
  /sim/joint_states  (sensor_msgs/JointState)
  /sim/imu           (sensor_msgs/Imu)        body IMU
  /sim/odometry      (nav_msgs/Odometry)      ground truth, twist in base_link
  /sim/utlidar/imu   (sensor_msgs/Imu)        L1 lidar IMU

Publishes, with the rates seen on the robot (bag go2_all_topics_6_10_11_28):
  /lowstate            unitree_go/LowState       ~450 Hz
  /sportmodestate      unitree_go/SportModeState ~260 Hz
  /lf/lowstate, /lf/sportmodestate               ~18 Hz
  /utlidar/robot_odom  nav_msgs/Odometry         ~130 Hz, odom -> base_link
  /utlidar/robot_pose  geometry_msgs/PoseStamped ~17 Hz, odom
  /utlidar/imu         sensor_msgs/Imu           as received (250 Hz), utlidar_imu

Fields the sim has no source for (battery, foot force, temperatures) are set to
the values the robot reported in that bag.

Joint order in LowState.motor_state[0..11] — matches Unitree SDK convention:
  0  FR_hip_joint    3  FL_hip_joint    6  RR_hip_joint    9  RL_hip_joint
  1  FR_thigh_joint  4  FL_thigh_joint  7  RR_thigh_joint  10 RL_thigh_joint
  2  FR_calf_joint   5  FL_calf_joint   8  RR_calf_joint   11 RL_calf_joint
"""

import time

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from unitree_go.msg import IMUState, LowState, SportModeState

from go2_robot_emulator.common import ROBOT_QOS, SIM_QOS, quat_to_rpy, run

JOINT_ORDER = [
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
]

# Values reported by the robot in bag go2_all_topics_6_10_11_28.
MOTOR_TEMPERATURES = [37, 35, 38, 36, 35, 36, 50, 37, 38, 51, 33, 37]
FOOT_FORCE = [89, 78, 86, 85]
IMU_TEMPERATURE = 79
SPORT_ERROR_CODE = 100


def fill_imu_state(state: IMUState, imu: Imu) -> None:
    q = imu.orientation
    # Unitree quaternion order: [w, x, y, z]
    state.quaternion[:] = (q.w, q.x, q.y, q.z)
    state.gyroscope[:] = (imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z)
    state.accelerometer[:] = (imu.linear_acceleration.x, imu.linear_acceleration.y,
                              imu.linear_acceleration.z)
    state.rpy[:] = quat_to_rpy(q.x, q.y, q.z, q.w)
    state.temperature = IMU_TEMPERATURE


class State(Node):

    def __init__(self):
        super().__init__('go2_state')
        self.declare_parameter('lowstate_rate', 450.0)
        self.declare_parameter('sportmodestate_rate', 260.0)
        self.declare_parameter('lf_rate', 18.0)
        self.declare_parameter('odom_rate', 130.0)
        self.declare_parameter('pose_rate', 17.0)

        self._joints: JointState | None = None
        self._imu: Imu | None = None
        self._odom: Odometry | None = None
        self._start = time.monotonic()

        self._lowstate = self._initial_lowstate()
        self._sportstate = SportModeState()
        self._sportstate.error_code = SPORT_ERROR_CODE

        self.create_subscription(JointState, '/sim/joint_states', self._joints_cb, SIM_QOS)
        self.create_subscription(Imu, '/sim/imu', self._imu_cb, SIM_QOS)
        self.create_subscription(Odometry, '/sim/odometry', self._odom_cb, SIM_QOS)
        self.create_subscription(Imu, '/sim/utlidar/imu', self._lidar_imu_cb, SIM_QOS)

        self.lowstate_pub = self.create_publisher(LowState, '/lowstate', ROBOT_QOS)
        self.sportstate_pub = self.create_publisher(SportModeState, '/sportmodestate', ROBOT_QOS)
        self.lf_lowstate_pub = self.create_publisher(LowState, '/lf/lowstate', ROBOT_QOS)
        self.lf_sportstate_pub = self.create_publisher(
            SportModeState, '/lf/sportmodestate', ROBOT_QOS)
        self.odom_pub = self.create_publisher(Odometry, '/utlidar/robot_odom', ROBOT_QOS)
        self.pose_pub = self.create_publisher(PoseStamped, '/utlidar/robot_pose', ROBOT_QOS)
        self.lidar_imu_pub = self.create_publisher(Imu, '/utlidar/imu', ROBOT_QOS)

        def rate(name):
            return 1.0 / self.get_parameter(name).value

        self.create_timer(rate('lowstate_rate'), self._publish_lowstate)
        self.create_timer(rate('sportmodestate_rate'), self._publish_sportstate)
        self.create_timer(rate('lf_rate'), self._publish_lf)
        self.create_timer(rate('odom_rate'), self._publish_odom)
        self.create_timer(rate('pose_rate'), self._publish_pose)

        self.get_logger().info('go2_state started — /lowstate, /sportmodestate, /utlidar/robot_odom')

    @staticmethod
    def _initial_lowstate() -> LowState:
        msg = LowState()
        msg.head[:] = (254, 239)
        msg.level_flag = 0
        for i, m in enumerate(msg.motor_state[:12]):
            m.mode = 1
            m.temperature = MOTOR_TEMPERATURES[i]
        msg.foot_force[:] = FOOT_FORCE
        msg.bms_state.version_high = 1
        msg.bms_state.version_low = 23
        msg.bms_state.status = 8
        msg.bms_state.soc = 91
        msg.bms_state.current = -1704
        msg.bms_state.cycle = 16
        msg.bms_state.bq_ntc[:] = (28, 27)
        msg.bms_state.mcu_ntc[:] = (31, 29)
        msg.bms_state.cell_vol[:8] = (4018, 4024, 4025, 4019, 4024, 4025, 4021, 4016)
        msg.bit_flag = 36
        msg.temperature_ntc1 = 44
        msg.temperature_ntc2 = 42
        msg.power_v = 31.14
        msg.power_a = 1.13
        return msg

    def _joints_cb(self, msg: JointState) -> None:
        self._joints = msg

    def _imu_cb(self, msg: Imu) -> None:
        self._imu = msg

    def _odom_cb(self, msg: Odometry) -> None:
        self._odom = msg

    def _lidar_imu_cb(self, msg: Imu) -> None:
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'utlidar_imu'
        self.lidar_imu_pub.publish(msg)

    def _update_lowstate(self) -> bool:
        if self._joints is None:
            return False
        msg = self._lowstate
        joints = self._joints
        index = {name: i for i, name in enumerate(joints.name)}
        for motor, name in zip(msg.motor_state, JOINT_ORDER):
            i = index.get(name)
            if i is None:
                continue
            if i < len(joints.position):
                motor.q = float(joints.position[i])
            if i < len(joints.velocity):
                motor.dq = float(joints.velocity[i])
            if i < len(joints.effort):
                motor.tau_est = float(joints.effort[i])
        if self._imu is not None:
            fill_imu_state(msg.imu_state, self._imu)
        # Milliseconds, like the robot's tick
        msg.tick = int((time.monotonic() - self._start) * 1000) & 0xFFFFFFFF
        return True

    def _update_sportstate(self) -> bool:
        if self._odom is None:
            return False
        msg = self._sportstate
        now = time.time_ns()
        msg.stamp.sec = now // 1_000_000_000
        msg.stamp.nanosec = now % 1_000_000_000

        pose = self._odom.pose.pose
        twist = self._odom.twist.twist
        msg.position[:] = (pose.position.x, pose.position.y, pose.position.z)
        msg.body_height = float(pose.position.z)
        # Body-frame velocity, like the robot (checked against d(position)/dt in
        # the bag); Gazebo's odometry twist is already in base_link.
        msg.velocity[:] = (twist.linear.x, twist.linear.y, twist.linear.z)
        msg.yaw_speed = float(twist.angular.z)
        if self._imu is not None:
            fill_imu_state(msg.imu_state, self._imu)
        return True

    def _publish_lowstate(self) -> None:
        if self._update_lowstate():
            self.lowstate_pub.publish(self._lowstate)

    def _publish_sportstate(self) -> None:
        if self._update_sportstate():
            self.sportstate_pub.publish(self._sportstate)

    def _publish_lf(self) -> None:
        if self._update_lowstate():
            self.lf_lowstate_pub.publish(self._lowstate)
        if self._update_sportstate():
            self.lf_sportstate_pub.publish(self._sportstate)

    def _publish_odom(self) -> None:
        if self._odom is None:
            return
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'
        msg.pose.pose = self._odom.pose.pose
        msg.twist.twist = self._odom.twist.twist
        self.odom_pub.publish(msg)

    def _publish_pose(self) -> None:
        if self._odom is None:
            return
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.pose = self._odom.pose.pose
        self.pose_pub.publish(msg)


def main():
    run(State)


if __name__ == '__main__':
    main()
