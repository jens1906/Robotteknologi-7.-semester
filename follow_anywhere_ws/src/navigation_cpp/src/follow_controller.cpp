#include <chrono>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>

#include <Eigen/Dense>

#include "mpc.hpp"
#include "perception_msgs/msg/detection_array.hpp"
#include "rclcpp/rclcpp.hpp"
#include "unitree_api/msg/request.hpp"

using navigation_cpp::Leader;
using navigation_cpp::Obstacle;
using perception_msgs::msg::Detection;
using perception_msgs::msg::DetectionArray;
using unitree_api::msg::Request;

constexpr int API_STOP_MOVE = 1003;
constexpr int API_MOVE = 1008;

constexpr int CLASS_LEADER = Detection::CLASS_LEADER;

// python: time.monotonic()
static double monotonic()
{
  return std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Command
{
  double vx = 0.0;
  double vy = 0.0;
  double wz = 0.0;

  bool is_zero() const
  {
    return vx == 0.0 && vy == 0.0 && wz == 0.0;
  }
};


class FollowLeader : public rclcpp::Node
{
public:
  FollowLeader()
  : Node("follow_leader")
  {
    pub_ = create_publisher<Request>("/api/sport/request", 10);
    sub_ = create_subscription<DetectionArray>(
      "/perception/detections", 10,
      [this](const DetectionArray & msg) {on_detections(msg);});
    const double rate = declare_parameter("rate", 10.0);
    dt_ = 1.0 / rate;
    timer_ = create_wall_timer(
      std::chrono::duration<double>(dt_), [this]() {tick();});
  }

  void on_detections(const DetectionArray & msg)
  {
    const double now = monotonic();

    obstacles_.clear();
    for (const Detection & i : msg.detections) {
      if (i.class_id == CLASS_LEADER) {
        leader_ = i;
        last_seen_ = now;
        has_leader_ = true;
        leader_id_ = i.id;

      } else if (has_leader_ && leader_id_ && i.id == *leader_id_) {
        leader_ = i;
        last_seen_ = now;

      } else {
        // speed_triplet = (0, 0, 0) unless moving
        const Obstacle o = i.moving ?
          navigation_cpp::make_obstacle(
          i.nearest_range, i.nearest_bearing,
          i.velocity.x, i.velocity.y, i.velocity_std) :
          navigation_cpp::make_obstacle(i.nearest_range, i.nearest_bearing);

        // only keep the MAX_OBSTACLES nearest
        obstacles_.push_nearest(o);
      }
    }
  }

  void tick()
  {
    const Command cmd = compute();

    if (!cmd.is_zero()) {
      char param[128];
      std::snprintf(param, sizeof(param), "{\"x\": %f, \"y\": %f, \"z\": %f}", cmd.vx, cmd.vy, cmd.wz);
      send(API_MOVE, param);
      moving_ = true;
    } else if (moving_) {
      stop();
    }
  }

  void send(int api_id, const std::string & parameter = "")
  {
    Request msg;
    msg.header.identity.id = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
    msg.header.identity.api_id = api_id;
    msg.parameter = parameter;
    pub_->publish(msg);
  }

  void stop()
  {
    send(API_STOP_MOVE);
    moving_ = false;
  }

  Command compute()
  {
    const double now = monotonic();
    if (!has_leader_) {
      return Command();
    }

    if (!last_seen_ || now - *last_seen_ > 1.0) {
      has_leader_ = false;
      last_path_.reset();
      return Command();
    }

    if (!leader_) {
      return Command();
    }

    const Leader leader{
      leader_->range, leader_->bearing,
      leader_->velocity.x, leader_->velocity.y, leader_->velocity_std};

    const auto t0 = std::chrono::steady_clock::now();
    const std::optional<Eigen::MatrixXd> u_value =
      navigation_cpp::compute_mpc(leader, obstacles_, last_path_, dt_);
    RCLCPP_DEBUG(
      get_logger(), "mpc solve %.3f ms (%d obstacles)",
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
      obstacles_.size);

    Command cmd;

    if (u_value) {
      last_path_ = u_value;  // store the last computed path for future use
      cmd.vx = (*u_value)(0, 0);
      cmd.vy = (*u_value)(1, 0);
      cmd.wz = (*u_value)(2, 0);
      speed_ = Eigen::Vector3d(cmd.vx, cmd.vy, cmd.wz);
    } else {
      last_path_.reset();  // reset the last path if the problem is infeasible
    }
    return cmd;
  }

private:
  bool has_leader_ = false;
  std::optional<uint32_t> leader_id_;
  std::optional<double> last_seen_;
  std::optional<Detection> leader_;
  std::optional<Eigen::MatrixXd> last_path_;
  bool moving_ = false;
  Eigen::Vector3d speed_ = Eigen::Vector3d::Zero();

  navigation_cpp::ObstacleList obstacles_;

  double dt_;
  rclcpp::Publisher<Request>::SharedPtr pub_;
  rclcpp::Subscription<DetectionArray>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<FollowLeader>();
  // python: finally: node.stop(); spin_once(timeout_sec=0.1)
  // On Ctrl-C rclcpp shuts the context down itself; this runs just before, while
  // publishing still works.
  rclcpp::contexts::get_global_default_context()->add_pre_shutdown_callback(
    [node]() {
      node->stop();
      rclcpp::sleep_for(std::chrono::milliseconds(100));
    });
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
