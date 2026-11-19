// ROS 2 wrapper of the search core: converts messages and calls SearchFilter. No algorithm logic.
//
// In:  /perception/leader (PerceptionLeader, base_link): VISIBLE = re-ID match, LOST = camera frame
//      without a match ("looked here, not there"). /perception/scan (LaserScan, base_link).
//      /utlidar/robot_odom (Odometry).
// Out: /search/path (nav_msgs/Path, odom), /search/mode (UInt8), /search/target (PointStamped,
//      odom: the predicted leader, for the MPC's line-of-sight rows), debug /search/belief and
//      /search/markers.
#include <ament_index_cpp/get_package_share_path.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "follow_anywhere_interfaces/msg/perception_leader.hpp"
#include "leader_search/search_filter.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "std_msgs/msg/u_int8.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace leader_search {

using follow_anywhere_interfaces::msg::PerceptionLeader;
using nav_msgs::msg::OccupancyGrid;
using nav_msgs::msg::Odometry;
using nav_msgs::msg::Path;
using sensor_msgs::msg::LaserScan;
using visualization_msgs::msg::Marker;
using visualization_msgs::msg::MarkerArray;

namespace {

Pose2 compose(const Pose2& a, const Pose2& b) {  // a * b
  const float c = std::cos(a.yaw), s = std::sin(a.yaw);
  return {a.x + c * b.x - s * b.y, a.y + s * b.x + c * b.y, a.yaw + b.yaw};
}
Pose2 inverse(const Pose2& a) {
  const float c = std::cos(a.yaw), s = std::sin(a.yaw);
  return {-c * a.x - s * a.y, s * a.x - c * a.y, -a.yaw};
}
Vec2 rotate(float yaw, float x, float y) {
  return {std::cos(yaw) * x - std::sin(yaw) * y, std::sin(yaw) * x + std::cos(yaw) * y};
}
geometry_msgs::msg::Quaternion yaw_quat(float yaw) {
  geometry_msgs::msg::Quaternion q;
  q.z = std::sin(0.5 * yaw);
  q.w = std::cos(0.5 * yaw);
  return q;
}
geometry_msgs::msg::Point point(float x, float y, float z = 0.0f) {
  geometry_msgs::msg::Point p;
  p.x = x;
  p.y = y;
  p.z = z;
  return p;
}

}  // namespace

class LeaderSearchNode : public rclcpp::Node {
 public:
  explicit LeaderSearchNode(const rclcpp::NodeOptions& options)
      : Node("leader_search", options) {
    const std::string share = ament_index_cpp::get_package_share_path("leader_search").string();
    const auto file = declare_parameter<std::string>("params_file", share + "/config/leader_search.yaml");
    Params p;
    std::string err;
    if (!load_params(file, &p, &err) || !filter_.init(p, &err)) {
      RCLCPP_FATAL(get_logger(), "leader_search params %s: %s", file.c_str(), err.c_str());
      throw std::runtime_error(err);
    }
    const auto leader_topic = declare_parameter<std::string>("leader_topic", "/perception/leader");
    const auto scan_topic = declare_parameter<std::string>("scan_topic", "/perception/scan");
    const auto odom_topic = declare_parameter<std::string>("odom_topic", "/utlidar/robot_odom");
    belief_period_ = 1.0 / declare_parameter("belief_rate_hz", 2.0);

    ranges_.reserve(4096);
    marginal_.resize(filter_.num_cells());
    belief_msg_.data.resize(filter_.num_cells());

    path_pub_ = create_publisher<Path>("/search/path", 10);
    mode_pub_ = create_publisher<std_msgs::msg::UInt8>("/search/mode", 10);
    target_pub_ = create_publisher<geometry_msgs::msg::PointStamped>("/search/target", 10);
    belief_pub_ = create_publisher<OccupancyGrid>("/search/belief", 1);
    marker_pub_ = create_publisher<MarkerArray>("/search/markers", 1);
    odom_sub_ = create_subscription<Odometry>(odom_topic, rclcpp::SensorDataQoS(),
                                              [this](Odometry::ConstSharedPtr m) { on_odom(*m); });
    leader_sub_ = create_subscription<PerceptionLeader>(
        leader_topic, 10, [this](PerceptionLeader::ConstSharedPtr m) { on_leader(*m); });
    scan_sub_ = create_subscription<LaserScan>(scan_topic, rclcpp::SensorDataQoS(),
                                               [this](LaserScan::ConstSharedPtr m) { on_scan(*m); });
    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / p.rate_hz), [this] { tick(); });
    RCLCPP_INFO(get_logger(), "leader_search: %dx%d cells of %.2f m at %.0f Hz, %zu KB, params %s",
                filter_.window_cells(), filter_.window_cells(), filter_.resolution(), p.rate_hz,
                filter_.arena_bytes() / 1024, file.c_str());
  }

 private:
  void on_odom(const Odometry& m) {
    const auto& q = m.pose.pose.orientation;
    robot_ = Pose2{float(m.pose.pose.position.x), float(m.pose.pose.position.y),
                   float(std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)))};
    if (!m.header.frame_id.empty()) odom_frame_ = m.header.frame_id;
    have_odom_ = true;
  }

  void on_scan(const LaserScan& m) {
    if (!have_odom_) return;
    const size_t n = std::min(m.ranges.size(), ranges_.capacity());
    ranges_.assign(m.ranges.begin(), m.ranges.begin() + n);
    scan_ = ScanView{nullptr, int32_t(n), m.angle_min, m.angle_increment, m.range_min, m.range_max, {}};
    scan_robot_ = robot_;  // robot pose when the scan arrived (scan is in base_link)
    have_scan_ = true;
  }

  void on_leader(const PerceptionLeader& m) {
    if (!have_odom_ || m.status == PerceptionLeader::NO_LEADER) return;
    Observation o;
    o.robot = robot_;
    o.matched = m.status == PerceptionLeader::VISIBLE;
    if (o.matched && std::isfinite(m.closest_point.x) && std::isfinite(m.closest_point.y)) {
      // base_link -> odom with the current robot pose.
      const Vec2 p = rotate(robot_.yaw, m.closest_point.x, m.closest_point.y);
      const Vec2 v = rotate(robot_.yaw, m.velocity.x, m.velocity.y);
      const bool v_ok = std::isfinite(v.x) && std::isfinite(v.y);
      o.track.p = Vec2{robot_.x + p.x, robot_.y + p.y};
      o.track.v = v_ok ? v : Vec2{};
      const float sd = std::isfinite(m.position_uncertainty) ? m.position_uncertainty : 0.0f;
      o.track.cov[0] = o.track.cov[3] = sd * sd;
      o.has_track = true;
    }
    if (n_obs_ < obs_.size()) {
      obs_[n_obs_++] = o;
    } else {
      obs_[obs_.size() - 1] = o;  // full: keep the newest
    }
  }

  void tick() {
    const auto now = std::chrono::steady_clock::now();
    const float dt = last_tick_ ? std::min(0.5f, std::chrono::duration<float>(now - *last_tick_).count())
                                : float(1.0 / filter_.params().rate_hz);
    last_tick_ = now;
    if (!have_odom_) return;
    // Latest scan (taken at scan_robot_), placed relative to the current pose. It is passed every
    // tick, so the search always starts with walls even without a new scan in that tick.
    ScanView* scan = nullptr;
    if (have_scan_) {
      scan_.ranges = ranges_.data();
      scan_.sensor_in_base = compose(inverse(robot_), scan_robot_);
      scan = &scan_;
    }
    const SearchOutput out = filter_.step(robot_, scan, obs_.data(), int32_t(n_obs_), dt);
    n_obs_ = 0;
    if (out.clamped) RCLCPP_WARN(get_logger(), "leader loss point outside the window, clamped");
    if (out.mode != last_mode_)
      RCLCPP_INFO(get_logger(), "mode %d -> %d", int(last_mode_), int(out.mode));
    publish(out);
    last_mode_ = out.mode;
  }

  void publish(const SearchOutput& out) {
    const auto stamp = now();
    std_msgs::msg::UInt8 mode;
    mode.data = static_cast<uint8_t>(out.mode);
    mode_pub_->publish(mode);

    Path path;
    path.header.stamp = stamp;
    path.header.frame_id = odom_frame_;
    if (out.mode != Mode::TRACKING && out.path_valid) {
      path.poses.resize(out.path.size);
      for (int i = 0; i < out.path.size; ++i) {
        auto& ps = path.poses[i];
        ps.header = path.header;
        ps.pose.position = point(out.path.poses[i].x, out.path.poses[i].y);
        ps.pose.orientation = yaw_quat(out.path.poses[i].yaw);
      }
    }
    path_pub_->publish(path);
    if (out.mode == Mode::SEARCH && out.path_valid) {
      geometry_msgs::msg::PointStamped target;
      target.header = path.header;
      target.point = point(out.target.x, out.target.y);
      target_pub_->publish(target);
    }

    if (out.mode == Mode::TRACKING) {
      if (last_mode_ != Mode::TRACKING) {
        MarkerArray clear;
        clear.markers.emplace_back().action = Marker::DELETEALL;
        marker_pub_->publish(clear);
      }
      return;
    }
    publish_markers(out, stamp);
    if ((stamp - last_belief_).seconds() >= belief_period_) {
      last_belief_ = stamp;
      publish_belief(stamp);
    }
  }

  void publish_markers(const SearchOutput& out, const rclcpp::Time& stamp) {
    MarkerArray arr;
    auto make = [&](int id, int type) -> Marker& {
      Marker& m = arr.markers.emplace_back();
      m.header.stamp = stamp;
      m.header.frame_id = odom_frame_;
      m.ns = "search";
      m.id = id;
      m.type = type;
      m.pose.orientation.w = 1.0;
      m.color.a = 1.0;
      return m;
    };
    Marker& target = make(0, Marker::SPHERE);
    target.pose.position = point(out.target.x, out.target.y, 0.3f);
    target.scale.x = target.scale.y = target.scale.z = 0.3;
    target.color.r = 1.0;
    target.text = std::to_string(out.target_heading);
    Marker& line = make(1, Marker::LINE_STRIP);
    line.scale.x = 0.05;
    line.color.g = 1.0;
    line.points = {point(out.target.x, out.target.y, 0.1f), point(out.intercept.x, out.intercept.y, 0.1f)};
    Marker& fov = make(2, Marker::LINE_STRIP);
    fov.scale.x = 0.03;
    fov.color.b = 1.0;
    const Params& p = filter_.params();
    const float yaw = robot_.yaw + p.cam_yaw;
    fov.points.push_back(point(robot_.x, robot_.y));
    for (int i = 0; i <= 8; ++i) {
      const float a = yaw - 0.5f * p.cam_fov + p.cam_fov * i / 8.0f;
      fov.points.push_back(point(robot_.x + p.cam_range * std::cos(a), robot_.y + p.cam_range * std::sin(a)));
    }
    fov.points.push_back(point(robot_.x, robot_.y));
    marker_pub_->publish(arr);
  }

  void publish_belief(const rclcpp::Time& stamp) {
    filter_.belief_marginal(marginal_.data());
    float mx = 0.0f;
    for (float v : marginal_) mx = std::max(mx, v);
    belief_msg_.header.stamp = stamp;
    belief_msg_.header.frame_id = odom_frame_;
    belief_msg_.info.resolution = filter_.resolution();
    belief_msg_.info.width = belief_msg_.info.height = filter_.window_cells();
    belief_msg_.info.origin.position = point(filter_.window_origin().x, filter_.window_origin().y);
    belief_msg_.info.origin.orientation.w = 1.0;
    for (size_t c = 0; c < marginal_.size(); ++c)
      belief_msg_.data[c] = mx > 0.0f ? int8_t(std::lround(100.0f * marginal_[c] / mx)) : 0;
    belief_pub_->publish(belief_msg_);
  }

  SearchFilter filter_;
  Pose2 robot_, scan_robot_;
  bool have_odom_ = false, have_scan_ = false;
  std::string odom_frame_ = "odom";
  std::vector<float> ranges_;
  ScanView scan_;
  std::array<Observation, 32> obs_;
  size_t n_obs_ = 0;
  Mode last_mode_ = Mode::TRACKING;
  std::optional<std::chrono::steady_clock::time_point> last_tick_;
  std::vector<float> marginal_;
  OccupancyGrid belief_msg_;
  double belief_period_ = 0.5;
  rclcpp::Time last_belief_{0, 0, RCL_ROS_TIME};

  rclcpp::Publisher<Path>::SharedPtr path_pub_;
  rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr mode_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr target_pub_;
  rclcpp::Publisher<OccupancyGrid>::SharedPtr belief_pub_;
  rclcpp::Publisher<MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Subscription<Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<PerceptionLeader>::SharedPtr leader_sub_;
  rclcpp::Subscription<LaserScan>::SharedPtr scan_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace leader_search

RCLCPP_COMPONENTS_REGISTER_NODE(leader_search::LeaderSearchNode)
