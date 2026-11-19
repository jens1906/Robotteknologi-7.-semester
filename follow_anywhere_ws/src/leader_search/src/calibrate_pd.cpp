// Estimates the re-ID detection probability p_d per 1 m range bin and cam_range from a rosbag of
// normal following (spec: "Calibrating p_d and range").
//
//   ros2 run leader_search calibrate_pd <bag> [--max-gap 0.3] [--fov 1.52] [--cam-yaw 0]
//
// 1. Ground truth: frames where the tracker holds the leader (VISIBLE with a position), moved to
//    odom with the robot pose. LOST gaps shorter than --max-gap between two of them are
//    interpolated (default t_grace = 0.3 s).
// 2. Every camera frame (VISIBLE or LOST inside such a gap) whose ground truth is inside the FOV
//    and not occluded by the lidar scan counts as one trial at its range; VISIBLE = match.
// 3. p_d[k] = matches / frames per bin, capped at 0.95.
// 4. cam_range = far edge of the last bin with p_d >= 0.3.
// 5. Frames are correlated, so the rates are optimistic: keep the cap.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "follow_anywhere_interfaces/msg/perception_leader.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/serialization.hpp"
#include "rclcpp/serialized_message.hpp"
#include "rosbag2_cpp/reader.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

using follow_anywhere_interfaces::msg::PerceptionLeader;
using nav_msgs::msg::Odometry;
using sensor_msgs::msg::LaserScan;

namespace {

constexpr int kBins = 10;
constexpr double kPi = 3.14159265358979;

struct Pose {
  double t, x, y, yaw;
};
struct Frame {
  double t;
  bool visible;
  bool has_point;
  double bx, by;  // base_link point (VISIBLE)
  double ox = 0, oy = 0;  // odom ground truth
  bool truth = false;
};
struct Scan {
  double t;
  LaserScan msg;
};

double stamp(const builtin_interfaces::msg::Time& s) { return s.sec + 1e-9 * s.nanosec; }
double wrap(double a) { return std::remainder(a, 2 * kPi); }

template <class T>
T deserialize(const rosbag2_storage::SerializedBagMessage& m) {
  rclcpp::SerializedMessage sm(*m.serialized_data);
  T out;
  rclcpp::Serialization<T>().deserialize_message(&sm, &out);
  return out;
}

// Latest pose at or before t (poses sorted by time).
const Pose* pose_at(const std::vector<Pose>& poses, double t) {
  auto it = std::upper_bound(poses.begin(), poses.end(), t,
                             [](double v, const Pose& p) { return v < p.t; });
  return it == poses.begin() ? nullptr : &*(it - 1);
}
const Scan* scan_at(const std::vector<Scan>& scans, double t) {
  auto it = std::upper_bound(scans.begin(), scans.end(), t,
                             [](double v, const Scan& s) { return v < s.t; });
  return it == scans.begin() ? nullptr : &*(it - 1);
}

// The scan beam at bearing b (base_link) hits something nearer than range - margin.
bool occluded(const LaserScan& s, double bearing, double range) {
  if (s.ranges.empty() || s.angle_increment == 0.0f) return false;
  const int i = static_cast<int>(std::lround(wrap(bearing - s.angle_min) / s.angle_increment));
  const int n = static_cast<int>(s.ranges.size());
  const int k = ((i % n) + n) % n;
  const float r = s.ranges[k];
  return std::isfinite(r) && r >= s.range_min && r < range - 0.3;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: calibrate_pd <bag> [--max-gap s] [--fov rad] [--cam-yaw rad]\n"
                         "  [--leader-topic t] [--odom-topic t] [--scan-topic t]\n");
    return 2;
  }
  std::string bag = argv[1], leader_topic = "/perception/leader",
              odom_topic = "/utlidar/robot_odom", scan_topic = "/perception/scan";
  double max_gap = 0.3, fov = 1.52, cam_yaw = 0.0;
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string k = argv[i], v = argv[i + 1];
    if (k == "--max-gap") max_gap = std::atof(v.c_str());
    else if (k == "--fov") fov = std::atof(v.c_str());
    else if (k == "--cam-yaw") cam_yaw = std::atof(v.c_str());
    else if (k == "--leader-topic") leader_topic = v;
    else if (k == "--odom-topic") odom_topic = v;
    else if (k == "--scan-topic") scan_topic = v;
    else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
  }

  std::vector<Pose> poses;
  std::vector<Frame> frames;
  std::vector<Scan> scans;
  rosbag2_cpp::Reader reader;
  reader.open(bag);
  while (reader.has_next()) {
    auto m = reader.read_next();
    if (m->topic_name == odom_topic) {
      const auto o = deserialize<Odometry>(*m);
      const auto& q = o.pose.pose.orientation;
      poses.push_back({stamp(o.header.stamp), o.pose.pose.position.x, o.pose.pose.position.y,
                       std::atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))});
    } else if (m->topic_name == leader_topic) {
      const auto l = deserialize<PerceptionLeader>(*m);
      if (l.status == PerceptionLeader::NO_LEADER) continue;
      Frame f{stamp(l.header.stamp), l.status == PerceptionLeader::VISIBLE, false, 0, 0};
      f.has_point = f.visible && std::isfinite(l.closest_point.x) && std::isfinite(l.closest_point.y);
      f.bx = l.closest_point.x;
      f.by = l.closest_point.y;
      frames.push_back(f);
    } else if (m->topic_name == scan_topic) {
      const auto s = deserialize<LaserScan>(*m);
      scans.push_back({stamp(s.header.stamp), s});
    }
  }
  auto by_t = [](const auto& a, const auto& b) { return a.t < b.t; };
  std::sort(poses.begin(), poses.end(), by_t);
  std::sort(frames.begin(), frames.end(), by_t);
  std::sort(scans.begin(), scans.end(), by_t);
  std::printf("bag %s: %zu leader frames, %zu odometry, %zu scans\n", bag.c_str(), frames.size(),
              poses.size(), scans.size());
  if (frames.empty() || poses.empty()) {
    std::fprintf(stderr, "need %s and %s in the bag\n", leader_topic.c_str(), odom_topic.c_str());
    return 1;
  }

  // 1. Ground truth from VISIBLE frames, in odom.
  for (Frame& f : frames) {
    const Pose* p = f.has_point ? pose_at(poses, f.t) : nullptr;
    if (!p) continue;
    const double c = std::cos(p->yaw), s = std::sin(p->yaw);
    f.ox = p->x + c * f.bx - s * f.by;
    f.oy = p->y + s * f.bx + c * f.by;
    f.truth = true;
  }
  // Interpolate short LOST gaps between two sightings.
  int prev = -1;
  for (int i = 0; i < static_cast<int>(frames.size()); ++i) {
    if (!frames[i].truth) continue;
    if (prev >= 0 && i - prev > 1 && frames[i].t - frames[prev].t <= max_gap) {
      for (int j = prev + 1; j < i; ++j) {
        const double a = (frames[j].t - frames[prev].t) / (frames[i].t - frames[prev].t);
        frames[j].ox = frames[prev].ox + a * (frames[i].ox - frames[prev].ox);
        frames[j].oy = frames[prev].oy + a * (frames[i].oy - frames[prev].oy);
        frames[j].truth = true;
      }
    }
    prev = i;
  }

  // 2. Trials per range bin.
  long trials[kBins] = {0}, matches[kBins] = {0}, outside = 0, hidden = 0;
  for (const Frame& f : frames) {
    if (!f.truth) continue;
    const Pose* p = pose_at(poses, f.t);
    if (!p) continue;
    const double dx = f.ox - p->x, dy = f.oy - p->y;
    const double range = std::hypot(dx, dy), bearing = wrap(std::atan2(dy, dx) - p->yaw);
    if (std::fabs(wrap(bearing - cam_yaw)) > fov / 2) {
      ++outside;
      continue;
    }
    const Scan* s = scan_at(scans, f.t);
    if (!f.visible && s && occluded(s->msg, bearing, range)) {
      ++hidden;
      continue;
    }
    const int k = std::min(kBins - 1, static_cast<int>(range));
    ++trials[k];
    matches[k] += f.visible;
  }

  // 3./4. p_d per bin (cap 0.95) and cam_range.
  std::printf("\nbin    frames  matches  p_d\n");
  double pd[kBins];
  double cam_range = 0.0;
  for (int k = 0; k < kBins; ++k) {
    pd[k] = trials[k] ? std::min(0.95, double(matches[k]) / trials[k]) : -1.0;
    if (pd[k] >= 0.3) cam_range = k + 1.0;
    std::printf("%d-%2d m %7ld %8ld  %s\n", k, k + 1, trials[k], matches[k],
                trials[k] ? std::to_string(pd[k]).substr(0, 5).c_str() : "n/a");
  }
  std::printf("\nskipped: %ld outside the FOV, %ld occluded by the lidar\n", outside, hidden);
  std::printf("note: consecutive frames are correlated, so these rates are optimistic.\n\n");
  std::printf("# for config/leader_search.yaml (bins without frames keep 0.6)\n");
  std::printf("    p_d: [");
  for (int k = 0; k < kBins; ++k)
    std::printf("%.2f%s", pd[k] >= 0 ? pd[k] : 0.6, k + 1 < kBins ? ", " : "]\n");
  std::printf("    cam_range: %.1f\n", cam_range);
  return 0;
}
