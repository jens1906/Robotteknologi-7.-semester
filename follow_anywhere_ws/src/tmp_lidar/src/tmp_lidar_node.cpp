// tmp_lidar_node: the lidar part of perception, fused with the camera part. The only file with ROS I/O.
//
// Subscribes:
//   /utlidar/cloud_base           sensor_msgs/PointCloud2  base_link, Go2 clock, ~15 Hz
//   /utlidar/robot_odom           nav_msgs/Odometry        odom -> base_link, Go2 clock
//   /perception/camera/leader     follow_anywhere_interfaces/PerceptionLeader         (perception_node, remapped)
//   /perception/camera/obstacles  follow_anywhere_interfaces/PerceptionObstacleArray  (perception_node, remapped)
// Publishes (what navigation reads; same topics and types as 2dSim's fake_perception):
//   /perception/leader      PerceptionLeader         enriched with the lidar
//   /perception/obstacles   PerceptionObstacleArray  camera people (enriched) + lidar-only clusters, no duplicates
//   /perception/scan        sensor_msgs/LaserScan    360 deg, obstacle points of the newest cloud
//   /perception/lidar_debug visualization_msgs/MarkerArray  clusters (grey static, green person, blue leader)
// Every topic name is a parameter.
//
// Time: the Go2 stamps the cloud and odometry with its own clock (~38 s off the PC). Cloud and odometry are
// matched on the Go2 clock directly. The offset Go2 -> this computer is the smallest (arrival - stamp) over
// clockOffsetWindow (as perception's ClockOffsetEstimator); with it the camera messages (stamped on this
// computer) are moved to the cloud's time, and the outputs are stamped on this computer's clock, because
// navigation measures message age against its own now().
// The fusion runs once per cloud (cloudsPerUpdate > 1: every that many clouds). A nearly empty cloud is skipped and
// the last outputs are repeated (CloudGate). /perception/scan is made from the published obstacles (scanFromObstacles)
// or from the points of the last scanClouds clouds, held in odom. While no cloud arrived for lidarTimeout, camera messages are passed through
// unchanged, so the system still works without the lidar.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "follow_anywhere_interfaces/msg/perception_leader.hpp"
#include "follow_anywhere_interfaces/msg/perception_obstacle.hpp"
#include "follow_anywhere_interfaces/msg/perception_obstacle_array.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "tmp_lidar/cloud_filter.hpp"
#include "tmp_lidar/cloud_gate.hpp"
#include "tmp_lidar/cluster_tracker.hpp"
#include "tmp_lidar/clustering.hpp"
#include "tmp_lidar/fusion.hpp"
#include "tmp_lidar/hit_grid.hpp"
#include "tmp_lidar/scan_builder.hpp"

namespace tmp_lidar
{

using follow_anywhere_interfaces::msg::PerceptionLeader;
using follow_anywhere_interfaces::msg::PerceptionObstacle;
using follow_anywhere_interfaces::msg::PerceptionObstacleArray;
using nav_msgs::msg::Odometry;
using sensor_msgs::msg::LaserScan;
using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;
using visualization_msgs::msg::Marker;
using visualization_msgs::msg::MarkerArray;

namespace
{

constexpr double kPoseHistorySeconds = 3.0;
constexpr double kPoseMaxGap = 0.5;   // s, a lookup farther than this from any sample fails

double toSeconds(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<double>(stamp.sec) + 1e-9 * static_cast<double>(stamp.nanosec);
}

builtin_interfaces::msg::Time toStamp(double seconds)
{
  return rclcpp::Time(static_cast<int64_t>(std::llround(seconds * 1e9)), RCL_ROS_TIME);
}

double headingOf(const geometry_msgs::msg::Quaternion & q)
{
  return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

double wrapAngle(double a)
{
  return std::atan2(std::sin(a), std::cos(a));
}

Point2 fromMsg(const geometry_msgs::msg::Point & p) {return {p.x, p.y};}
Point2 fromMsg(const geometry_msgs::msg::Vector3 & v) {return {v.x, v.y};}

geometry_msgs::msg::Point toPointMsg(const Point2 & p)
{
  geometry_msgs::msg::Point out;
  out.x = p.x;
  out.y = p.y;
  out.z = isFinite(p) ? 0.0 : kNaN;
  return out;
}

geometry_msgs::msg::Vector3 toVectorMsg(const Point2 & v)
{
  geometry_msgs::msg::Vector3 out;
  out.x = v.x;
  out.y = v.y;
  out.z = 0.0;
  return out;
}

// x, y, z of a PointCloud2 (float32 or float64 fields), in the cloud's frame.
std::vector<Point3> readCloud(const PointCloud2 & msg)
{
  int offsets[3] = {-1, -1, -1};
  uint8_t types[3] = {0, 0, 0};
  const char * names[3] = {"x", "y", "z"};
  for (const auto & field : msg.fields) {
    for (int k = 0; k < 3; ++k) {
      if (field.name == names[k]) {
        offsets[k] = static_cast<int>(field.offset);
        types[k] = field.datatype;
      }
    }
  }
  std::vector<Point3> out;
  for (int k = 0; k < 3; ++k) {
    if (offsets[k] < 0 || (types[k] != PointField::FLOAT32 && types[k] != PointField::FLOAT64)) {return out;}
  }
  const std::size_t count = static_cast<std::size_t>(msg.width) * msg.height;
  if (msg.point_step == 0 || msg.data.size() < count * msg.point_step) {return out;}
  out.reserve(count);
  auto read = [&](const uint8_t * base, int k) {
      if (types[k] == PointField::FLOAT32) {
        float v;
        std::memcpy(&v, base + offsets[k], sizeof v);
        return static_cast<double>(v);
      }
      double v;
      std::memcpy(&v, base + offsets[k], sizeof v);
      return v;
    };
  for (std::size_t i = 0; i < count; ++i) {
    const uint8_t * base = msg.data.data() + i * msg.point_step;
    out.push_back({read(base, 0), read(base, 1), read(base, 2)});
  }
  return out;
}

// Robot poses on the Go2 clock, interpolated.
class PoseHistory
{
public:
  void add(double t, const Pose2 & pose)
  {
    if (!samples_.empty() && t < samples_.back().first) {samples_.clear();}   // time jumped back (bag loop)
    samples_.emplace_back(t, pose);
    while (!samples_.empty() && samples_.front().first < t - kPoseHistorySeconds) {samples_.pop_front();}
  }

  std::optional<Pose2> at(double t) const
  {
    if (samples_.empty()) {return std::nullopt;}
    if (t <= samples_.front().first) {
      return samples_.front().first - t <= kPoseMaxGap ? std::optional<Pose2>(samples_.front().second) :
             std::nullopt;
    }
    if (t >= samples_.back().first) {
      return t - samples_.back().first <= kPoseMaxGap ? std::optional<Pose2>(samples_.back().second) :
             std::nullopt;
    }
    for (std::size_t i = 1; i < samples_.size(); ++i) {
      if (samples_[i].first < t) {continue;}
      const auto & [t0, a] = samples_[i - 1];
      const auto & [t1, b] = samples_[i];
      const double s = t1 > t0 ? (t - t0) / (t1 - t0) : 0.0;
      return Pose2{a.x + s * (b.x - a.x), a.y + s * (b.y - a.y), a.yaw + s * wrapAngle(b.yaw - a.yaw)};
    }
    return samples_.back().second;
  }

  void clear() {samples_.clear();}

private:
  std::deque<std::pair<double, Pose2>> samples_;
};

// Go2 clock -> this computer's clock: smallest (arrival - stamp) over a window.
class ClockOffset
{
public:
  explicit ClockOffset(double window)
  : window_(window) {}

  void add(double stamp, double arrival)
  {
    const double offset = arrival - stamp;
    while (!samples_.empty() && samples_.back().second >= offset) {samples_.pop_back();}
    samples_.emplace_back(arrival, offset);
    while (samples_.front().first < arrival - window_) {samples_.pop_front();}
  }

  bool known() const {return !samples_.empty();}
  double offset() const {return samples_.empty() ? 0.0 : samples_.front().second;}
  void reset() {samples_.clear();}

private:
  double window_;
  std::deque<std::pair<double, double>> samples_;   // arrival, offset (offsets increasing)
};

}  // namespace

class TmpLidarNode : public rclcpp::Node
{
public:
  TmpLidarNode()
  : Node("tmp_lidar"), clockOffset_(declare_parameter("clockOffsetWindow", 5.0))
  {
    // --- topics
    const auto cloudTopic = declare_parameter<std::string>("cloudTopic", "/utlidar/cloud_base");
    const auto odometryTopic = declare_parameter<std::string>("odometryTopic", "/utlidar/robot_odom");
    const auto cameraLeaderTopic = declare_parameter<std::string>("cameraLeaderTopic", "/perception/camera/leader");
    const auto cameraObstaclesTopic = declare_parameter<std::string>("cameraObstaclesTopic", "/perception/camera/obstacles");
    const auto leaderTopic = declare_parameter<std::string>("leaderTopic", "/perception/leader");
    const auto obstaclesTopic = declare_parameter<std::string>("obstaclesTopic", "/perception/obstacles");
    const auto scanTopic = declare_parameter<std::string>("scanTopic", "/perception/scan");
    const auto debugTopic = declare_parameter<std::string>("debugTopic", "/perception/lidar_debug");
    baseFrame_ = declare_parameter<std::string>("baseFrame", "base_link");

    // --- cloud filter
    filter_.floorZ = declare_parameter("floorZ", filter_.floorZ);
    filter_.fitFloorPlane = declare_parameter("fitFloorPlane", filter_.fitFloorPlane);
    filter_.floorSearch = declare_parameter("floorSearch", filter_.floorSearch);
    filter_.floorFitRange = declare_parameter("floorFitRange", filter_.floorFitRange);
    filter_.floorMaxTiltDeg = declare_parameter("floorMaxTilt", filter_.floorMaxTiltDeg);
    filter_.floorInlierBand = declare_parameter("floorInlierBand", filter_.floorInlierBand);
    filter_.floorMinimumPoints = static_cast<int>(
      declare_parameter("floorMinimumPoints", static_cast<int64_t>(filter_.floorMinimumPoints)));
    filter_.groundHeight = declare_parameter("groundHeight", filter_.groundHeight);
    filter_.maxHeight = declare_parameter("maxHeight", filter_.maxHeight);
    const auto bodyMin = declare_parameter("bodyBoxMin", std::vector<double>{filter_.bodyMin.x, filter_.bodyMin.y});
    const auto bodyMax = declare_parameter("bodyBoxMax", std::vector<double>{filter_.bodyMax.x, filter_.bodyMax.y});
    if (bodyMin.size() >= 2) {filter_.bodyMin = {bodyMin[0], bodyMin[1]};}
    if (bodyMax.size() >= 2) {filter_.bodyMax = {bodyMax[0], bodyMax[1]};}
    filter_.rangeMin = declare_parameter("rangeMin", filter_.rangeMin);
    filter_.rangeMax = declare_parameter("obstacleMaxRange", filter_.rangeMax);

    // --- hit grid (short memory in odom)
    HitGridParams gridParams;
    gridParams.cellSize = declare_parameter("hitCellSize", gridParams.cellSize);
    gridParams.memorySeconds = declare_parameter("memorySeconds", gridParams.memorySeconds);
    gridParams.minCellHits = static_cast<int>(
      declare_parameter("minCellHits", static_cast<int64_t>(gridParams.minCellHits)));
    hitGrid_ = HitGrid(gridParams);

    // --- clustering
    ClusterParams clusterParams;
    clusterParams.cellSize = declare_parameter("cellSize", clusterParams.cellSize);
    clusterParams.linkDistance = declare_parameter("linkDistance", clusterParams.linkDistance);
    clusterParams.linkAngle = declare_parameter("linkAngle", clusterParams.linkAngle);
    clusterParams.minClusterPoints = static_cast<int>(
      declare_parameter("minClusterPoints", static_cast<int64_t>(clusterParams.minClusterPoints)));
    clusterParams.minClusterPointsFar = static_cast<int>(
      declare_parameter("minClusterPointsFar", static_cast<int64_t>(clusterParams.minClusterPointsFar)));
    clusterParams.farRange = filter_.rangeMax;
    clusterParams.maxClusterRadius = declare_parameter("maxClusterRadius", clusterParams.maxClusterRadius);
    clusterParams.chunkSize = declare_parameter("wallChunk", clusterParams.chunkSize);
    clusterParams.freshWindow = declare_parameter("freshWindow", clusterParams.freshWindow);
    clusterParams.minFreshPoints = static_cast<int>(
      declare_parameter("minFreshPoints", static_cast<int64_t>(clusterParams.minFreshPoints)));
    clusterParams_ = clusterParams;

    // --- tracking
    TrackerParams trackerParams;
    auto & tp = trackerParams;
    tp.gateChi2 = declare_parameter("gateChi2", tp.gateChi2);
    tp.minimumGate = declare_parameter("minimumGate", tp.minimumGate);
    tp.maximumGate = declare_parameter("maximumGate", tp.maximumGate);
    tp.mergeRadius = declare_parameter("mergeRadius", tp.mergeRadius);
    tp.shapeSigmaScale = declare_parameter("shapeSigmaScale", tp.shapeSigmaScale);
    tp.staleCentroidSigma = declare_parameter("staleCentroidSigma", tp.staleCentroidSigma);
    tp.noiseAdaptRate = declare_parameter("noiseAdaptRate", tp.noiseAdaptRate);
    tp.maxNoiseScale = declare_parameter("maxNoiseScale", tp.maxNoiseScale);
    tp.velocitySightings = static_cast<int>(
      declare_parameter("velocitySightings", static_cast<int64_t>(tp.velocitySightings)));
    tp.imm.staticPositionNoise = declare_parameter("staticPositionNoise", tp.imm.staticPositionNoise);
    tp.imm.cvAccelerationNoise = declare_parameter("cvAccelerationNoise", tp.imm.cvAccelerationNoise);
    tp.imm.ctAccelerationNoise = declare_parameter("ctAccelerationNoise", tp.imm.ctAccelerationNoise);
    tp.imm.ctTurnNoise = declare_parameter("ctTurnNoise", tp.imm.ctTurnNoise);
    tp.imm.modeSwitchRate = declare_parameter("modeSwitchRate", tp.imm.modeSwitchRate);
    tp.minimumSpeed = declare_parameter("minimumSpeed", tp.minimumSpeed);
    tp.staticPublishProbability = declare_parameter("staticPublishProbability", tp.staticPublishProbability);
    tp.unknownVelocitySigma = declare_parameter("unknownVelocitySigma", tp.unknownVelocitySigma);
    tp.minimumVelocitySigma = declare_parameter("minimumVelocitySigma", tp.minimumVelocitySigma);
    tp.staticVelocitySigma = declare_parameter("staticVelocitySigma", tp.staticVelocitySigma);
    tp.birthExistence = declare_parameter("birthExistence", tp.birthExistence);
    tp.maxExistence = declare_parameter("maxExistence", tp.maxExistence);
    tp.confirmExistence = declare_parameter("confirmExistence", tp.confirmExistence);
    tp.keepExistence = declare_parameter("keepExistence", tp.keepExistence);
    tp.deleteExistence = declare_parameter("deleteExistence", tp.deleteExistence);
    tp.detectionProbabilityNear = declare_parameter("detectionProbabilityNear", tp.detectionProbabilityNear);
    tp.detectionProbabilityFar = declare_parameter("detectionProbabilityFar", tp.detectionProbabilityFar);
    tp.detectionNearRange = declare_parameter("detectionNearRange", tp.detectionNearRange);
    tp.detectionFarRange = declare_parameter("detectionFarRange", tp.detectionFarRange);
    tp.falseAlarmProbability = declare_parameter("falseAlarmProbability", tp.falseAlarmProbability);
    tp.falseAlarmPoints = static_cast<int>(
      declare_parameter("falseAlarmPoints", static_cast<int64_t>(tp.falseAlarmPoints)));
    tp.occludedDetectionFactor = declare_parameter("occludedDetectionFactor", tp.occludedDetectionFactor);
    tp.occlusionMargin = declare_parameter("occlusionMargin", tp.occlusionMargin);
    tp.trackTimeout = declare_parameter("trackTimeout", tp.trackTimeout);
    tp.occludedTimeout = declare_parameter("occludedTimeout", tp.occludedTimeout);
    tp.maxPositionSigma = declare_parameter("maxPositionSigma", tp.maxPositionSigma);
    tp.rangeMin = filter_.rangeMin;
    tp.rangeMax = filter_.rangeMax;

    // --- fusion
    FusionParams fusionParams;
    fusionParams.matchGate = declare_parameter("matchGate", fusionParams.matchGate);
    fusionParams.matchSigmas = declare_parameter("matchSigmas", fusionParams.matchSigmas);
    fusionParams.stickyBonus = declare_parameter("stickyBonus", fusionParams.stickyBonus);
    fusionParams.leaderHoldTime = declare_parameter("leaderHoldTime", fusionParams.leaderHoldTime);
    const auto rangeStd = declare_parameter("lidarRangeStd",
        std::vector<double>{fusionParams.lidarRangeStd[0], fusionParams.lidarRangeStd[1]});
    if (rangeStd.size() >= 2) {fusionParams.lidarRangeStd = {rangeStd[0], rangeStd[1]};}
    fusionParams.minimumPositionUncertainty =
      declare_parameter("minimumPositionUncertainty", fusionParams.minimumPositionUncertainty);
    const auto camera = declare_parameter("cameraPosition",
        std::vector<double>{fusionParams.cameraPosition.x, fusionParams.cameraPosition.y});
    if (camera.size() >= 2) {fusionParams.cameraPosition = {camera[0], camera[1]};}
    fusion_ = Fusion(fusionParams);
    trackerParams.lidarRangeStd = fusionParams.lidarRangeStd;
    tracker_ = ClusterTracker(trackerParams);
    cameraTimeout_ = declare_parameter("cameraTimeout", 0.5);
    lidarTimeout_ = declare_parameter("lidarTimeout", 0.5);
    odometryTimeout_ = declare_parameter("odometryTimeout", 0.5);

    // --- scan, debug
    scanRays_ = static_cast<int>(declare_parameter("scanRays", static_cast<int64_t>(720)));
    cloudsPerUpdate_ = std::max(1, static_cast<int>(declare_parameter("cloudsPerUpdate", static_cast<int64_t>(1))));
    scanFromObstacles_ = declare_parameter("scanFromObstacles", true);
    scanClouds_ = std::max(1, static_cast<int>(declare_parameter("scanClouds", static_cast<int64_t>(2))));
    CloudGateParams gateParams;
    gateParams.minFraction = declare_parameter("emptyCloudFraction", gateParams.minFraction);
    gateParams.minPoints = static_cast<std::size_t>(
      declare_parameter("emptyCloudPoints", static_cast<int64_t>(gateParams.minPoints)));
    gateParams.maxSkipped = static_cast<int>(
      declare_parameter("maxSkippedClouds", static_cast<int64_t>(gateParams.maxSkipped)));
    cloudGate_ = CloudGate(gateParams);
    publishDebug_ = declare_parameter("publishDebug", true);

    // Odometry first, so each executor cycle has the newest pose before the cloud that uses it.
    const auto sensorQos = rclcpp::SensorDataQoS().keep_last(5);
    odometrySub_ = create_subscription<Odometry>(odometryTopic, sensorQos,
        [this](Odometry::ConstSharedPtr msg) {handleOdometry(*msg);});
    cloudSub_ = create_subscription<PointCloud2>(cloudTopic, rclcpp::SensorDataQoS().keep_last(1),
        [this](PointCloud2::ConstSharedPtr msg) {handleCloud(*msg);});
    cameraLeaderSub_ = create_subscription<PerceptionLeader>(cameraLeaderTopic, 10,
        [this](PerceptionLeader::ConstSharedPtr msg) {handleCameraLeader(msg);});
    cameraObstaclesSub_ = create_subscription<PerceptionObstacleArray>(cameraObstaclesTopic, 10,
        [this](PerceptionObstacleArray::ConstSharedPtr msg) {handleCameraObstacles(msg);});

    leaderPub_ = create_publisher<PerceptionLeader>(leaderTopic, 10);
    obstaclesPub_ = create_publisher<PerceptionObstacleArray>(obstaclesTopic, 10);
    scanPub_ = create_publisher<LaserScan>(scanTopic, 10);
    if (publishDebug_) {debugPub_ = create_publisher<MarkerArray>(debugTopic, 10);}

    RCLCPP_INFO(get_logger(), "tmp_lidar: %s + %s -> %s, %s, %s", cloudTopic.c_str(), cameraLeaderTopic.c_str(),
      leaderTopic.c_str(), obstaclesTopic.c_str(), scanTopic.c_str());
  }

private:
  double nowSeconds() {return now().seconds();}

  void handleOdometry(const Odometry & msg)
  {
    const double arrival = nowSeconds();
    if (lastOdometryArrival_ >= 0.0 && arrival - lastOdometryArrival_ > odometryTimeout_) {
      RCLCPP_WARN(get_logger(), "odometry back after %.1f s: tracks restarted", arrival - lastOdometryArrival_);
      clockOffset_.reset();
      poses_.clear();
      hitGrid_.clear();
      tracker_.reset();
      recentClouds_.clear();
      cloudsSinceUpdate_ = 0;
      cloudGate_.reset();
    }
    lastOdometryArrival_ = arrival;
    const double stamp = toSeconds(msg.header.stamp);
    clockOffset_.add(stamp, arrival);
    const auto & p = msg.pose.pose.position;
    poses_.add(stamp, Pose2{p.x, p.y, headingOf(msg.pose.pose.orientation)});
  }

  void handleCameraLeader(PerceptionLeader::ConstSharedPtr msg)
  {
    cameraLeader_ = msg;
    cameraLeaderArrival_ = nowSeconds();
    if (lidarStale()) {leaderPub_->publish(*msg);}   // no lidar: camera only
  }

  void handleCameraObstacles(PerceptionObstacleArray::ConstSharedPtr msg)
  {
    cameraObstacles_ = msg;
    cameraObstaclesArrival_ = nowSeconds();
    if (lidarStale()) {obstaclesPub_->publish(*msg);}
  }

  bool lidarStale() {return lastCloudArrival_ < 0.0 || nowSeconds() - lastCloudArrival_ > lidarTimeout_;}

  void handleCloud(const PointCloud2 & msg)
  {
    const double arrival = nowSeconds();
    if (lastCloudArrival_ < 0.0 || arrival - lastCloudArrival_ > lidarTimeout_) {
      RCLCPP_INFO(get_logger(), "lidar cloud received: fusing camera and lidar");
    }
    lastCloudArrival_ = arrival;

    const double cloudTime = toSeconds(msg.header.stamp);   // Go2 clock
    const bool haveOdometry = lastOdometryArrival_ >= 0.0 && arrival - lastOdometryArrival_ <= odometryTimeout_;
    std::optional<Pose2> pose = haveOdometry ? poses_.at(cloudTime) : std::nullopt;
    const Pose2 robot = pose.value_or(Pose2{});
    // Output stamp on this computer's clock.
    const double outputTime = clockOffset_.known() ? cloudTime + clockOffset_.offset() : arrival;
    const auto stamp = toStamp(outputTime);

    // --- floor plane, obstacle points
    const auto cloud = readCloud(msg);
    if (cloud.empty() && msg.width * msg.height > 0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "cloud without float x/y/z fields, ignored");
      return;
    }
    // --- empty / nearly empty cloud (dropped data): skip it, repeat the last output so nothing blinks
    if (pose && !cloudGate_.accept(cloud.size())) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "cloud with %zu points (typical %.0f): skipped, last output repeated", cloud.size(), cloudGate_.typical());
      repeatLastOutput(stamp);
      return;
    }
    const GroundPlane plane = ground_.update(cloud, filter_);
    const FilterResult filtered = filterCloud(cloud, filter_, plane);

    // --- short memory in odom: only cells hit by several clouds count
    std::vector<Point2> odomPoints;
    odomPoints.reserve(filtered.points.size());
    for (const auto & p : filtered.points) {odomPoints.push_back(robot.toOdom(p));}
    if (!pose) {
      if (!warnedNoOdometry_) {
        RCLCPP_WARN(get_logger(), "no odometry for the cloud: no memory between clouds, velocities unknown");
        warnedNoOdometry_ = true;
      }
      hitGrid_.clear();
    }
    hitGrid_.add(odomPoints, cloudTime);

    // --- update every cloudsPerUpdate clouds (default every cloud; the tracker's missWindow already ignores an
    // object missing from a single cloud). The last scanClouds clouds are kept in odom for a point scan.
    if (pose) {
      recentClouds_.push_back(std::move(odomPoints));
      while (static_cast<int>(recentClouds_.size()) > std::max(cloudsPerUpdate_, scanClouds_)) {
        recentClouds_.pop_front();
      }
      if (++cloudsSinceUpdate_ < cloudsPerUpdate_) {return;}
    } else {
      recentClouds_.clear();
    }
    cloudsSinceUpdate_ = 0;
    const auto occupied = pose ? hitGrid_.occupiedStampedPoints(cloudTime) :
      hitGrid_.occupiedStampedPoints(cloudTime, 1);
    std::vector<StampedPoint> nearby;   // in range of the current base_link
    nearby.reserve(occupied.size());
    for (const auto & q : occupied) {
      const double r = norm(robot.toBase(q.p));
      if (r >= filter_.rangeMin && r <= filter_.rangeMax) {nearby.push_back(q);}
    }

    // --- clusters (grid in odom), tracks, what is steady enough to publish
    auto clusters = clusterPoints(nearby, clusterParams_, robot);
    tracker_.update(clusters, robot, pose.has_value(), cloudTime);
    const auto steady = tracker_.publishable(clusters, robot, cloudTime);

    // --- fuse with the camera
    FusionInput input = buildCameraInput(arrival, robot, pose.has_value());
    const FusionOutput fused = fusion_.fuse(input, steady, robot, cloudTime);

    lastLeader_.reset();
    if (input.haveCamera) {
      lastLeader_ = buildLeaderMessage(fused.leader, stamp);
      leaderPub_->publish(*lastLeader_);
    }
    lastObstacles_ = buildObstaclesMessage(fused, stamp);
    obstaclesPub_->publish(*lastObstacles_);
    std::vector<float> ranges;
    if (scanFromObstacles_) {
      ranges = tmp_lidar::buildScan(steady, scanRays_, filter_.rangeMin, filter_.rangeMax);
    } else {
      std::vector<Point2> scanPoints;   // the last scanClouds clouds, in the current base_link
      if (pose) {
        const int skip = static_cast<int>(recentClouds_.size()) - scanClouds_;
        for (std::size_t k = static_cast<std::size_t>(std::max(0, skip)); k < recentClouds_.size(); ++k) {
          for (const auto & q : recentClouds_[k]) {scanPoints.push_back(robot.toBase(q));}
        }
      } else {
        scanPoints = filtered.points;
      }
      ranges = tmp_lidar::buildScan(scanPoints, scanRays_, filter_.rangeMin, filter_.rangeMax);
    }
    lastScan_ = scanMessage(std::move(ranges), stamp);
    scanPub_->publish(*lastScan_);
    if (debugPub_) {debugPub_->publish(buildMarkers(steady, fused, plane, stamp));}

    std::size_t coasting = 0;
    for (const auto & c : steady) {coasting += c.isCoasting ? 1 : 0;}
    RCLCPP_DEBUG(get_logger(),
      "cloud %zu pts -> %zu kept (plane z0 %.3f tilt %.1f deg, %d inliers%s) -> %zu occupied pts -> "
      "%zu clusters -> %zu steady (%zu coasting), %zu obstacles",
      cloud.size(), filtered.points.size(), plane.c, plane.tiltDeg(), plane.inliers, plane.fitted ? "" : ", flat",
      nearby.size(), clusters.size(), steady.size(), coasting, fused.obstacles.size());
  }

  // The newest camera messages, moved from the camera's time to the cloud's (robot pose `robot`).
  FusionInput buildCameraInput(double arrival, const Pose2 & robot, bool havePose)
  {
    FusionInput input;
    const bool leaderFresh = cameraLeader_ && arrival - cameraLeaderArrival_ <= cameraTimeout_;
    const bool obstaclesFresh = cameraObstacles_ && arrival - cameraObstaclesArrival_ <= cameraTimeout_;
    input.haveCamera = leaderFresh || obstaclesFresh;
    if (!input.haveCamera) {return input;}

    // Pose at the camera's time (camera stamps are on this computer's clock).
    auto cameraPose = [&](const builtin_interfaces::msg::Time & cameraStamp) {
        if (!havePose || !clockOffset_.known()) {return robot;}
        return poses_.at(toSeconds(cameraStamp) - clockOffset_.offset()).value_or(robot);
      };

    if (leaderFresh) {
      const auto & m = *cameraLeader_;
      const Pose2 from = cameraPose(m.header.stamp);
      LeaderData & l = input.leader;
      l.status = m.status;
      l.closest = shiftPoint(fromMsg(m.closest_point), from, robot);
      l.velocity = shiftVector(fromMsg(m.velocity), from, robot);
      l.positionSigma = m.position_uncertainty;
      l.velocitySigma = m.velocity_uncertainty;
      l.lostDuration = m.lost_duration;
      l.lastKnown = shiftPoint(fromMsg(m.last_known_point), from, robot);
    }
    if (obstaclesFresh) {
      const auto & m = *cameraObstacles_;
      const Pose2 from = cameraPose(m.header.stamp);
      for (const auto & o : m.obstacles) {
        ObstacleData d;
        d.closest = shiftPoint(fromMsg(o.closest_point), from, robot);
        d.lineOfSight = shiftPoint(fromMsg(o.line_of_sight_point), from, robot);
        d.velocity = shiftVector(fromMsg(o.velocity), from, robot);
        d.positionSigma = o.position_uncertainty;
        d.velocitySigma = o.velocity_uncertainty;
        input.obstacles.push_back(d);
      }
    }
    return input;
  }

  PerceptionLeader buildLeaderMessage(const LeaderData & l, const builtin_interfaces::msg::Time & stamp)
  {
    PerceptionLeader msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = baseFrame_;
    msg.status = static_cast<uint8_t>(l.status);
    msg.closest_point = toPointMsg(l.closest);
    msg.velocity = toVectorMsg(l.velocity);
    msg.position_uncertainty = static_cast<float>(l.positionSigma);
    msg.velocity_uncertainty = static_cast<float>(l.velocitySigma);
    msg.lost_duration = static_cast<float>(l.lostDuration);
    msg.last_known_point = toPointMsg(l.lastKnown);
    return msg;
  }

  PerceptionObstacleArray buildObstaclesMessage(const FusionOutput & fused,
    const builtin_interfaces::msg::Time & stamp)
  {
    PerceptionObstacleArray msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = baseFrame_;
    msg.obstacles.reserve(fused.obstacles.size());
    for (const auto & o : fused.obstacles) {
      PerceptionObstacle out;
      out.closest_point = toPointMsg(o.data.closest);
      out.line_of_sight_point = toPointMsg(o.data.lineOfSight);
      out.velocity = toVectorMsg(o.data.velocity);
      out.position_uncertainty = static_cast<float>(o.data.positionSigma);
      out.velocity_uncertainty = static_cast<float>(o.data.velocitySigma);
      msg.obstacles.push_back(out);
    }
    return msg;
  }

  // The last published messages again, with a new stamp (a skipped cloud).
  void repeatLastOutput(const builtin_interfaces::msg::Time & stamp)
  {
    if (lastLeader_) {
      lastLeader_->header.stamp = stamp;
      leaderPub_->publish(*lastLeader_);
    }
    if (lastObstacles_) {
      lastObstacles_->header.stamp = stamp;
      obstaclesPub_->publish(*lastObstacles_);
    }
    if (lastScan_) {
      lastScan_->header.stamp = stamp;
      scanPub_->publish(*lastScan_);
    }
  }

  LaserScan scanMessage(std::vector<float> ranges, const builtin_interfaces::msg::Time & stamp)
  {
    LaserScan msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = baseFrame_;
    const double step = 2.0 * M_PI / scanRays_;
    msg.angle_min = static_cast<float>(-M_PI);
    msg.angle_increment = static_cast<float>(step);
    msg.angle_max = static_cast<float>(-M_PI + (scanRays_ - 1) * step);
    msg.range_min = static_cast<float>(filter_.rangeMin);
    msg.range_max = static_cast<float>(filter_.rangeMax);
    msg.ranges = std::move(ranges);
    return msg;
  }

  MarkerArray buildMarkers(const std::vector<Cluster> & clusters, const FusionOutput & fused,
    const GroundPlane & plane, const builtin_interfaces::msg::Time & stamp)
  {
    MarkerArray array;
    Marker clear;
    clear.action = Marker::DELETEALL;
    array.markers.push_back(clear);
    Marker text;
    text.header.stamp = stamp;
    text.header.frame_id = baseFrame_;
    text.ns = "floor";
    text.type = Marker::TEXT_VIEW_FACING;
    text.pose.position.z = 1.2;
    text.pose.orientation.w = 1.0;
    text.scale.z = 0.15;
    text.color.r = text.color.g = text.color.b = text.color.a = 1.0f;
    char buffer[96];
    std::snprintf(buffer, sizeof buffer, "floor z0 %.2f tilt %.1f deg%s", plane.c, plane.tiltDeg(),
      plane.fitted ? "" : " (flat)");
    text.text = buffer;
    array.markers.push_back(text);
    for (std::size_t i = 0; i < clusters.size(); ++i) {
      const Cluster & c = clusters[i];
      const int owner = fused.clusterOwner[i];
      Marker m;
      m.header.stamp = stamp;
      m.header.frame_id = baseFrame_;
      m.ns = "clusters";
      m.id = c.trackId >= 0 ? c.trackId : static_cast<int>(i);   // stable per track in RViz
      m.type = Marker::CYLINDER;
      m.pose.position.x = c.centroid.x;
      m.pose.position.y = c.centroid.y;
      m.pose.position.z = 0.5;
      m.pose.orientation.w = 1.0;
      const double d = std::max(0.1, 2.0 * c.radius);
      m.scale.x = d;
      m.scale.y = d;
      m.scale.z = 1.0;
      // Opacity: how sure the tracker is the object is real.
      m.color.a = std::isfinite(c.existence) ? static_cast<float>(0.2 + 0.6 * c.existence) : 0.6f;
      if (owner == kOwnerLeader) {
        m.color.b = 1.0f;                       // leader
      } else if (owner >= 0) {
        m.color.g = 1.0f;                       // camera person
      } else if (c.isCoasting) {
        m.color.r = 1.0f;                       // lidar only, missed this cloud
        m.color.g = 0.5f;
      } else {
        m.color.r = m.color.g = m.color.b = 0.6f;   // lidar only
      }
      array.markers.push_back(m);

      Marker closest = m;
      closest.ns = "closest";
      closest.type = Marker::SPHERE;
      closest.pose.position.x = c.closest.x;
      closest.pose.position.y = c.closest.y;
      closest.pose.position.z = 0.0;
      closest.scale.x = closest.scale.y = closest.scale.z = 0.08;
      closest.color.r = 1.0f;
      closest.color.g = closest.color.b = 0.0f;
      closest.color.a = 1.0f;
      array.markers.push_back(closest);

      Marker label = text;
      label.ns = "tracks";
      label.id = m.id;
      label.pose.position.x = c.centroid.x;
      label.pose.position.y = c.centroid.y;
      label.pose.position.z = 1.15;
      label.scale.z = 0.12;
      static const char * const kModes[] = {"static", "CV", "turn"};
      const char * mode = c.isChunk ? "wall" : (c.motionMode >= 0 && c.motionMode < 3 ? kModes[c.motionMode] : "?");
      std::snprintf(buffer, sizeof buffer, "#%d %s p=%.2f %.1f m/s", c.trackId, mode,
        std::isfinite(c.existence) ? c.existence : 0.0, norm(c.velocity));
      label.text = buffer;
      array.markers.push_back(label);
    }
    return array;
  }

  // parameters
  std::string baseFrame_;
  FilterParams filter_;
  ClusterParams clusterParams_;
  double cameraTimeout_ = 0.5;
  double lidarTimeout_ = 0.5;
  double odometryTimeout_ = 0.5;
  int scanRays_ = 720;
  int cloudsPerUpdate_ = 1;
  bool scanFromObstacles_ = true;
  int scanClouds_ = 2;
  bool publishDebug_ = true;

  // state
  ClockOffset clockOffset_;
  PoseHistory poses_;
  GroundEstimator ground_;
  HitGrid hitGrid_;
  std::deque<std::vector<Point2>> recentClouds_;   // odom obstacle points of the clouds since the last update
  int cloudsSinceUpdate_ = 0;
  CloudGate cloudGate_;
  std::optional<PerceptionLeader> lastLeader_;
  std::optional<PerceptionObstacleArray> lastObstacles_;
  std::optional<LaserScan> lastScan_;
  bool warnedNoOdometry_ = false;
  ClusterTracker tracker_;
  Fusion fusion_;
  double lastOdometryArrival_ = -1.0;
  double lastCloudArrival_ = -1.0;
  PerceptionLeader::ConstSharedPtr cameraLeader_;
  PerceptionObstacleArray::ConstSharedPtr cameraObstacles_;
  double cameraLeaderArrival_ = 0.0;
  double cameraObstaclesArrival_ = 0.0;

  // ROS
  rclcpp::Subscription<Odometry>::SharedPtr odometrySub_;
  rclcpp::Subscription<PointCloud2>::SharedPtr cloudSub_;
  rclcpp::Subscription<PerceptionLeader>::SharedPtr cameraLeaderSub_;
  rclcpp::Subscription<PerceptionObstacleArray>::SharedPtr cameraObstaclesSub_;
  rclcpp::Publisher<PerceptionLeader>::SharedPtr leaderPub_;
  rclcpp::Publisher<PerceptionObstacleArray>::SharedPtr obstaclesPub_;
  rclcpp::Publisher<LaserScan>::SharedPtr scanPub_;
  rclcpp::Publisher<MarkerArray>::SharedPtr debugPub_;
};

}  // namespace tmp_lidar

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<tmp_lidar::TmpLidarNode>());
  rclcpp::shutdown();
  return 0;
}
