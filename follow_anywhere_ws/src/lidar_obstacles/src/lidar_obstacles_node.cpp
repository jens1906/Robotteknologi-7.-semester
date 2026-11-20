// lidar_obstacles_node: the lidar part of perception, fused with the camera part. The only file with ROS I/O.
//
// Subscribes:
//   /utlidar/cloud_base           sensor_msgs/PointCloud2  base_link, Go2 clock, ~15 Hz
//   /utlidar/robot_odom           nav_msgs/Odometry        odom -> base_link, Go2 clock
//   /perception/camera/leader     follow_anywhere_interfaces/PerceptionLeader         (perception_node, remapped)
//   /perception/camera/obstacles  follow_anywhere_interfaces/PerceptionObstacleArray  (perception_node, remapped)
// Publishes (what navigation reads; same topics and types as 2dSim's fake_perception):
//   /perception/leader              PerceptionLeader         enriched with the lidar
//   /perception/obstacles_static    PerceptionObstacleArray  lidar-only clusters (no camera person, not the leader)
//   /perception/obstacles_combined  PerceptionObstacleArray  camera people (enriched) + lidar-only clusters, no duplicates
//   /perception/scan                sensor_msgs/LaserScan    360 deg, obstacle points of the newest cloud
//   /perception/lidar_markers       visualization_msgs/MarkerArray  clusters (grey static, green person, blue leader)
//   /perception/debug               std_msgs/String          'LidarObstacles: <what happened>'
// Topic names are written where the subscriptions and publishers are created: change them with a remapping.
//
// Time: the Go2 stamps the cloud and odometry with its own clock (~38 s off the PC). Cloud and odometry are
// matched on the Go2 clock directly. The offset Go2 -> this computer is the smallest (arrival - stamp) over
// clockOffsetWindow (as perception's ClockOffsetEstimator); with it the camera messages (stamped on this
// computer) are moved to the cloud's time, and the outputs are stamped on this computer's clock, because
// navigation measures message age against its own now().
// The fusion runs once per cloud. A nearly empty cloud is skipped and the last outputs are repeated (CloudGate).
// /perception/scan is made from the published obstacles. While no cloud arrived for lidarTimeout, camera messages
// are passed through unchanged, so the system still works without the lidar.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
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
#include "std_msgs/msg/string.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "lidar_obstacles/cloud_filter.hpp"
#include "lidar_obstacles/cluster_tracker.hpp"
#include "lidar_obstacles/clustering.hpp"
#include "lidar_obstacles/fusion.hpp"

namespace lidar_obstacles
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

// Pose history
// How many seconds of odometry poses are kept, and how far outside the stored poses a lookup may be.
constexpr double POSE_HISTORY_SECONDS = 3.0;
constexpr double POSE_MAX_GAP_SECONDS = 0.5;   // a lookup farther than this from any sample fails

// Debug
// The same debug message is published at most once per this many seconds.
constexpr double DEBUG_REPEAT_SECONDS = 2.0;

// Markers
// Names of the motion models (ImmFilter::STATIC_MODEL, ...) shown in the RViz track labels.
const char * const MOTION_MODE_NAMES[] = {"static", "CV", "turn"};

// One odometry pose and the Go2-clock time it was measured at.
struct PoseSample
{
  double seconds = 0.0;
  Pose2 pose;
};

// One clock offset measurement: when a message arrived (this computer's clock) and arrival - stamp.
struct ClockSample
{
  double arrivalSeconds = 0.0;
  double offsetSeconds = 0.0;
};

// ROS time stamp (seconds + nanoseconds) -> seconds as one double.
double ConvertStampToSeconds(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<double>(stamp.sec) + 1e-9 * static_cast<double>(stamp.nanosec);
}

// Seconds as one double -> ROS time stamp.
builtin_interfaces::msg::Time ConvertSecondsToStamp(double seconds)
{
  return rclcpp::Time(static_cast<int64_t>(std::llround(seconds * 1e9)), RCL_ROS_TIME);
}

// Heading (yaw, rotation around z) from a quaternion orientation, standard formula.
double CalculateHeading(const geometry_msgs::msg::Quaternion & orientation)
{
  return std::atan2(2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
    1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z));
}

// Brings an angle back into -pi..pi (e.g. 3.5 rad -> -2.78 rad), so the difference between two headings
// across +-pi is the short way round.
double WrapAngle(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

// A number as text with one decimal, for the debug messages.
std::string FormatOneDecimal(double value)
{
  char textBuffer[32];
  std::snprintf(textBuffer, sizeof textBuffer, "%.1f", value);
  return textBuffer;
}

// ROS point / vector message -> Point2 (z is dropped: everything is on the floor plane).
Point2 ConvertPointMessage(const geometry_msgs::msg::Point & pointMessage) {return {pointMessage.x, pointMessage.y};}
Point2 ConvertVectorMessage(const geometry_msgs::msg::Vector3 & vectorMessage)
{
  return {vectorMessage.x, vectorMessage.y};
}

// Point2 -> ROS point message. z = 0 for a real point; for an unknown point z is NaN too, so the whole
// message clearly says "no point".
geometry_msgs::msg::Point MakePointMessage(const Point2 & point)
{
  geometry_msgs::msg::Point pointMessage;
  pointMessage.x = point.x;
  pointMessage.y = point.y;
  pointMessage.z = IsFinite(point) ? 0.0 : NOT_A_NUMBER;
  return pointMessage;
}

// Point2 velocity -> ROS vector message (z = 0).
geometry_msgs::msg::Vector3 MakeVectorMessage(const Point2 & vector)
{
  geometry_msgs::msg::Vector3 vectorMessage;
  vectorMessage.x = vector.x;
  vectorMessage.y = vector.y;
  vectorMessage.z = 0.0;
  return vectorMessage;
}

// x, y, z of a PointCloud2 (float32 or float64 fields), in the cloud's frame.
// A PointCloud2 is a block of raw bytes: every point is point_step bytes long, and the "fields" list says at
// which byte offset (and in which number type) x, y and z sit inside a point. Returns an empty list when the
// message has no usable x/y/z fields or is shorter than it claims.
std::vector<Point3> ReadCloudPoints(const PointCloud2 & cloudMessage)
{
  // Find the offset and type of the x, y and z fields.
  int fieldOffsets[3] = {-1, -1, -1};
  uint8_t fieldTypes[3] = {0, 0, 0};
  const char * fieldNames[3] = {"x", "y", "z"};
  for (const auto & field : cloudMessage.fields) {
    for (int axis = 0; axis < 3; ++axis) {
      if (field.name == fieldNames[axis]) {
        fieldOffsets[axis] = static_cast<int>(field.offset);
        fieldTypes[axis] = field.datatype;
      }
    }
  }
  // All three must exist and be 32- or 64-bit floats.
  std::vector<Point3> cloudPoints;
  for (int axis = 0; axis < 3; ++axis) {
    if (fieldOffsets[axis] < 0 || (fieldTypes[axis] != PointField::FLOAT32 && fieldTypes[axis] != PointField::FLOAT64)) {
      return cloudPoints;
    }
  }
  // The data must be at least as long as width * height points.
  const std::size_t pointCount = static_cast<std::size_t>(cloudMessage.width) * cloudMessage.height;
  if (cloudMessage.point_step == 0 || cloudMessage.data.size() < pointCount * cloudMessage.point_step) {
    return cloudPoints;
  }
  cloudPoints.reserve(pointCount);
  // Reads one field of one point. memcpy is used instead of casting the byte pointer, because the bytes may not
  // be aligned for a float/double.
  auto ReadField = [&](const uint8_t * pointStart, int axis) {
      if (fieldTypes[axis] == PointField::FLOAT32) {
        float floatValue;
        std::memcpy(&floatValue, pointStart + fieldOffsets[axis], sizeof floatValue);
        return static_cast<double>(floatValue);
      }
      double doubleValue;
      std::memcpy(&doubleValue, pointStart + fieldOffsets[axis], sizeof doubleValue);
      return doubleValue;
    };
  for (std::size_t pointIndex = 0; pointIndex < pointCount; ++pointIndex) {
    const uint8_t * pointStart = cloudMessage.data.data() + pointIndex * cloudMessage.point_step;
    cloudPoints.push_back({ReadField(pointStart, 0), ReadField(pointStart, 1), ReadField(pointStart, 2)});
  }
  return cloudPoints;
}

// Robot poses on the Go2 clock, interpolated.
// Odometry arrives at ~130 Hz, the lidar at ~15 Hz and the camera at its own times. This keeps the last
// POSE_HISTORY_SECONDS of poses so the node can ask "where was the robot at time t?" for any of them.
class PoseHistory
{
public:
  // Stores a pose. Time going backwards (a looped rosbag) clears the history; old poses are dropped.
  void Add(double seconds, const Pose2 & pose)
  {
    if (!samples.empty() && seconds < samples.back().seconds) {
      samples.clear();
    }
    samples.push_back({seconds, pose});
    while (!samples.empty() && samples.front().seconds < seconds - POSE_HISTORY_SECONDS) {
      samples.pop_front();
    }
  }

  // The robot pose at `seconds`. Between two stored poses it is interpolated in a straight line (the heading
  // the short way round). Before the first / after the last stored pose, that pose is used if it is at most
  // POSE_MAX_GAP_SECONDS away; otherwise there is no pose (std::nullopt).
  std::optional<Pose2> FindPose(double seconds) const
  {
    if (samples.empty()) {
      return std::nullopt;
    }
    if (seconds <= samples.front().seconds) {
      return samples.front().seconds - seconds <= POSE_MAX_GAP_SECONDS ?
             std::optional<Pose2>(samples.front().pose) : std::nullopt;
    }
    if (seconds >= samples.back().seconds) {
      return seconds - samples.back().seconds <= POSE_MAX_GAP_SECONDS ?
             std::optional<Pose2>(samples.back().pose) : std::nullopt;
    }
    // Find the first stored pose at or after `seconds`, and blend it with the one before.
    for (std::size_t sampleIndex = 1; sampleIndex < samples.size(); ++sampleIndex) {
      if (samples[sampleIndex].seconds < seconds) {
        continue;
      }
      const PoseSample & earlierSample = samples[sampleIndex - 1];
      const PoseSample & laterSample = samples[sampleIndex];
      const Pose2 & earlierPose = earlierSample.pose;
      const Pose2 & laterPose = laterSample.pose;
      const double fraction = laterSample.seconds > earlierSample.seconds ?
        (seconds - earlierSample.seconds) / (laterSample.seconds - earlierSample.seconds) : 0.0;
      return Pose2{earlierPose.x + fraction * (laterPose.x - earlierPose.x),
        earlierPose.y + fraction * (laterPose.y - earlierPose.y),
        earlierPose.yaw + fraction * WrapAngle(laterPose.yaw - earlierPose.yaw)};
    }
    return samples.back().pose;
  }

  void Clear() {samples.clear();}

private:
  std::deque<PoseSample> samples;   // oldest first
};

// Go2 clock -> this computer's clock: smallest (arrival - stamp) over a window.
// arrival - stamp = clock difference + network delay. The delay is never negative, so the smallest value seen
// recently is the best estimate of the pure clock difference. The deque keeps only the values that can still
// become the smallest ("monotonic queue"), so the minimum is always at the front.
class ClockOffset
{
public:
  explicit ClockOffset(double offsetWindowSeconds)
  : windowSeconds(offsetWindowSeconds) {}

  // Adds one measurement: drop older samples that are larger (they can never be the minimum again), then drop
  // samples that fell out of the time window.
  void Add(double stampSeconds, double arrivalSeconds)
  {
    const double offsetSeconds = arrivalSeconds - stampSeconds;
    while (!samples.empty() && samples.back().offsetSeconds >= offsetSeconds) {
      samples.pop_back();
    }
    samples.push_back({arrivalSeconds, offsetSeconds});
    while (samples.front().arrivalSeconds < arrivalSeconds - windowSeconds) {
      samples.pop_front();
    }
  }

  // True once at least one odometry message gave an offset.
  bool IsKnown() const {return !samples.empty();}
  // The current offset: add it to a Go2 stamp to get this computer's time.
  double GetOffset() const {return samples.empty() ? 0.0 : samples.front().offsetSeconds;}
  void Reset() {samples.clear();}

private:
  double windowSeconds;
  std::deque<ClockSample> samples;   // offsets increasing
};

}  // namespace

// The ROS node. Order: constructor (parameters + topics), callbacks (one per input topic), logic functions.
class LidarObstaclesNode : public rclcpp::Node
{
public:
  // Reads every parameter (default value = the one in the settings structs, overwritten by the yaml), builds
  // the processing objects with them, and creates the subscriptions and publishers.
  LidarObstaclesNode()
  : Node("lidar_obstacles"), clockOffset(declare_parameter("clockOffsetWindow", 5.0))
  {
    DeclareParameter("baseFrame", baseFrame);

    // Cloud filter parameters
    // The yaml gives the body box as [x, y] lists; a list that is too short keeps the default.
    DeclareParameter("floorZ", filterParams.floorZ);
    DeclareParameter("floorSearch", filterParams.floorSearch);
    DeclareParameter("floorFitRange", filterParams.floorFitRange);
    DeclareParameter("floorMaxTilt", filterParams.floorMaxTiltDeg);
    DeclareParameter("floorInlierBand", filterParams.floorInlierBand);
    DeclareParameter("floorMinimumPoints", filterParams.floorMinimumPoints);
    DeclareParameter("groundHeight", filterParams.groundHeight);
    DeclareParameter("nearBodyRange", filterParams.nearBodyRange);
    DeclareParameter("nearBodyGroundHeight", filterParams.nearBodyGroundHeight);
    DeclareParameter("maxHeight", filterParams.maxHeight);
    DeclareParameter("bodyBoxMin", filterParams.bodyMin.x, filterParams.bodyMin.y);
    DeclareParameter("bodyBoxMax", filterParams.bodyMax.x, filterParams.bodyMax.y);
    DeclareParameter("rangeMin", filterParams.rangeMin);
    DeclareParameter("obstacleMaxRange", filterParams.rangeMax);

    // Hit grid parameters (short memory in odom)
    HitGridParams gridParams;
    DeclareParameter("hitCellSize", gridParams.cellSize);
    DeclareParameter("memorySeconds", gridParams.memorySeconds);
    DeclareParameter("minCellHits", gridParams.minCellHits);
    hitGrid = HitGrid(gridParams);

    // Clustering parameters
    DeclareParameter("cellSize", clusterParams.cellSize);
    DeclareParameter("linkDistance", clusterParams.linkDistance);
    DeclareParameter("linkAngle", clusterParams.linkAngle);
    DeclareParameter("minClusterPoints", clusterParams.minClusterPoints);
    DeclareParameter("minClusterPointsFar", clusterParams.minClusterPointsFar);
    clusterParams.farRange = filterParams.rangeMax;
    DeclareParameter("maxClusterRadius", clusterParams.maxClusterRadius);
    DeclareParameter("wallChunk", clusterParams.chunkSize);

    // Tracking parameters
    TrackerParams trackerParams;
    DeclareParameter("gateChi2", trackerParams.gateChi2);
    DeclareParameter("minimumGate", trackerParams.minimumGate);
    DeclareParameter("maximumGate", trackerParams.maximumGate);
    DeclareParameter("mergeRadius", trackerParams.mergeRadius);
    DeclareParameter("shapeSigmaScale", trackerParams.shapeSigmaScale);
    DeclareParameter("noiseAdaptRate", trackerParams.noiseAdaptRate);
    DeclareParameter("maxNoiseScale", trackerParams.maxNoiseScale);
    DeclareParameter("velocitySightings", trackerParams.velocitySightings);
    DeclareParameter("staticPositionNoise", trackerParams.imm.staticPositionNoise);
    DeclareParameter("cvAccelerationNoise", trackerParams.imm.cvAccelerationNoise);
    DeclareParameter("ctAccelerationNoise", trackerParams.imm.ctAccelerationNoise);
    DeclareParameter("ctTurnNoise", trackerParams.imm.ctTurnNoise);
    DeclareParameter("modeSwitchRate", trackerParams.imm.modeSwitchRate);
    DeclareParameter("minimumSpeed", trackerParams.minimumSpeed);
    DeclareParameter("staticPublishProbability", trackerParams.staticPublishProbability);
    DeclareParameter("unknownVelocitySigma", trackerParams.unknownVelocitySigma);
    DeclareParameter("minimumVelocitySigma", trackerParams.minimumVelocitySigma);
    DeclareParameter("staticVelocitySigma", trackerParams.staticVelocitySigma);
    DeclareParameter("missWindow", trackerParams.missWindow);
    DeclareParameter("birthExistence", trackerParams.birthExistence);
    DeclareParameter("maxExistence", trackerParams.maxExistence);
    DeclareParameter("confirmExistence", trackerParams.confirmExistence);
    DeclareParameter("keepExistence", trackerParams.keepExistence);
    DeclareParameter("deleteExistence", trackerParams.deleteExistence);
    DeclareParameter("detectionProbabilityNear", trackerParams.detectionProbabilityNear);
    DeclareParameter("detectionProbabilityFar", trackerParams.detectionProbabilityFar);
    DeclareParameter("detectionNearRange", trackerParams.detectionNearRange);
    DeclareParameter("detectionFarRange", trackerParams.detectionFarRange);
    DeclareParameter("falseAlarmProbability", trackerParams.falseAlarmProbability);
    DeclareParameter("falseAlarmPoints", trackerParams.falseAlarmPoints);
    DeclareParameter("occludedDetectionFactor", trackerParams.occludedDetectionFactor);
    DeclareParameter("occlusionMargin", trackerParams.occlusionMargin);
    DeclareParameter("trackTimeout", trackerParams.trackTimeout);
    DeclareParameter("occludedTimeout", trackerParams.occludedTimeout);
    DeclareParameter("maxPositionSigma", trackerParams.maxPositionSigma);
    // The tracker's view range is the same as the filter's (no separate parameter).
    trackerParams.rangeMin = filterParams.rangeMin;
    trackerParams.rangeMax = filterParams.rangeMax;

    // Fusion parameters
    FusionParams fusionParams;
    DeclareParameter("matchGate", fusionParams.matchGate);
    DeclareParameter("matchSigmas", fusionParams.matchSigmas);
    DeclareParameter("stickyBonus", fusionParams.stickyBonus);
    DeclareParameter("leaderHoldTime", fusionParams.leaderHoldTime);
    DeclareParameter("lidarRangeStd", fusionParams.lidarRangeStd[0], fusionParams.lidarRangeStd[1]);
    DeclareParameter("minimumPositionUncertainty", fusionParams.minimumPositionUncertainty);
    DeclareParameter("cameraPosition", fusionParams.cameraPosition.x, fusionParams.cameraPosition.y);
    fusion = Fusion(fusionParams);
    // The tracker uses the same lidar range noise as the fusion, so it is only set once (lidarRangeStd).
    trackerParams.lidarRangeStd = fusionParams.lidarRangeStd;
    tracker = ClusterTracker(trackerParams);
    // How old camera messages, lidar clouds and odometry may be before they count as missing.
    DeclareParameter("cameraTimeout", cameraTimeout);
    DeclareParameter("lidarTimeout", lidarTimeout);
    DeclareParameter("odometryTimeout", odometryTimeout);

    // Scan and marker parameters
    // scanRays: number of rays in /perception/scan (720 = 0.5 deg each). The cloud gate settings follow.
    DeclareParameter("scanRays", scanRays);
    CloudGateParams gateParams;
    DeclareParameter("emptyCloudFraction", gateParams.minFraction);
    DeclareParameter("emptyCloudPoints", gateParams.minPoints);
    DeclareParameter("maxSkippedClouds", gateParams.maxSkipped);
    cloudGate = CloudGate(gateParams);
    DeclareParameter("publishMarkers", publishMarkers);

    // Input topics (odometry first, so each executor cycle has the newest pose before the cloud that uses it)
    // SensorDataQoS = best effort, matching how the Go2 publishes. The cloud keeps only the newest message
    // (keep_last(1)): when processing is slow, old clouds are dropped instead of queued.
    odometrySubscription = create_subscription<Odometry>("/utlidar/robot_odom",
        rclcpp::SensorDataQoS().keep_last(5),
        [this](Odometry::ConstSharedPtr odometryMessage) {HandleOdometry(*odometryMessage);});
    cloudSubscription = create_subscription<PointCloud2>("/utlidar/cloud_base",
        rclcpp::SensorDataQoS().keep_last(1),
        [this](PointCloud2::ConstSharedPtr cloudMessage) {HandleCloud(*cloudMessage);});
    cameraLeaderSubscription = create_subscription<PerceptionLeader>("/perception/camera/leader", 10,
        [this](PerceptionLeader::ConstSharedPtr leaderMessage) {HandleCameraLeader(leaderMessage);});
    cameraObstaclesSubscription = create_subscription<PerceptionObstacleArray>("/perception/camera/obstacles", 10,
        [this](PerceptionObstacleArray::ConstSharedPtr obstaclesMessage) {HandleCameraObstacles(obstaclesMessage);});

    // Output topics
    // The markers publisher is only created when publishMarkers is true (saves the work when nobody uses RViz).
    leaderPublisher = create_publisher<PerceptionLeader>("/perception/leader", 10);
    staticObstaclesPublisher = create_publisher<PerceptionObstacleArray>("/perception/obstacles_static", 10);
    combinedObstaclesPublisher = create_publisher<PerceptionObstacleArray>("/perception/obstacles_combined", 10);
    scanPublisher = create_publisher<LaserScan>("/perception/scan", 10);
    debugPublisher = create_publisher<std_msgs::msg::String>("/perception/debug", 10);
    if (publishMarkers) {
      markersPublisher = create_publisher<MarkerArray>("/perception/lidar_markers", 10);
    }

    RCLCPP_INFO(get_logger(), "LidarObstacles: started");
  }

private:
  // /utlidar/robot_odom: stores the robot pose (for moving things between odom and base_link) and updates the
  // Go2 clock offset. If odometry was missing for more than odometryTimeout, everything that depends on it
  // (poses, the hit grid in odom, the tracks) is restarted, because the robot may have moved in the gap.
  void HandleOdometry(const Odometry & odometryMessage)
  {
    const double arrivalSeconds = GetNowSeconds();
    if (lastOdometryArrival >= 0.0 && arrivalSeconds - lastOdometryArrival > odometryTimeout) {
      PublishDebugMessage("odometry back after " + FormatOneDecimal(arrivalSeconds - lastOdometryArrival) +
        " s, tracks restarted");
      clockOffset.Reset();
      poseHistory.Clear();
      hitGrid.Clear();
      tracker.Reset();
      cloudGate.Reset();
    }
    lastOdometryArrival = arrivalSeconds;
    const double stampSeconds = ConvertStampToSeconds(odometryMessage.header.stamp);
    clockOffset.Add(stampSeconds, arrivalSeconds);
    const auto & position = odometryMessage.pose.pose.position;
    poseHistory.Add(stampSeconds, Pose2{position.x, position.y, CalculateHeading(odometryMessage.pose.pose.orientation)});
  }

  // /perception/camera/leader: keeps the newest camera leader for the next cloud. While the lidar is missing it
  // is passed straight through to /perception/leader, so following still works without the lidar.
  void HandleCameraLeader(PerceptionLeader::ConstSharedPtr leaderMessage)
  {
    cameraLeader = leaderMessage;
    cameraLeaderArrival = GetNowSeconds();
    if (IsLidarStale()) {
      leaderPublisher->publish(*leaderMessage);
    }
  }

  // /perception/camera/obstacles: keeps the newest camera people for the next cloud. While the lidar is
  // missing, they are passed through as the combined obstacles and an empty static list is published.
  void HandleCameraObstacles(PerceptionObstacleArray::ConstSharedPtr obstaclesMessage)
  {
    cameraObstacles = obstaclesMessage;
    cameraObstaclesArrival = GetNowSeconds();
    if (IsLidarStale()) {
      combinedObstaclesPublisher->publish(*obstaclesMessage);
      PerceptionObstacleArray emptyObstacles;
      emptyObstacles.header = obstaclesMessage->header;
      staticObstaclesPublisher->publish(emptyObstacles);
    }
  }

  // /utlidar/cloud_base: the main pipeline, run once per lidar cloud:
  // read points -> empty-cloud check -> floor fit + filter -> hit grid -> clusters -> tracker -> fusion with
  // the camera -> publish leader, obstacles, scan, markers.
  void HandleCloud(const PointCloud2 & cloudMessage)
  {
    // Log once when the lidar starts (or comes back after lidarTimeout).
    const double arrivalSeconds = GetNowSeconds();
    if (lastCloudArrival < 0.0 || arrivalSeconds - lastCloudArrival > lidarTimeout) {
      RCLCPP_INFO(get_logger(), "LidarObstacles: lidar cloud received, fusing camera and lidar");
    }
    lastCloudArrival = arrivalSeconds;

    // The robot pose at the cloud's time. Without recent odometry there is none, and `robot` is the zero pose:
    // then odom and base_link are treated as the same frame for this cloud.
    const double cloudSeconds = ConvertStampToSeconds(cloudMessage.header.stamp);   // Go2 clock
    const bool hasOdometry = lastOdometryArrival >= 0.0 && arrivalSeconds - lastOdometryArrival <= odometryTimeout;
    std::optional<Pose2> robotPose = hasOdometry ? poseHistory.FindPose(cloudSeconds) : std::nullopt;
    const Pose2 robot = robotPose.value_or(Pose2{});
    // Output stamp on this computer's clock.
    const double outputSeconds = clockOffset.IsKnown() ? cloudSeconds + clockOffset.GetOffset() : arrivalSeconds;
    const auto outputStamp = ConvertSecondsToStamp(outputSeconds);

    // --- floor plane, obstacle points
    const auto cloudPoints = ReadCloudPoints(cloudMessage);
    if (cloudPoints.empty() && cloudMessage.width * cloudMessage.height > 0) {
      PublishDebugMessage("cloud without float x/y/z fields, ignored");
      return;
    }
    // --- empty / nearly empty cloud (dropped data): skip it, repeat the last output so nothing blinks
    if (robotPose && !cloudGate.Accept(cloudPoints.size())) {
      PublishDebugMessage("nearly empty cloud skipped, last output repeated", " (" +
        std::to_string(cloudPoints.size()) + " points, typical " +
        std::to_string(std::lround(cloudGate.CalculateTypicalCount())) + ")");
      RepeatLastOutput(outputStamp);
      return;
    }
    // Fit the floor, then keep only the points that stand up from it.
    const GroundPlane floorPlane = groundEstimator.Update(cloudPoints, filterParams);
    const std::vector<Point2> obstaclePoints = FilterCloud(cloudPoints, filterParams, floorPlane);

    // --- short memory in odom: only cells hit by several clouds count
    // The obstacle points are moved to odom and added to the hit grid. Without odometry the grid cannot be kept
    // between clouds (the robot's motion is unknown), so it is cleared and holds only this cloud.
    std::vector<Point2> odomPoints;
    odomPoints.reserve(obstaclePoints.size());
    for (const auto & point : obstaclePoints) {
      odomPoints.push_back(robot.ConvertToOdom(point));
    }
    if (!robotPose) {
      if (!warnedNoOdometry) {
        PublishDebugMessage("no odometry for the cloud, no memory between clouds, velocities unknown");
        warnedNoOdometry = true;
      }
      hitGrid.Clear();
    }
    hitGrid.Add(odomPoints, cloudSeconds);

    // The points of the occupied cells (with no odometry 1 hit is enough, since there is only one cloud),
    // limited to the obstacle range around where the robot is now.
    const auto occupiedPoints = robotPose ? hitGrid.FindOccupiedPoints(cloudSeconds) :
      hitGrid.FindOccupiedPoints(cloudSeconds, 1);
    std::vector<Point2> pointsInRange;
    pointsInRange.reserve(occupiedPoints.size());
    for (const auto & occupiedPoint : occupiedPoints) {
      const double range = CalculateLength(robot.ConvertToBase(occupiedPoint));
      if (range >= filterParams.rangeMin && range <= filterParams.rangeMax) {
        pointsInRange.push_back(occupiedPoint);
      }
    }

    // --- clusters (grid in odom), tracks, what is steady enough to publish
    // steadyClusters = confirmed tracks seen now + confirmed tracks briefly missed (coasting).
    auto clusters = ClusterPoints(pointsInRange, clusterParams, robot);
    tracker.Update(clusters, robot, robotPose.has_value(), cloudSeconds);
    const auto steadyClusters = tracker.FindPublishable(clusters, robot, cloudSeconds);

    // --- fuse with the camera
    FusionInput fusionInput = BuildCameraInput(arrivalSeconds, robot, robotPose.has_value());
    const FusionOutput fused = fusion.Fuse(fusionInput, steadyClusters, robot, cloudSeconds);

    // Publish everything, and keep a copy of each message so a skipped cloud can repeat it.
    // The leader is only published when there is camera data (the lidar alone does not know who the leader is).
    lastLeader.reset();
    if (fusionInput.hasCamera) {
      lastLeader = BuildLeaderMessage(fused.leader, outputStamp);
      leaderPublisher->publish(*lastLeader);
    }
    lastStaticObstacles = BuildObstaclesMessage(fused, outputStamp, true);
    staticObstaclesPublisher->publish(*lastStaticObstacles);
    lastCombinedObstacles = BuildObstaclesMessage(fused, outputStamp, false);
    combinedObstaclesPublisher->publish(*lastCombinedObstacles);
    lastScan = BuildScanMessage(BuildScan(steadyClusters, scanRays, filterParams.rangeMin, filterParams.rangeMax),
      outputStamp);
    scanPublisher->publish(*lastScan);
    if (markersPublisher) {
      markersPublisher->publish(BuildMarkers(steadyClusters, fused, floorPlane, outputStamp));
    }

    // One summary line per cloud at DEBUG log level (shown with --ros-args --log-level debug).
    std::size_t coastingCount = 0;
    for (const auto & cluster : steadyClusters) {
      coastingCount += cluster.isCoasting ? 1 : 0;
    }
    RCLCPP_DEBUG(get_logger(),
      "cloud %zu pts -> %zu kept (plane z0 %.3f tilt %.1f deg, %d inliers%s) -> %zu occupied pts -> "
      "%zu clusters -> %zu steady (%zu coasting), %zu obstacles",
      cloudPoints.size(), obstaclePoints.size(), floorPlane.heightAtOrigin, floorPlane.CalculateTiltDegrees(),
      floorPlane.inliers, floorPlane.fitted ? "" : ", flat", pointsInRange.size(), clusters.size(),
      steadyClusters.size(), coastingCount, fused.obstacles.size());
  }

  // This computer's current time (or the simulation time with use_sim_time), in seconds.
  double GetNowSeconds() {return now().seconds();}

  // True when no lidar cloud arrived yet, or the last one is older than lidarTimeout.
  bool IsLidarStale()
  {
    return lastCloudArrival < 0.0 || GetNowSeconds() - lastCloudArrival > lidarTimeout;
  }

  // The newest camera messages, moved from the camera's time to the cloud's (robot pose `robot`).
  // Only messages younger than cameraTimeout are used; with none, hasCamera is false and the fusion runs
  // lidar-only.
  FusionInput BuildCameraInput(double arrivalSeconds, const Pose2 & robot, bool hasRobotPose)
  {
    FusionInput fusionInput;
    const bool leaderFresh = cameraLeader && arrivalSeconds - cameraLeaderArrival <= cameraTimeout;
    const bool obstaclesFresh = cameraObstacles && arrivalSeconds - cameraObstaclesArrival <= cameraTimeout;
    fusionInput.hasCamera = leaderFresh || obstaclesFresh;
    if (!fusionInput.hasCamera) {
      return fusionInput;
    }

    // Pose at the camera's time (camera stamps are on this computer's clock).
    // The camera stamp is turned into Go2 time (minus the clock offset) to look it up in the pose history.
    // Without odometry or a known offset, the robot is assumed not to have moved (cameraPose = robot).
    auto FindCameraPose = [&](const builtin_interfaces::msg::Time & cameraStamp) {
        if (!hasRobotPose || !clockOffset.IsKnown()) {
          return robot;
        }
        return poseHistory.FindPose(ConvertStampToSeconds(cameraStamp) - clockOffset.GetOffset()).value_or(robot);
      };

    // Leader: copy the message into LeaderData, moving its points and velocity from the camera's moment to the
    // cloud's moment.
    if (leaderFresh) {
      const PerceptionLeader & leaderMessage = *cameraLeader;
      const Pose2 cameraPose = FindCameraPose(leaderMessage.header.stamp);
      LeaderData & leader = fusionInput.leader;
      leader.status = leaderMessage.status;
      leader.closest = ShiftPointBetweenPoses(ConvertPointMessage(leaderMessage.closest_point), cameraPose, robot);
      leader.velocity = RotateVectorBetweenPoses(ConvertVectorMessage(leaderMessage.velocity), cameraPose, robot);
      leader.positionSigma = leaderMessage.position_uncertainty;
      leader.velocitySigma = leaderMessage.velocity_uncertainty;
      leader.lostDuration = leaderMessage.lost_duration;
      leader.lastKnown =
        ShiftPointBetweenPoses(ConvertPointMessage(leaderMessage.last_known_point), cameraPose, robot);
    }
    // Camera people: the same, for each obstacle in the message.
    if (obstaclesFresh) {
      const PerceptionObstacleArray & obstaclesMessage = *cameraObstacles;
      const Pose2 cameraPose = FindCameraPose(obstaclesMessage.header.stamp);
      for (const auto & obstacleMessage : obstaclesMessage.obstacles) {
        ObstacleData obstacle;
        obstacle.closest = ShiftPointBetweenPoses(ConvertPointMessage(obstacleMessage.closest_point), cameraPose, robot);
        obstacle.lineOfSight =
          ShiftPointBetweenPoses(ConvertPointMessage(obstacleMessage.line_of_sight_point), cameraPose, robot);
        obstacle.velocity = RotateVectorBetweenPoses(ConvertVectorMessage(obstacleMessage.velocity), cameraPose, robot);
        obstacle.positionSigma = obstacleMessage.position_uncertainty;
        obstacle.velocitySigma = obstacleMessage.velocity_uncertainty;
        fusionInput.obstacles.push_back(obstacle);
      }
    }
    return fusionInput;
  }

  // LeaderData -> PerceptionLeader message (base_link, stamped on this computer's clock).
  PerceptionLeader BuildLeaderMessage(const LeaderData & leader, const builtin_interfaces::msg::Time & stamp)
  {
    PerceptionLeader leaderMessage;
    leaderMessage.header.stamp = stamp;
    leaderMessage.header.frame_id = baseFrame;
    leaderMessage.status = static_cast<uint8_t>(leader.status);
    leaderMessage.closest_point = MakePointMessage(leader.closest);
    leaderMessage.velocity = MakeVectorMessage(leader.velocity);
    leaderMessage.position_uncertainty = static_cast<float>(leader.positionSigma);
    leaderMessage.velocity_uncertainty = static_cast<float>(leader.velocitySigma);
    leaderMessage.lost_duration = static_cast<float>(leader.lostDuration);
    leaderMessage.last_known_point = MakePointMessage(leader.lastKnown);
    return leaderMessage;
  }

  // lidarOnly: only the lidar-only clusters (the static output); otherwise every fused obstacle.
  // Fused obstacles -> PerceptionObstacleArray message, one PerceptionObstacle per obstacle.
  PerceptionObstacleArray BuildObstaclesMessage(const FusionOutput & fused,
    const builtin_interfaces::msg::Time & stamp, bool lidarOnly)
  {
    PerceptionObstacleArray obstaclesMessage;
    obstaclesMessage.header.stamp = stamp;
    obstaclesMessage.header.frame_id = baseFrame;
    obstaclesMessage.obstacles.reserve(fused.obstacles.size());
    for (const auto & fusedObstacle : fused.obstacles) {
      if (lidarOnly && !fusedObstacle.isLidarOnly) {
        continue;
      }
      PerceptionObstacle obstacleMessage;
      obstacleMessage.closest_point = MakePointMessage(fusedObstacle.data.closest);
      obstacleMessage.line_of_sight_point = MakePointMessage(fusedObstacle.data.lineOfSight);
      obstacleMessage.velocity = MakeVectorMessage(fusedObstacle.data.velocity);
      obstacleMessage.position_uncertainty = static_cast<float>(fusedObstacle.data.positionSigma);
      obstacleMessage.velocity_uncertainty = static_cast<float>(fusedObstacle.data.velocitySigma);
      obstaclesMessage.obstacles.push_back(obstacleMessage);
    }
    return obstaclesMessage;
  }

  // The last published messages again, with a new stamp (a skipped cloud).
  // Navigation checks how old messages are; repeating keeps the obstacles from disappearing for one cloud.
  void RepeatLastOutput(const builtin_interfaces::msg::Time & stamp)
  {
    if (lastLeader) {
      lastLeader->header.stamp = stamp;
      leaderPublisher->publish(*lastLeader);
    }
    if (lastStaticObstacles) {
      lastStaticObstacles->header.stamp = stamp;
      staticObstaclesPublisher->publish(*lastStaticObstacles);
    }
    if (lastCombinedObstacles) {
      lastCombinedObstacles->header.stamp = stamp;
      combinedObstaclesPublisher->publish(*lastCombinedObstacles);
    }
    if (lastScan) {
      lastScan->header.stamp = stamp;
      scanPublisher->publish(*lastScan);
    }
  }

  // Scan ranges -> LaserScan message: scanRays rays from -pi (behind) going counter-clockwise, 2 pi / scanRays apart.
  LaserScan BuildScanMessage(std::vector<float> scanRanges, const builtin_interfaces::msg::Time & stamp)
  {
    LaserScan scanMessage;
    scanMessage.header.stamp = stamp;
    scanMessage.header.frame_id = baseFrame;
    const double rayStep = 2.0 * M_PI / scanRays;
    scanMessage.angle_min = static_cast<float>(-M_PI);
    scanMessage.angle_increment = static_cast<float>(rayStep);
    scanMessage.angle_max = static_cast<float>(-M_PI + (scanRays - 1) * rayStep);
    scanMessage.range_min = static_cast<float>(filterParams.rangeMin);
    scanMessage.range_max = static_cast<float>(filterParams.rangeMax);
    scanMessage.ranges = std::move(scanRanges);
    return scanMessage;
  }

  // RViz markers for /perception/lidar_markers: a text with the floor plane, and per cluster a cylinder
  // (colour = who owns it, opacity = existence), a red sphere on its nearest point and a text label
  // "#id model p=existence speed".
  MarkerArray BuildMarkers(const std::vector<Cluster> & clusters, const FusionOutput & fused,
    const GroundPlane & floorPlane, const builtin_interfaces::msg::Time & stamp)
  {
    // First marker deletes everything from the previous cloud, so old clusters do not stay on screen.
    MarkerArray markerArray;
    Marker clearMarker;
    clearMarker.action = Marker::DELETEALL;
    markerArray.markers.push_back(clearMarker);
    // Floor text above the robot: the fitted floor height and tilt.
    Marker floorText;
    floorText.header.stamp = stamp;
    floorText.header.frame_id = baseFrame;
    floorText.ns = "floor";
    floorText.type = Marker::TEXT_VIEW_FACING;
    floorText.pose.position.z = 1.2;
    floorText.pose.orientation.w = 1.0;
    floorText.scale.z = 0.15;
    floorText.color.r = floorText.color.g = floorText.color.b = floorText.color.a = 1.0f;
    char textBuffer[96];
    std::snprintf(textBuffer, sizeof textBuffer, "floor z0 %.2f tilt %.1f deg%s", floorPlane.heightAtOrigin,
      floorPlane.CalculateTiltDegrees(), floorPlane.fitted ? "" : " (flat)");
    floorText.text = textBuffer;
    markerArray.markers.push_back(floorText);
    for (std::size_t clusterIndex = 0; clusterIndex < clusters.size(); ++clusterIndex) {
      const Cluster & cluster = clusters[clusterIndex];
      const int owner = fused.clusterOwner[clusterIndex];
      // Cylinder around the cluster, 1 m tall, as wide as the cluster.
      Marker clusterMarker;
      clusterMarker.header.stamp = stamp;
      clusterMarker.header.frame_id = baseFrame;
      clusterMarker.ns = "clusters";
      clusterMarker.id = cluster.trackId >= 0 ? cluster.trackId : static_cast<int>(clusterIndex);   // stable per track in RViz
      clusterMarker.type = Marker::CYLINDER;
      clusterMarker.pose.position.x = cluster.centroid.x;
      clusterMarker.pose.position.y = cluster.centroid.y;
      clusterMarker.pose.position.z = 0.5;
      clusterMarker.pose.orientation.w = 1.0;
      const double diameter = std::max(0.1, 2.0 * cluster.radius);
      clusterMarker.scale.x = diameter;
      clusterMarker.scale.y = diameter;
      clusterMarker.scale.z = 1.0;
      // Opacity: how sure the tracker is the object is real.
      clusterMarker.color.a =
        std::isfinite(cluster.existence) ? static_cast<float>(0.2 + 0.6 * cluster.existence) : 0.6f;
      if (owner == OWNER_LEADER) {
        clusterMarker.color.b = 1.0f;                       // leader
      } else if (owner >= 0) {
        clusterMarker.color.g = 1.0f;                       // camera person
      } else if (cluster.isCoasting) {
        clusterMarker.color.r = 1.0f;                       // lidar only, missed this cloud
        clusterMarker.color.g = 0.5f;
      } else {
        clusterMarker.color.r = clusterMarker.color.g = clusterMarker.color.b = 0.6f;   // lidar only
      }
      markerArray.markers.push_back(clusterMarker);

      // Small red sphere on the cluster's point nearest the robot (what navigation keeps distance from).
      Marker closestMarker = clusterMarker;
      closestMarker.ns = "closest";
      closestMarker.type = Marker::SPHERE;
      closestMarker.pose.position.x = cluster.closest.x;
      closestMarker.pose.position.y = cluster.closest.y;
      closestMarker.pose.position.z = 0.0;
      closestMarker.scale.x = closestMarker.scale.y = closestMarker.scale.z = 0.08;
      closestMarker.color.r = 1.0f;
      closestMarker.color.g = closestMarker.color.b = 0.0f;
      closestMarker.color.a = 1.0f;
      markerArray.markers.push_back(closestMarker);

      // Text label above the cluster: track id, motion model ("wall" for wall pieces), existence and speed.
      Marker trackLabel = floorText;
      trackLabel.ns = "tracks";
      trackLabel.id = clusterMarker.id;
      trackLabel.pose.position.x = cluster.centroid.x;
      trackLabel.pose.position.y = cluster.centroid.y;
      trackLabel.pose.position.z = 1.15;
      trackLabel.scale.z = 0.12;
      const char * motionModeName = cluster.isChunk ? "wall" :
        (cluster.motionMode >= 0 && cluster.motionMode < 3 ? MOTION_MODE_NAMES[cluster.motionMode] : "?");
      std::snprintf(textBuffer, sizeof textBuffer, "#%d %s p=%.2f %.1f m/s", cluster.trackId, motionModeName,
        std::isfinite(cluster.existence) ? cluster.existence : 0.0, CalculateLength(cluster.velocity));
      trackLabel.text = textBuffer;
      markerArray.markers.push_back(trackLabel);
    }
    return markerArray;
  }

  // Reports a problem: "LidarObstacles: <whatHappened><details>" as a warning in the log (/rosout) and on
  // /perception/debug. The same whatHappened is sent at most once per DEBUG_REPEAT_SECONDS (a steady clock is
  // used so this also works with simulation time); details (changing numbers) do not count for that.
  void PublishDebugMessage(const std::string & whatHappened, const std::string & details = "")
  {
    const double nowSeconds = steadyClock.now().seconds();
    auto lastPublished = lastDebugSeconds.find(whatHappened);
    if (lastPublished != lastDebugSeconds.end() && nowSeconds - lastPublished->second < DEBUG_REPEAT_SECONDS) {
      return;
    }
    lastDebugSeconds[whatHappened] = nowSeconds;
    std_msgs::msg::String debugMessage;
    debugMessage.data = "LidarObstacles: " + whatHappened + details;
    RCLCPP_WARN(get_logger(), "%s", debugMessage.data.c_str());
    debugPublisher->publish(debugMessage);
  }

  // Declares one parameter with the current value as its default, then writes the value from the yaml back
  // into it. Used by the constructor for every setting. Works for text, true/false, decimals and whole numbers;
  // ROS stores whole numbers as int64, so int and size_t settings are converted to int64 and back.
  template<typename Value>
  void DeclareParameter(const std::string & name, Value & value)
  {
    if constexpr (std::is_integral_v<Value> && !std::is_same_v<Value, bool>) {
      value = static_cast<Value>(declare_parameter(name, static_cast<int64_t>(value)));
    } else {
      value = declare_parameter(name, value);
    }
  }

  // Declares a parameter that the yaml gives as a [first, second] list (a box corner, a position or the two
  // lidar noise numbers) and writes the two values back. A list that is too short keeps the defaults.
  void DeclareParameter(const std::string & name, double & first, double & second)
  {
    const auto values = declare_parameter(name, std::vector<double>{first, second});
    if (values.size() >= 2) {
      first = values[0];
      second = values[1];
    }
  }

  // Parameters
  std::string baseFrame = "base_link";
  FilterParams filterParams;
  ClusterParams clusterParams;
  double cameraTimeout = 0.5;
  double lidarTimeout = 0.5;
  double odometryTimeout = 0.5;
  int scanRays = 720;
  bool publishMarkers = true;

  // Processing state
  // One object per pipeline step, each keeping its own memory between clouds.
  ClockOffset clockOffset;
  PoseHistory poseHistory;
  GroundEstimator groundEstimator;
  HitGrid hitGrid;
  CloudGate cloudGate;
  ClusterTracker tracker;
  Fusion fusion;
  bool warnedNoOdometry = false;   // the "no odometry" message is sent only once

  // Last outputs
  // Kept so RepeatLastOutput can publish them again for a skipped cloud (empty until the first publish).
  std::optional<PerceptionLeader> lastLeader;
  std::optional<PerceptionObstacleArray> lastStaticObstacles;
  std::optional<PerceptionObstacleArray> lastCombinedObstacles;
  std::optional<LaserScan> lastScan;

  // Latest inputs
  // Arrival times are on this computer's clock; -1 = nothing received yet.
  double lastOdometryArrival = -1.0;
  double lastCloudArrival = -1.0;
  PerceptionLeader::ConstSharedPtr cameraLeader;
  PerceptionObstacleArray::ConstSharedPtr cameraObstacles;
  double cameraLeaderArrival = 0.0;
  double cameraObstaclesArrival = 0.0;

  // Debug
  // When each debug message was last sent, for the rate limit in PublishDebugMessage.
  rclcpp::Clock steadyClock{RCL_STEADY_TIME};
  std::unordered_map<std::string, double> lastDebugSeconds;

  // Topics
  rclcpp::Subscription<Odometry>::SharedPtr odometrySubscription;
  rclcpp::Subscription<PointCloud2>::SharedPtr cloudSubscription;
  rclcpp::Subscription<PerceptionLeader>::SharedPtr cameraLeaderSubscription;
  rclcpp::Subscription<PerceptionObstacleArray>::SharedPtr cameraObstaclesSubscription;
  rclcpp::Publisher<PerceptionLeader>::SharedPtr leaderPublisher;
  rclcpp::Publisher<PerceptionObstacleArray>::SharedPtr staticObstaclesPublisher;
  rclcpp::Publisher<PerceptionObstacleArray>::SharedPtr combinedObstaclesPublisher;
  rclcpp::Publisher<LaserScan>::SharedPtr scanPublisher;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr debugPublisher;
  rclcpp::Publisher<MarkerArray>::SharedPtr markersPublisher;
};

}  // namespace lidar_obstacles

// Starts ROS, runs the node until Ctrl+C (spin calls the callbacks whenever a message arrives), then shuts down.
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<lidar_obstacles::LidarObstaclesNode>());
  rclcpp::shutdown();
  return 0;
}
