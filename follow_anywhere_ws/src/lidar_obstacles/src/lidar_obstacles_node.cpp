#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <follow_anywhere_interfaces/msg/perception_leader.hpp>
#include <follow_anywhere_interfaces/msg/perception_obstacle.hpp>
#include <follow_anywhere_interfaces/msg/perception_obstacle_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/string.hpp>

using follow_anywhere_interfaces::msg::PerceptionLeader;
using follow_anywhere_interfaces::msg::PerceptionObstacle;
using follow_anywhere_interfaces::msg::PerceptionObstacleArray;
using geometry_msgs::msg::Point;
using geometry_msgs::msg::Vector3;
using std::placeholders::_1;

// Frames
const std::string BASE_FRAME = "base_link";

// Obstacle values
constexpr double UNKNOWN_VELOCITY_UNCERTAINTY = 100.0;
constexpr double NOT_A_NUMBER = std::numeric_limits<double>::quiet_NaN();

// Debug
constexpr double DEBUG_REPEAT_SECONDS = 2.0;

// Parameter defaults
struct LidarObstaclesSettings
{
  double minRange = 0.15;
  double maxRange = 6.0;
  double minObstacleHeight = -0.20;
  double maxObstacleHeight = 1.0;
  double cellSize = 0.10;
  int minPointsPerCell = 2;
  int minClusterPoints = 5;
  double lidarPositionUncertainty = 0.10;
  double matchDistance = 0.5;
  double maxCameraAge = 0.5;
  double odometryTimeout = 0.5;
  double clockOffsetWindow = 5.0;
  double poseHistorySeconds = 1.0;
};

// Data structures
struct LidarCluster
{
  Point closestPoint;
  int pointCount = 0;
};

struct GridCell
{
  int cellX = 0;
  int cellY = 0;
  int pointCount = 0;
  Point closestPoint;
  double closestDistance = std::numeric_limits<double>::infinity();
  bool visited = false;
};

struct RobotPose
{
  double seconds = 0.0;
  double positionX = 0.0;
  double positionY = 0.0;
  double heading = 0.0;
};

struct ClockSample
{
  double arrivalSeconds = 0.0;
  double offsetSeconds = 0.0;
};

struct PredictedPerson
{
  PerceptionObstacle obstacle;
  bool isLeader = false;
  int matchedClusterIndex = -1;
};

struct ClusterMatch
{
  size_t personIndex = 0;
  size_t clusterIndex = 0;
  double distance = 0.0;
};

double ConvertStampToSeconds(const builtin_interfaces::msg::Time &stamp)
{
  return stamp.sec + stamp.nanosec * 1e-9;
}

double WrapAngle(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

double CalculateHorizontalDistance(const Point &firstPoint, const Point &secondPoint)
{
  return std::hypot(firstPoint.x - secondPoint.x, firstPoint.y - secondPoint.y);
}

int64_t MakeCellKey(int cellX, int cellY)
{
  return (static_cast<int64_t>(cellX) << 32) | static_cast<uint32_t>(cellY);
}

Point MakeNotANumberPoint()
{
  Point notANumberPoint;
  notANumberPoint.x = NOT_A_NUMBER;
  notANumberPoint.y = NOT_A_NUMBER;
  notANumberPoint.z = NOT_A_NUMBER;
  return notANumberPoint;
}

Point ShiftPoint(const Point &point, double shiftX, double shiftY)
{
  Point shiftedPoint = point;
  shiftedPoint.x += shiftX;
  shiftedPoint.y += shiftY;
  return shiftedPoint;
}

Point PredictPosition(const Point &point, const Vector3 &velocity, double secondsAhead)
{
  return ShiftPoint(point, velocity.x * secondsAhead, velocity.y * secondsAhead);
}

RobotPose InterpolateRobotPose(const RobotPose &earlierPose, const RobotPose &laterPose, double seconds)
{
  double poseSpan = laterPose.seconds - earlierPose.seconds;
  double fraction = poseSpan > 0.0 ? (seconds - earlierPose.seconds) / poseSpan : 0.0;
  RobotPose robotPose;
  robotPose.seconds = seconds;
  robotPose.positionX = earlierPose.positionX + fraction * (laterPose.positionX - earlierPose.positionX);
  robotPose.positionY = earlierPose.positionY + fraction * (laterPose.positionY - earlierPose.positionY);
  robotPose.heading = WrapAngle(earlierPose.heading + fraction * WrapAngle(laterPose.heading - earlierPose.heading));
  return robotPose;
}

Point MovePointBetweenPoses(const Point &point, const RobotPose &poseThen, const RobotPose &poseNow)
{
  double odometryX = poseThen.positionX + std::cos(poseThen.heading) * point.x - std::sin(poseThen.heading) * point.y;
  double odometryY = poseThen.positionY + std::sin(poseThen.heading) * point.x + std::cos(poseThen.heading) * point.y;
  double offsetX = odometryX - poseNow.positionX;
  double offsetY = odometryY - poseNow.positionY;
  Point movedPoint = point;
  movedPoint.x = std::cos(poseNow.heading) * offsetX + std::sin(poseNow.heading) * offsetY;
  movedPoint.y = -std::sin(poseNow.heading) * offsetX + std::cos(poseNow.heading) * offsetY;
  return movedPoint;
}

Vector3 RotateVectorBetweenPoses(const Vector3 &vector, const RobotPose &poseThen, const RobotPose &poseNow)
{
  double headingChange = poseThen.heading - poseNow.heading;
  Vector3 rotatedVector = vector;
  rotatedVector.x = std::cos(headingChange) * vector.x - std::sin(headingChange) * vector.y;
  rotatedVector.y = std::sin(headingChange) * vector.x + std::cos(headingChange) * vector.y;
  return rotatedVector;
}

std::optional<PerceptionObstacle> MakeLeaderObstacle(const PerceptionLeader &leader, double matchDistance)
{
  PerceptionObstacle leaderObstacle;
  leaderObstacle.velocity = leader.velocity;
  leaderObstacle.line_of_sight_point = MakeNotANumberPoint();
  leaderObstacle.position_uncertainty = leader.position_uncertainty;
  leaderObstacle.velocity_uncertainty = leader.velocity_uncertainty;
  if (leader.status == PerceptionLeader::VISIBLE && std::isfinite(leader.closest_point.x))
  {
    leaderObstacle.closest_point = leader.closest_point;
    return leaderObstacle;
  }
  if (leader.status == PerceptionLeader::NO_LEADER || !std::isfinite(leader.last_known_point.x) ||
      leader.position_uncertainty > matchDistance)
  {
    return std::nullopt;
  }
  double lostSeconds = std::isfinite(leader.lost_duration) ? leader.lost_duration : 0.0;
  leaderObstacle.closest_point = PredictPosition(leader.last_known_point, leader.velocity, lostSeconds);
  return leaderObstacle;
}

class LidarObstaclesNode : public rclcpp::Node
{
public:
  LidarObstaclesNode() : Node("lidar_obstacles_node")
  {
    LidarObstaclesSettings defaults;
    settings.minRange = declare_parameter("minRange", defaults.minRange);
    settings.maxRange = declare_parameter("maxRange", defaults.maxRange);
    settings.minObstacleHeight = declare_parameter("minObstacleHeight", defaults.minObstacleHeight);
    settings.maxObstacleHeight = declare_parameter("maxObstacleHeight", defaults.maxObstacleHeight);
    settings.cellSize = declare_parameter("cellSize", defaults.cellSize);
    settings.minPointsPerCell = declare_parameter("minPointsPerCell", defaults.minPointsPerCell);
    settings.minClusterPoints = declare_parameter("minClusterPoints", defaults.minClusterPoints);
    settings.lidarPositionUncertainty = declare_parameter("lidarPositionUncertainty", defaults.lidarPositionUncertainty);
    settings.matchDistance = declare_parameter("matchDistance", defaults.matchDistance);
    settings.maxCameraAge = declare_parameter("maxCameraAge", defaults.maxCameraAge);
    settings.odometryTimeout = declare_parameter("odometryTimeout", defaults.odometryTimeout);
    settings.clockOffsetWindow = declare_parameter("clockOffsetWindow", defaults.clockOffsetWindow);
    settings.poseHistorySeconds = declare_parameter("poseHistorySeconds", defaults.poseHistorySeconds);

    // Output topics
    staticObstaclesPublisher = create_publisher<PerceptionObstacleArray>("/perception/obstacles_static", 10);
    combinedObstaclesPublisher = create_publisher<PerceptionObstacleArray>("/perception/obstacles_combined", 10);
    debugPublisher = create_publisher<std_msgs::msg::String>("/perception/debug", 10);

    // Input topics
    rclcpp::QoS newestMessageQos = rclcpp::QoS(1).best_effort();
    robotOdometrySubscription = create_subscription<nav_msgs::msg::Odometry>(
      "/utlidar/robot_odom", newestMessageQos, std::bind(&LidarObstaclesNode::HandleRobotOdometry, this, _1));
    cameraObstaclesSubscription = create_subscription<PerceptionObstacleArray>(
      "/perception/obstacles", 10, std::bind(&LidarObstaclesNode::HandleCameraObstacles, this, _1));
    leaderSubscription = create_subscription<PerceptionLeader>(
      "/perception/leader", 10, std::bind(&LidarObstaclesNode::HandleLeader, this, _1));
    pointCloudSubscription = create_subscription<sensor_msgs::msg::PointCloud2>(
      "/utlidar/cloud_base", rclcpp::SensorDataQoS(), std::bind(&LidarObstaclesNode::HandlePointCloud, this, _1));
  }

private:
  void HandleCameraObstacles(const PerceptionObstacleArray::ConstSharedPtr cameraObstaclesMessage)
  {
    latestCameraObstacles = cameraObstaclesMessage;
  }

  void HandleLeader(const PerceptionLeader::ConstSharedPtr leaderMessage)
  {
    latestLeader = leaderMessage;
  }

  void HandleRobotOdometry(const nav_msgs::msg::Odometry::ConstSharedPtr odometryMessage)
  {
    double arrivalSeconds = now().seconds();
    if (std::isnan(lastOdometryArrivalSeconds))
    {
      RCLCPP_INFO(get_logger(), "LidarObstacles: odometry started, people are now moved with the robot's motion");
    }
    else if (arrivalSeconds - lastOdometryArrivalSeconds > settings.odometryTimeout)
    {
      PublishDebugMessage("odometry back, pose history restarted");
      clockSamples.clear();
      poseHistory.clear();
    }
    lastOdometryArrivalSeconds = arrivalSeconds;
    double correctedSeconds = CorrectStamp(ConvertStampToSeconds(odometryMessage->header.stamp), arrivalSeconds);
    if (!poseHistory.empty() && correctedSeconds <= poseHistory.back().seconds)
    {
      if (poseHistory.back().seconds - correctedSeconds < settings.poseHistorySeconds)
      {
        return;
      }
      poseHistory.clear();
    }
    const auto &orientation = odometryMessage->pose.pose.orientation;
    RobotPose robotPose;
    robotPose.seconds = correctedSeconds;
    robotPose.positionX = odometryMessage->pose.pose.position.x;
    robotPose.positionY = odometryMessage->pose.pose.position.y;
    robotPose.heading = std::atan2(2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
                                   1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z));
    poseHistory.push_back(robotPose);
    while (poseHistory.front().seconds < correctedSeconds - settings.poseHistorySeconds)
    {
      poseHistory.pop_front();
    }
  }

  void HandlePointCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloudMessage)
  {
    if (cloudMessage->width * cloudMessage->height == 0)
    {
      PublishDebugMessage("point cloud empty, skipping frame");
      return;
    }
    if (cloudMessage->header.frame_id != BASE_FRAME)
    {
      PublishDebugMessage("point cloud in frame '" + cloudMessage->header.frame_id + "', expected " + BASE_FRAME +
                          ", skipping frame");
      return;
    }
    rclcpp::Time lidarTime = std::isnan(clockOffsetSeconds)
      ? now()
      : rclcpp::Time(cloudMessage->header.stamp, get_clock()->get_clock_type()) +
          rclcpp::Duration::from_seconds(clockOffsetSeconds);
    std::vector<LidarCluster> lidarClusters;
    try
    {
      lidarClusters = FindLidarClusters(*cloudMessage);
    }
    catch (const std::runtime_error &error)
    {
      PublishDebugMessage(std::string("point cloud unreadable (") + error.what() + "), skipping frame");
      return;
    }
    std::vector<PredictedPerson> predictedPeople = CollectPredictedPeople(lidarTime.seconds());
    MatchClustersToPeople(predictedPeople, lidarClusters);

    std_msgs::msg::Header obstaclesHeader;
    obstaclesHeader.stamp = lidarTime;
    obstaclesHeader.frame_id = BASE_FRAME;
    PerceptionObstacleArray staticObstacles = BuildStaticObstacles(obstaclesHeader, lidarClusters, predictedPeople);
    PerceptionObstacleArray combinedObstacles = BuildCombinedObstacles(staticObstacles, lidarClusters, predictedPeople);
    staticObstaclesPublisher->publish(staticObstacles);
    combinedObstaclesPublisher->publish(combinedObstacles);
  }

  double CorrectStamp(double stampSeconds, double arrivalSeconds)
  {
    double offsetSeconds = arrivalSeconds - stampSeconds;
    while (!clockSamples.empty() && clockSamples.back().offsetSeconds >= offsetSeconds)
    {
      clockSamples.pop_back();
    }
    clockSamples.push_back({arrivalSeconds, offsetSeconds});
    while (clockSamples.front().arrivalSeconds < arrivalSeconds - settings.clockOffsetWindow)
    {
      clockSamples.pop_front();
    }
    clockOffsetSeconds = clockSamples.front().offsetSeconds;
    return stampSeconds + clockOffsetSeconds;
  }

  std::optional<RobotPose> FindRobotPose(double seconds)
  {
    if (poseHistory.empty() || seconds - poseHistory.back().seconds > settings.odometryTimeout)
    {
      return std::nullopt;
    }
    if (seconds <= poseHistory.front().seconds)
    {
      return poseHistory.front();
    }
    if (seconds >= poseHistory.back().seconds)
    {
      return poseHistory.back();
    }
    auto laterPose = std::lower_bound(poseHistory.begin(), poseHistory.end(), seconds,
      [](const RobotPose &robotPose, double searchedSeconds) { return robotPose.seconds < searchedSeconds; });
    return InterpolateRobotPose(*std::prev(laterPose), *laterPose, seconds);
  }

  std::vector<LidarCluster> FindLidarClusters(const sensor_msgs::msg::PointCloud2 &cloud)
  {
    std::unordered_map<int64_t, GridCell> gridCells;
    sensor_msgs::PointCloud2ConstIterator<float> iteratorX(cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iteratorY(cloud, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iteratorZ(cloud, "z");
    for (; iteratorX != iteratorX.end(); ++iteratorX, ++iteratorY, ++iteratorZ)
    {
      double pointX = *iteratorX;
      double pointY = *iteratorY;
      double pointZ = *iteratorZ;
      if (!std::isfinite(pointX) || !std::isfinite(pointY) || !std::isfinite(pointZ))
      {
        continue;
      }
      double horizontalDistance = std::hypot(pointX, pointY);
      if (horizontalDistance < settings.minRange || horizontalDistance > settings.maxRange ||
          pointZ < settings.minObstacleHeight || pointZ > settings.maxObstacleHeight)
      {
        continue;
      }
      int cellX = static_cast<int>(std::floor(pointX / settings.cellSize));
      int cellY = static_cast<int>(std::floor(pointY / settings.cellSize));
      GridCell &gridCell = gridCells[MakeCellKey(cellX, cellY)];
      gridCell.cellX = cellX;
      gridCell.cellY = cellY;
      gridCell.pointCount++;
      if (horizontalDistance < gridCell.closestDistance)
      {
        gridCell.closestDistance = horizontalDistance;
        gridCell.closestPoint.x = pointX;
        gridCell.closestPoint.y = pointY;
        gridCell.closestPoint.z = 0.0;
      }
    }

    std::vector<LidarCluster> lidarClusters;
    for (auto &cellEntry : gridCells)
    {
      GridCell &gridCell = cellEntry.second;
      if (gridCell.visited || gridCell.pointCount < settings.minPointsPerCell)
      {
        continue;
      }
      LidarCluster lidarCluster = GrowClusterFromCell(gridCells, gridCell);
      if (lidarCluster.pointCount >= settings.minClusterPoints)
      {
        lidarClusters.push_back(lidarCluster);
      }
    }
    return lidarClusters;
  }

  LidarCluster GrowClusterFromCell(std::unordered_map<int64_t, GridCell> &gridCells, GridCell &startCell)
  {
    LidarCluster lidarCluster;
    double closestDistance = std::numeric_limits<double>::infinity();
    std::vector<GridCell *> cellsToVisit{&startCell};
    startCell.visited = true;
    while (!cellsToVisit.empty())
    {
      GridCell *gridCell = cellsToVisit.back();
      cellsToVisit.pop_back();
      lidarCluster.pointCount += gridCell->pointCount;
      if (gridCell->closestDistance < closestDistance)
      {
        closestDistance = gridCell->closestDistance;
        lidarCluster.closestPoint = gridCell->closestPoint;
      }
      for (int stepX = -1; stepX <= 1; ++stepX)
      {
        for (int stepY = -1; stepY <= 1; ++stepY)
        {
          auto neighbourCell = gridCells.find(MakeCellKey(gridCell->cellX + stepX, gridCell->cellY + stepY));
          if (neighbourCell == gridCells.end() || neighbourCell->second.visited ||
              neighbourCell->second.pointCount < settings.minPointsPerCell)
          {
            continue;
          }
          neighbourCell->second.visited = true;
          cellsToVisit.push_back(&neighbourCell->second);
        }
      }
    }
    return lidarCluster;
  }

  std::vector<PredictedPerson> CollectPredictedPeople(double lidarSeconds)
  {
    std::vector<PredictedPerson> predictedPeople;
    if (!FindRobotPose(lidarSeconds))
    {
      PublishDebugMessage("no recent odometry, robot counted as standing still");
    }
    double cameraSeconds = latestCameraObstacles ? ConvertStampToSeconds(latestCameraObstacles->header.stamp) : NOT_A_NUMBER;
    if (!latestCameraObstacles || lidarSeconds - cameraSeconds > settings.maxCameraAge)
    {
      PublishDebugMessage("no recent camera obstacles, people stay in the static obstacles");
    }
    else
    {
      for (const PerceptionObstacle &cameraObstacle : latestCameraObstacles->obstacles)
      {
        PredictedPerson predictedPerson;
        predictedPerson.obstacle = MoveObstacleToLidarTime(cameraObstacle, cameraSeconds, lidarSeconds);
        predictedPeople.push_back(predictedPerson);
      }
    }
    double leaderSeconds = latestLeader ? ConvertStampToSeconds(latestLeader->header.stamp) : NOT_A_NUMBER;
    if (latestLeader && lidarSeconds - leaderSeconds <= settings.maxCameraAge)
    {
      std::optional<PerceptionObstacle> leaderObstacle = MakeLeaderObstacle(*latestLeader, settings.matchDistance);
      if (leaderObstacle)
      {
        PredictedPerson predictedLeader;
        predictedLeader.obstacle = MoveObstacleToLidarTime(*leaderObstacle, leaderSeconds, lidarSeconds);
        predictedLeader.isLeader = true;
        predictedPeople.push_back(predictedLeader);
      }
    }
    return predictedPeople;
  }

  PerceptionObstacle MoveObstacleToLidarTime(const PerceptionObstacle &obstacle, double cameraSeconds, double lidarSeconds)
  {
    std::optional<RobotPose> poseThen = FindRobotPose(cameraSeconds);
    std::optional<RobotPose> poseNow = FindRobotPose(lidarSeconds);
    RobotPose stillPose;
    bool odometryAvailable = poseThen && poseNow;
    const RobotPose &robotPoseThen = odometryAvailable ? *poseThen : stillPose;
    const RobotPose &robotPoseNow = odometryAvailable ? *poseNow : stillPose;
    double secondsAhead = lidarSeconds - cameraSeconds;
    PerceptionObstacle movedObstacle = obstacle;
    movedObstacle.velocity = RotateVectorBetweenPoses(obstacle.velocity, robotPoseThen, robotPoseNow);
    movedObstacle.closest_point = PredictPosition(
      MovePointBetweenPoses(obstacle.closest_point, robotPoseThen, robotPoseNow), movedObstacle.velocity, secondsAhead);
    movedObstacle.line_of_sight_point = PredictPosition(
      MovePointBetweenPoses(obstacle.line_of_sight_point, robotPoseThen, robotPoseNow), movedObstacle.velocity, secondsAhead);
    return movedObstacle;
  }

  void MatchClustersToPeople(std::vector<PredictedPerson> &predictedPeople, const std::vector<LidarCluster> &lidarClusters)
  {
    std::vector<ClusterMatch> candidateMatches;
    for (size_t personIndex = 0; personIndex < predictedPeople.size(); ++personIndex)
    {
      for (size_t clusterIndex = 0; clusterIndex < lidarClusters.size(); ++clusterIndex)
      {
        double distance = CalculateHorizontalDistance(predictedPeople[personIndex].obstacle.closest_point,
                                                      lidarClusters[clusterIndex].closestPoint);
        if (distance <= settings.matchDistance)
        {
          candidateMatches.push_back({personIndex, clusterIndex, distance});
        }
      }
    }
    std::sort(candidateMatches.begin(), candidateMatches.end(),
              [](const ClusterMatch &first, const ClusterMatch &second) { return first.distance < second.distance; });
    std::vector<bool> clusterTaken(lidarClusters.size(), false);
    for (const ClusterMatch &candidateMatch : candidateMatches)
    {
      PredictedPerson &predictedPerson = predictedPeople[candidateMatch.personIndex];
      if (predictedPerson.matchedClusterIndex >= 0 || clusterTaken[candidateMatch.clusterIndex])
      {
        continue;
      }
      predictedPerson.matchedClusterIndex = static_cast<int>(candidateMatch.clusterIndex);
      clusterTaken[candidateMatch.clusterIndex] = true;
    }
  }

  PerceptionObstacleArray BuildStaticObstacles(const std_msgs::msg::Header &obstaclesHeader,
                                               const std::vector<LidarCluster> &lidarClusters,
                                               const std::vector<PredictedPerson> &predictedPeople)
  {
    std::vector<bool> clusterIsPerson(lidarClusters.size(), false);
    for (const PredictedPerson &predictedPerson : predictedPeople)
    {
      if (predictedPerson.matchedClusterIndex >= 0)
      {
        clusterIsPerson[predictedPerson.matchedClusterIndex] = true;
      }
    }
    PerceptionObstacleArray staticObstacles;
    staticObstacles.header = obstaclesHeader;
    for (size_t clusterIndex = 0; clusterIndex < lidarClusters.size(); ++clusterIndex)
    {
      if (!clusterIsPerson[clusterIndex])
      {
        staticObstacles.obstacles.push_back(MakeLidarObstacle(lidarClusters[clusterIndex]));
      }
    }
    return staticObstacles;
  }

  PerceptionObstacleArray BuildCombinedObstacles(const PerceptionObstacleArray &staticObstacles,
                                                 const std::vector<LidarCluster> &lidarClusters,
                                                 const std::vector<PredictedPerson> &predictedPeople)
  {
    PerceptionObstacleArray combinedObstacles = staticObstacles;
    for (const PredictedPerson &predictedPerson : predictedPeople)
    {
      if (predictedPerson.isLeader)
      {
        continue;
      }
      PerceptionObstacle personObstacle = predictedPerson.obstacle;
      if (predictedPerson.matchedClusterIndex >= 0)
      {
        const Point &lidarPoint = lidarClusters[predictedPerson.matchedClusterIndex].closestPoint;
        double shiftX = lidarPoint.x - personObstacle.closest_point.x;
        double shiftY = lidarPoint.y - personObstacle.closest_point.y;
        personObstacle.closest_point = lidarPoint;
        personObstacle.line_of_sight_point = ShiftPoint(personObstacle.line_of_sight_point, shiftX, shiftY);
        personObstacle.position_uncertainty = std::min(personObstacle.position_uncertainty,
                                                       static_cast<float>(settings.lidarPositionUncertainty));
      }
      combinedObstacles.obstacles.push_back(personObstacle);
    }
    return combinedObstacles;
  }

  PerceptionObstacle MakeLidarObstacle(const LidarCluster &lidarCluster)
  {
    PerceptionObstacle lidarObstacle;
    lidarObstacle.closest_point = lidarCluster.closestPoint;
    lidarObstacle.line_of_sight_point = MakeNotANumberPoint();
    lidarObstacle.position_uncertainty = settings.lidarPositionUncertainty;
    lidarObstacle.velocity_uncertainty = UNKNOWN_VELOCITY_UNCERTAINTY;
    return lidarObstacle;
  }

  void PublishDebugMessage(const std::string &whatHappened)
  {
    std::string debugMessage = "LidarObstacles: " + whatHappened;
    double nowSeconds = steadyClock.now().seconds();
    auto lastPublished = lastDebugSeconds.find(debugMessage);
    if (lastPublished != lastDebugSeconds.end() && nowSeconds - lastPublished->second < DEBUG_REPEAT_SECONDS)
    {
      return;
    }
    lastDebugSeconds[debugMessage] = nowSeconds;
    RCLCPP_WARN(get_logger(), "%s", debugMessage.c_str());
    std_msgs::msg::String debugString;
    debugString.data = debugMessage;
    debugPublisher->publish(debugString);
  }

  // Settings
  LidarObstaclesSettings settings;

  // Latest camera data
  PerceptionObstacleArray::ConstSharedPtr latestCameraObstacles;
  PerceptionLeader::ConstSharedPtr latestLeader;

  // Robot odometry
  std::deque<RobotPose> poseHistory;
  std::deque<ClockSample> clockSamples;
  double clockOffsetSeconds = NOT_A_NUMBER;
  double lastOdometryArrivalSeconds = NOT_A_NUMBER;

  // Debug
  rclcpp::Clock steadyClock{RCL_STEADY_TIME};
  std::unordered_map<std::string, double> lastDebugSeconds;

  // Topics
  rclcpp::Publisher<PerceptionObstacleArray>::SharedPtr staticObstaclesPublisher;
  rclcpp::Publisher<PerceptionObstacleArray>::SharedPtr combinedObstaclesPublisher;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr debugPublisher;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr robotOdometrySubscription;
  rclcpp::Subscription<PerceptionObstacleArray>::SharedPtr cameraObstaclesSubscription;
  rclcpp::Subscription<PerceptionLeader>::SharedPtr leaderSubscription;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointCloudSubscription;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LidarObstaclesNode>());
  rclcpp::shutdown();
  return 0;
}
