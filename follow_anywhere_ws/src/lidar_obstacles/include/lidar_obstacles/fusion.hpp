// Outputs from the tracked clusters. No ROS.
// Two parts:
// - Fusion: merges camera people with lidar clusters into one leader and one obstacle list.
// - Scan builder: turns the clusters into a 360 deg laser scan.
#pragma once

#include <array>
#include <vector>

#include "lidar_obstacles/types.hpp"

namespace lidar_obstacles
{

// ===== Fusion ======================================================================================================
// Camera people + lidar clusters -> one leader and one obstacle list. No ROS.
//
// Inputs are already in the base_link of the cloud's time (the node shifts the camera message there with
// odometry). Steps:
//  1. Match: the camera's people (the leader while VISIBLE, then each camera obstacle) are matched to
//     clusters by the distance from the camera's closest point to the cluster's nearest lidar point,
//     gated by max(matchGate, matchSigmas * position_uncertainty). Greedy, lowest cost first, one cluster
//     per person. The leader's cluster of the last frame gets stickyBonus off its cost.
//  2. Enrich a matched person with the lidar: closest point = the cluster's nearest lidar point,
//     position_uncertainty = lidar range std (a + b * range, at least minimumPositionUncertainty),
//     line_of_sight_point = the cluster point nearest the line camera -> leader, velocity = inverse
//     variance blend of the camera's and the cluster track's (camera only when the track's is unknown).
//  3. Deduplicate: a matched cluster is not reported again. Every other cluster becomes a lidar-only
//     obstacle (static objects, walls, people the camera does not see).
//  4. Leader: last_known_point is kept in odom by this class (the enriched point), so it stays on the floor
//     while the robot moves. While LOST and the leader's last cluster is still tracked (same track id, at
//     most leaderHoldTime after the last camera match) and no camera person took it, that cluster is the
//     leader's: it is left out of the obstacles and moves last_known_point. The status stays LOST:
//     identity comes from the camera only.
// Camera people without a cluster (out of lidar range, hidden) pass through unchanged.

// Leader status, as follow_anywhere_interfaces/PerceptionLeader
constexpr int NO_LEADER = 0;
constexpr int LEADER_VISIBLE = 1;
constexpr int LEADER_LOST = 2;

// Owner of each cluster in FusionOutput::clusterOwner
constexpr int OWNER_NONE = -1;     // lidar-only obstacle
constexpr int OWNER_LEADER = -2;   // the leader (matched or held)

// The leader, without ROS: the same fields as a PerceptionLeader message, in base_link of the cloud's time.
// The node converts the camera's message into this, and the fused result back into a message.
struct LeaderData
{
  int status = NO_LEADER;
  Point2 closest = MakeNotANumberPoint();     // the leader's point nearest the robot
  Point2 velocity;
  double positionSigma = NOT_A_NUMBER;
  double velocitySigma = NOT_A_NUMBER;
  double lostDuration = NOT_A_NUMBER;         // s since the camera last saw the leader
  Point2 lastKnown = MakeNotANumberPoint();   // where the leader was last seen
};

// One obstacle, without ROS: the same fields as a PerceptionObstacle message, in base_link.
struct ObstacleData
{
  Point2 closest = MakeNotANumberPoint();       // point nearest the robot
  Point2 lineOfSight = MakeNotANumberPoint();   // point nearest the camera -> leader line
  Point2 velocity;
  double positionSigma = NOT_A_NUMBER;
  double velocitySigma = NOT_A_NUMBER;
};

// One obstacle of the fused output and where it came from.
struct FusedObstacle
{
  ObstacleData data;
  bool isLidarOnly = false;   // a cluster no camera person matched
};

// Settings of the fusion (same names in the yaml).
struct FusionParams
{
  double matchGate = 0.5;
  double matchSigmas = 3.0;
  double stickyBonus = 0.2;
  double leaderHoldTime = 1.0;
  std::array<double, 2> lidarRangeStd{0.02, 0.005};
  double minimumPositionUncertainty = 0.05;
  Point2 cameraPosition{0.30, 0.0};   // base_link; start of the line of sight
};

// What the camera reported (already moved to the cloud's time by the node).
struct FusionInput
{
  bool hasCamera = false;   // false: lidar only (no camera message, or too old)
  LeaderData leader;
  std::vector<ObstacleData> obstacles;
};

// The fused result: the leader and every obstacle (camera people first, then lidar-only clusters).
struct FusionOutput
{
  LeaderData leader;
  std::vector<FusedObstacle> obstacles;
  std::vector<int> clusterOwner;   // per cluster: OWNER_NONE, OWNER_LEADER, or the camera obstacle index
};

// Runs the fusion once per cloud. It is a class (not a plain function) because it remembers a few things about
// the leader between clouds: its cluster's track id and its last known point in odom.
class Fusion
{
public:
  explicit Fusion(FusionParams fusionParams = {})
  : params(fusionParams) {}

  // Steps 1-4 from the top of this file, for one cloud.
  FusionOutput Fuse(const FusionInput & input, const std::vector<Cluster> & clusters, const Pose2 & robot,
    double cloudSeconds);

  // Forget the leader memory.
  void Reset();

  // Position sigma of a cluster: lidarRangeStd at the distance of its nearest point (at least
  // minimumPositionUncertainty) + its extraSigma.
  double CalculateClusterSigma(const Cluster & cluster) const;

private:
  FusionParams params;
  int leaderTrackId = -1;               // track id of the leader's cluster (-1 = none)
  double leaderMatchSeconds = 0.0;      // time the camera leader was last matched to that cluster
  bool hasLeaderPoint = false;          // leaderPointOdom holds a value
  Point2 leaderPointOdom;               // the leader's last known point, kept in odom
  double leaderPointSigma = NOT_A_NUMBER;
};

// Inverse variance blend of two velocity estimates; secondVelocity is ignored when its sigma is not finite.
void BlendVelocity(const Point2 & firstVelocity, double firstSigma, const Point2 & secondVelocity,
  double secondSigma, Point2 & blendedVelocity, double & blendedSigma);

// ===== Scan builder ================================================================================================
// Obstacle clusters -> 360 deg laser scan ranges around base_link (as 2dSim's /perception/scan). No ROS.
// Ray i points at -pi + i * 2 pi / rayCount; each ray keeps the nearest range, +inf when empty.
//
// The node puts these ranges into a sensor_msgs/LaserScan on /perception/scan. Navigation can use the scan to
// check free space in any direction without handling the obstacle list itself.

// Each cluster covers its whole bearing span: between neighbouring points (by bearing) of a
// cluster the range is interpolated. The sparse L1 hits a different scatter of rays on an object every cloud, so a
// scan of raw points switches most rays on and off from cloud to cloud (bag: 94 % of the rays within 4 m); the
// tracked clusters are steady and every ray across an object returns.
// Ranges outside rangeMin..rangeMax are left out (the ray stays +inf).
std::vector<float> BuildScan(const std::vector<Cluster> & clusters, int rayCount, double rangeMin, double rangeMax);

}  // namespace lidar_obstacles
