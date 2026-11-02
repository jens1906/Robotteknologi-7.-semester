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
#pragma once

#include <array>
#include <vector>

#include "tmp_lidar/types.hpp"

namespace tmp_lidar
{

// As follow_anywhere_interfaces/PerceptionLeader.
constexpr int kNoLeader = 0;
constexpr int kVisible = 1;
constexpr int kLost = 2;

struct LeaderData
{
  int status = kNoLeader;
  Point2 closest = nanPoint();
  Point2 velocity;
  double positionSigma = kNaN;
  double velocitySigma = kNaN;
  double lostDuration = kNaN;
  Point2 lastKnown = nanPoint();
};

struct ObstacleData
{
  Point2 closest = nanPoint();
  Point2 lineOfSight = nanPoint();
  Point2 velocity;
  double positionSigma = kNaN;
  double velocitySigma = kNaN;
};

enum class Source { Camera, CameraLidar, Lidar };

struct FusedObstacle
{
  ObstacleData data;
  Source source = Source::Camera;
  int cluster = -1;   // index into the clusters, -1 = camera only
};

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

struct FusionInput
{
  bool haveCamera = false;   // false: lidar only (no camera message, or too old)
  LeaderData leader;
  std::vector<ObstacleData> obstacles;
};

// Owner of each cluster in FusionOutput::clusterOwner.
constexpr int kOwnerNone = -1;     // lidar-only obstacle
constexpr int kOwnerLeader = -2;   // the leader (matched or held)

struct FusionOutput
{
  LeaderData leader;
  bool leaderEnriched = false;
  int leaderCluster = -1;
  std::vector<FusedObstacle> obstacles;
  std::vector<int> clusterOwner;   // per cluster: kOwnerNone, kOwnerLeader, or the camera obstacle index
};

class Fusion
{
public:
  explicit Fusion(FusionParams params = {})
  : params_(params) {}

  FusionOutput fuse(const FusionInput & input, const std::vector<Cluster> & clusters, const Pose2 & robot,
    double t);

  void reset();

  double lidarSigma(double range) const;

private:
  FusionParams params_;
  int leaderTrack_ = -1;
  double leaderMatchTime_ = 0.0;
  bool haveLeaderPoint_ = false;
  Point2 leaderPointOdom_;
  double leaderPointSigma_ = kNaN;
};

// Inverse variance blend of two velocity estimates; b is ignored when its sigma is not finite.
void blendVelocity(const Point2 & a, double sigmaA, const Point2 & b, double sigmaB, Point2 & out,
  double & sigmaOut);

}  // namespace tmp_lidar
