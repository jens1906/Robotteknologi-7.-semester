// Gives clusters a stable id across clouds, a velocity over the ground, and keeps the published set steady.
// No ROS.
//
// Tracks live in odom, so the robot's own motion is removed. Each cloud, clusters are matched greedily to
// the nearest track centroid within trackGate. Velocity: finite difference of the centroid, first-order
// filtered (velocityAlpha, as velocity_alpha in 2dSim's fake perception); its sigma from the filtered
// spread of the raw differences. Before a track has two sightings its velocity is unknown
// (0, unknownVelocitySigma), as in perception's velocity_tracker. Published velocity is 0 below
// max(minimumSpeed, 2 sigma). Wall pieces (isChunk) are static by assumption. Without odometry every velocity
// is reported unknown.
//
// Against flicker (the sparse L1 sees an object in one cloud and misses it in the next):
// - confirmation: a track is published only after confirmHits sightings;
// - coasting: a confirmed track missed in a cloud is still published with its last points (moved by its
//   velocity), its position sigma growing by coastSigmaRate per second. For coastTime near the robot, rising
//   linearly to coastTimeFar at coastFarRange: far objects get few points per cloud and are missed more often
//   (bag go2_all_topics_6_10_11_28: ~1 % churn per cloud below 1 m, 12-20 % beyond 4 m).
// A track not seen for trackTimeout is dropped.
#pragma once

#include <vector>

#include "tmp_lidar/types.hpp"

namespace tmp_lidar
{

struct TrackerParams
{
  double trackGate = 0.5;
  double velocityAlpha = 0.3;
  double minimumSpeed = 0.15;
  double trackTimeout = 1.0;
  double unknownVelocitySigma = 1.5;
  double minimumVelocitySigma = 0.05;
  double staticVelocitySigma = 0.1;   // sigma reported for wall pieces
  int confirmHits = 2;
  double coastTime = 0.3;
  double coastTimeFar = 0.8;
  double coastFarRange = 5.0;
  double coastSigmaRate = 0.5;        // m/s
};

class ClusterTracker
{
public:
  explicit ClusterTracker(TrackerParams params = {})
  : params_(params) {}

  // Matches clusters (base_link of robot pose `robot`, time t) to tracks and fills their trackId, velocity
  // and velocitySigma.
  void update(std::vector<Cluster> & clusters, const Pose2 & robot, bool haveOdometry, double t);

  // What to publish after update(): the confirmed clusters of this cloud plus the coasting ones, in base_link
  // of `robot`.
  std::vector<Cluster> publishable(const std::vector<Cluster> & clusters, const Pose2 & robot, double t) const;

  // How long a track whose nearest point is `range` from the robot may coast.
  double coastLimit(double range) const;

  void reset() {tracks_.clear();}

private:
  struct Track
  {
    int id = 0;
    Point2 position;          // odom, centroid
    Point2 velocity;          // odom
    double variance = 0.0;    // (m/s)^2 per axis
    double lastSeen = 0.0;
    int sightings = 0;
    bool isChunk = false;
    std::vector<Point2> points;   // odom, last cluster
    Point2 motion;                // odom, the published velocity (0 when standing)
    double publishedSigma = kNaN;
  };

  TrackerParams params_;
  std::vector<Track> tracks_;
  int nextId_ = 0;
  bool haveOdometry_ = false;
};

}  // namespace tmp_lidar
