// Gives clusters a stable id across clouds and a velocity over the ground. No ROS.
//
// Tracks live in odom, so the robot's own motion is removed. Each cloud, clusters are matched greedily to
// the nearest track centroid within trackGate. Velocity: finite difference of the centroid, first-order
// filtered (velocityAlpha, as velocity_alpha in 2dSim's fake perception); its sigma from the filtered
// spread of the raw differences. A track not seen for trackTimeout is dropped; its next cluster starts over.
// Before a track has two sightings its velocity is unknown (0, unknownVelocitySigma), as in perception's
// velocity_tracker. Published velocity is 0 below max(minimumSpeed, 2 sigma).
// Wall pieces (isChunk) are static by assumption: their centroid slides as the visible part of the
// wall changes, which is not motion.
// Without odometry the robot's motion cannot be removed: every velocity is reported unknown.
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
};

class ClusterTracker
{
public:
  explicit ClusterTracker(TrackerParams params = {})
  : params_(params) {}

  // Fills trackId, velocity and velocitySigma of clusters (base_link of robot pose `robot`, time t).
  void update(std::vector<Cluster> & clusters, const Pose2 & robot, bool haveOdometry, double t);

  void reset() {tracks_.clear();}

private:
  struct Track
  {
    int id = 0;
    Point2 position;          // odom
    Point2 velocity;          // odom
    double variance = 0.0;    // (m/s)^2 per axis
    double lastSeen = 0.0;
    int sightings = 0;
  };

  TrackerParams params_;
  std::vector<Track> tracks_;
  int nextId_ = 0;
};

}  // namespace tmp_lidar
