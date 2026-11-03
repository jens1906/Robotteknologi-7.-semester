// Gives clusters a stable id across clouds, a velocity over the ground, and keeps the published set steady.
// No ROS.
//
// Tracks live in odom, so the robot's own motion is removed.
//
// Motion: each track runs an IMM Kalman filter (motion_model.hpp: static / constant velocity / turning) on the
// cluster's measuredCentroid, with measurement sigma = lidar range sigma + shapeSigmaScale * cluster radius
// (a partial view moves the centroid), + staleCentroidSigma when the centroid includes the memory trail. Before
// a track has two sightings its velocity is unknown (0, unknownVelocitySigma), as in perception's
// velocity_tracker. Published velocity is 0 when the static model is the likely one (staticPublishProbability)
// or the speed is below max(minimumSpeed, 2 sigma). Wall pieces (isChunk) are static by assumption and keep their
// track by their odom grid cell (chunkKey). Without odometry every velocity is reported unknown.
//
// Association: clusters and tracks are paired by minimum total cost (Hungarian) over Mahalanobis distance of the
// IMM prediction (gate gateChi2), always allowed within minimumGate, never beyond maximumGate. Before that, two
// or more clusters that each fit best to the same confirmed track and together stay within mergeRadius are
// joined (a person's legs, a person seen in pieces): otherwise the extra piece becomes a ghost track.
//
// Persistence against flicker (the sparse L1 sees an object in one cloud and misses it in the next): every track
// has an existence probability r, updated by Bayes each cloud:
//   seen:    r <- r P_D / (r P_D + (1 - r) P_FA)
//   missed:  r <- r (1 - P_D) / (r (1 - P_D) + (1 - r) (1 - P_FA))
// A miss counts only once the track was missed in all missWindow clouds of a window (2: missing from one cloud
// costs nothing), so the output keeps the full cloud rate with the protection of updating every 2nd cloud.
// P_D, the chance the lidar returns a cluster for a real object, falls linearly from detectionProbabilityNear
// at detectionNearRange to detectionProbabilityFar at detectionFarRange (bag go2_all_topics_6_10_11_28: ~1 %
// churn per cloud below 1 m, 12-20 % beyond 4 m; the values are much lower than 1 - churn since misses come in
// runs, and were tuned on that bag for the least flicker), times occludedDetectionFactor when most of the track's bearings are covered by a nearer cluster this
// cloud, and 0 outside [rangeMin, rangeMax]. P_FA, the chance a cluster is noise, is falseAlarmProbability,
// raised for clusters of fewer than falseAlarmPoints points. A new track starts at birthExistence; it is
// published once r >= confirmExistence and stays published while r >= keepExistence (hysteresis); it is
// dropped below deleteExistence, when its position sigma passes maxPositionSigma, or when unseen for
// trackTimeout (occludedTimeout while occluded). r is capped at maxExistence, which sets how many misses in a
// row a well-known object survives (with the defaults ~9 near = 0.6 s; far it is trackTimeout that ends it).
// The measurement variance of each track adapts to how much its centroid really jumps (noiseAdaptRate).
// A published track missed in this cloud is still published (isCoasting) with its last points moved by the
// IMM prediction, its extra position sigma the growth of the IMM position sigma since its last sighting.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "tmp_lidar/motion_model.hpp"
#include "tmp_lidar/types.hpp"

namespace tmp_lidar
{

struct TrackerParams
{
  // association
  double gateChi2 = 9.21;             // 99 % of a 2D Gaussian
  double minimumGate = 0.25;          // m
  double maximumGate = 0.5;           // m
  double mergeRadius = 0.45;          // m, clusters of one track are joined up to this radius

  // measurement
  std::array<double, 2> lidarRangeStd{0.02, 0.005};   // sigma = a + b * range
  double shapeSigmaScale = 0.25;
  double staleCentroidSigma = 0.05;   // m
  // Adaptive measurement noise: per track, the measurement variance is scaled by a running mean (weight
  // noiseAdaptRate) of the normalized innovation squared / 2 (1 when the model fits), clamped to
  // [1, maxNoiseScale]. Real centroids of the sparse L1 jump more than the model says; unscaled, the IMM takes
  // the jumps for motion (phantom velocities) and the gate for new objects.
  double noiseAdaptRate = 0.2;
  double maxNoiseScale = 25.0;

  // motion
  ImmParams imm;
  double minimumSpeed = 0.15;
  double staticPublishProbability = 0.6;
  double unknownVelocitySigma = 1.5;
  int velocitySightings = 2;          // sightings before a velocity is published (before: unknown)
  double minimumVelocitySigma = 0.05;
  double staticVelocitySigma = 0.1;   // sigma reported for wall pieces

  // existence
  int missWindow = 2;                 // clouds: a miss counts once per this many missed clouds in a row
  double birthExistence = 0.3;
  double maxExistence = 0.9999;
  double confirmExistence = 0.7;
  double keepExistence = 0.35;
  double deleteExistence = 0.05;
  double detectionProbabilityNear = 0.7;
  double detectionProbabilityFar = 0.4;
  double detectionNearRange = 1.0;
  double detectionFarRange = 5.0;
  double falseAlarmProbability = 0.1;
  int falseAlarmPoints = 6;
  double occludedDetectionFactor = 0.1;
  double occlusionMargin = 0.3;       // m a cluster must be nearer than the track to hide it
  double rangeMin = 0.35;             // m, the lidar sees nothing outside [rangeMin, rangeMax]
  double rangeMax = 6.0;
  double trackTimeout = 1.0;
  double occludedTimeout = 3.0;
  double maxPositionSigma = 1.5;      // m
};

class ClusterTracker
{
public:
  explicit ClusterTracker(TrackerParams params = {})
  : params_(params) {}

  // Matches clusters (base_link of robot pose `robot`, time t) to tracks and fills their trackId, velocity,
  // velocitySigma, existence and motionMode. Clusters that belong to one track may be joined (the vector
  // shrinks).
  void update(std::vector<Cluster> & clusters, const Pose2 & robot, bool haveOdometry, double t);

  // What to publish after update(): the confirmed clusters of this cloud plus the coasting ones, in base_link
  // of `robot`.
  std::vector<Cluster> publishable(const std::vector<Cluster> & clusters, const Pose2 & robot, double t) const;

  // Probability that the lidar returns a cluster for a visible object at `range`.
  double detectionProbability(double range) const;

  void reset()
  {
    tracks_.clear();
    lastTime_ = -1.0;
  }

private:
  struct Track
  {
    int id = 0;
    ImmFilter filter;               // odom
    double existence = 0.0;
    bool confirmed = false;
    bool isChunk = false;
    std::int64_t chunkKey = 0;
    double lastSeen = 0.0;
    double lastPredicted = 0.0;
    bool occluded = false;
    int sightings = 0;
    int missStreak = 0;             // clouds in a row without a sighting
    double noiseScale = 1.0;
    std::vector<Point2> points;     // odom, last cluster
    Point2 positionAtSighting;      // odom, filter position right after the last update
    double sigmaAtSighting = 0.0;
    Point2 motion;                  // odom, the published velocity (0 when standing)
    double publishedSigma = kNaN;
    int motionMode = -1;
  };

  double measurementVariance(const Cluster & c) const;
  double falseAlarm(const Cluster & c) const;
  void mergeByTracks(std::vector<Cluster> & clusters, const Pose2 & robot) const;
  bool isOccluded(const Track & track, const std::vector<double> & nearest, const Pose2 & robot) const;
  Point2 coastShift(const Track & track) const;

  TrackerParams params_;
  std::vector<Track> tracks_;
  int nextId_ = 0;
  double lastTime_ = -1.0;
};

}  // namespace tmp_lidar
