// Tracking of the clusters over time. No ROS.
// Three parts:
// - Assignment: best one-to-one pairing of clusters and tracks (Hungarian method).
// - Motion model: the IMM Kalman filter each track uses for its position and velocity.
// - Cluster tracker: ids, existence, coasting; uses the two parts above.
#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "lidar_obstacles/types.hpp"

namespace lidar_obstacles
{

// ===== Assignment ==================================================================================================
// Minimum cost one-to-one assignment (Hungarian method, O(n^3)). No ROS.
//
// Used by the tracker to pair this cloud's clusters (rows) with the existing tracks (columns) so that the total
// cost (how badly each cluster fits each track) is as small as possible. Unlike a greedy "take the best pair
// first" approach, it can give one pair a slightly worse match so that two other pairs both fit: this keeps
// the ids of two people walking past each other from swapping.

// cost[row][column]; entries >= forbiddenCost are never assigned. Returns for each row the assigned column or -1.
// The matrix does not have to be square: rows or columns left over stay unassigned (-1).
std::vector<int> SolveAssignment(const std::vector<std::vector<double>> & cost, double forbiddenCost);

// ===== Motion model ================================================================================================
// Motion model of one tracked obstacle: an IMM (interacting multiple model) Kalman filter. No ROS.
//
// State [x, y, vx, vy, omega] in odom. Three models run side by side and are mixed every step by how well
// each explained the measurements:
// - static:   standing still; velocity held at 0, position drifts by staticPositionNoise;
// - CV:       constant velocity, white acceleration noise cvAccelerationNoise;
// - CT:       coordinated turn (constant speed and turn rate), EKF with a numeric Jacobian,
//             ctAccelerationNoise on the velocity and ctTurnNoise on the turn rate.
// Static and CV carry the turn rate unchanged without using it, so the mixing does not reset CT's estimate.
// A person who stops is picked up by the static model within a few clouds (velocity goes to 0 quickly instead of
// decaying through a low-pass), one who walks a curve by CT. Models switch at modeSwitchRate per second.
// The measurement is a 2D position (cluster centroid) with an isotropic sigma.
//
// Background, in short: a Kalman filter keeps an estimate (the state) and how uncertain it is (the covariance).
// Predict() moves the estimate forward in time and makes it more uncertain; Update() pulls it toward a new
// measurement, weighted by how uncertain each one is. The IMM runs three such filters with different
// assumptions about the motion and keeps a probability for each, based on how well it predicted the
// measurements. The tracker owns one ImmFilter per track.

// Velocity of an object, assuming it moves, and how uncertain that is (one sigma).
struct MovingVelocity
{
  Point2 velocity;
  double sigma = NOT_A_NUMBER;
};

// Settings of the motion models (staticPositionNoise, cvAccelerationNoise, ... in the yaml).
// Larger noise = the model allows faster changes, so it follows quick moves but jitters more.
struct ImmParams
{
  double staticPositionNoise = 0.05;   // m / sqrt(s)
  double cvAccelerationNoise = 1.5;    // m/s^2
  double ctAccelerationNoise = 0.8;    // m/s^2
  double ctTurnNoise = 0.3;            // rad/s^2
  double modeSwitchRate = 0.3;         // 1/s, rate of leaving a model
  std::array<double, 3> initialModeProbability{0.6, 0.3, 0.1};   // static, CV, turning at the start of a track
};

class ImmFilter
{
public:
  // Index of each model in the arrays below.
  static constexpr int STATIC_MODEL = 0;
  static constexpr int CONSTANT_VELOCITY_MODEL = 1;
  static constexpr int TURNING_MODEL = 2;
  static constexpr int MODEL_COUNT = 3;

  // The state [x, y, vx, vy, omega] and its 5x5 covariance.
  using StateVector = Eigen::Matrix<double, 5, 1>;
  using StateMatrix = Eigen::Matrix<double, 5, 5>;

  ImmFilter() = default;

  // Starts at `position` (variance positionVariance per axis), velocity unknown with velocitySigma.
  void Initialize(const Point2 & position, double positionVariance, double velocitySigma,
    const ImmParams & immParams);

  // Mixes the models and moves the state elapsedSeconds ahead (elapsedSeconds <= 0 does nothing).
  void Predict(double elapsedSeconds);

  // Squared Mahalanobis distance of measurement from the predicted position and ln det of its covariance, for a
  // measurement of variance measurementVariance per axis.
  // Mahalanobis distance = distance measured in "standard deviations": 1 m away from a very uncertain track is
  // close, 1 m away from a very certain one is far. The tracker uses it to decide which cluster belongs to
  // which track.
  double CalculateMahalanobisSquared(const Point2 & measurement, double measurementVariance,
    double * logDeterminant = nullptr) const;

  // Measurement of the position (after Predict()).
  void Update(const Point2 & measurement, double measurementVariance);

  // The combined (probability-weighted) position and how uncertain it is.
  Point2 GetPosition() const;
  double GetPositionSigma() const;   // sqrt of the largest eigenvalue of the position covariance
  // Velocity and its sigma given that the object moves (the static model left out of the mix): the mixed
  // velocity is pulled toward 0 by the static model's share; whether it moves at all is
  // GetModeProbability(STATIC_MODEL).
  MovingVelocity CalculateMovingVelocity() const;
  // Probability (0..1) that the object currently follows the given model.
  double GetModeProbability(int model) const {return modelProbabilities[static_cast<std::size_t>(model)];}
  // The model with the highest probability (shown in the RViz labels).
  int FindLikeliestMode() const;

private:
  void CombineModels();

  ImmParams params;
  std::array<StateVector, MODEL_COUNT> modelStates{};        // each model's own estimate
  std::array<StateMatrix, MODEL_COUNT> modelCovariances{};   // each model's own uncertainty
  std::array<double, MODEL_COUNT> modelProbabilities{};      // how likely each model is (sums to 1)
  StateVector combinedState = StateVector::Zero();           // probability-weighted mix of the three
  StateMatrix combinedCovariance = StateMatrix::Identity();
};

// ===== Cluster tracker =============================================================================================
// Gives clusters a stable id across clouds, a velocity over the ground, and keeps the published set steady.
// No ROS.
//
// Tracks live in odom, so the robot's own motion is removed.
//
// Motion: each track runs an IMM Kalman filter (motion model section above: static / constant velocity / turning) on the
// cluster's centroid, with measurement sigma = lidar range sigma + shapeSigmaScale * cluster radius
// (a partial view moves the centroid). Before
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
//
// In the pipeline: clustering -> ClusterTracker::Update -> FindPublishable -> fusion. One Update per cloud.

// Settings of the tracker (same names in the yaml). The groups follow the steps described above.
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
  explicit ClusterTracker(TrackerParams trackerParams = {})
  : params(trackerParams) {}

  // Matches clusters (base_link of robot pose `robot`, time cloudSeconds) to tracks and fills their trackId,
  // velocity, velocitySigma, existence and motionMode. Clusters that belong to one track may be joined (the
  // vector shrinks).
  void Update(std::vector<Cluster> & clusters, const Pose2 & robot, bool hasOdometry, double cloudSeconds);

  // What to publish after Update(): the confirmed clusters of this cloud plus the coasting ones, in base_link
  // of `robot`.
  std::vector<Cluster> FindPublishable(const std::vector<Cluster> & clusters, const Pose2 & robot,
    double cloudSeconds) const;

  // Probability that the lidar returns a cluster for a visible object at `range`.
  double CalculateDetectionProbability(double range) const;

  // Forget all tracks (called when odometry comes back after a gap).
  void Reset()
  {
    tracks.clear();
    lastSeconds = -1.0;
  }

private:
  // Everything remembered about one tracked object between clouds.
  struct Track
  {
    int id = 0;                     // stable id, published as trackId (and the RViz marker id)
    ImmFilter filter;               // odom
    double existence = 0.0;         // probability that the object is real (0..1)
    bool confirmed = false;         // published (existence passed confirmExistence)
    bool isChunk = false;           // a wall piece, followed by its chunkKey instead of the filter
    std::int64_t chunkKey = 0;
    double lastSeen = 0.0;          // time of the last cloud a cluster was matched to it
    double lastPredicted = 0.0;     // time the filter was last predicted to
    bool occluded = false;          // hidden behind a nearer cluster in the last cloud
    int sightings = 0;              // how many clouds it was seen in
    int missStreak = 0;             // clouds in a row without a sighting
    double noiseScale = 1.0;        // adaptive measurement noise factor (1 .. maxNoiseScale)
    std::vector<Point2> points;     // odom, last cluster
    Point2 positionAtSighting;      // odom, filter position right after the last update
    double sigmaAtSighting = 0.0;   // filter position sigma right after the last update
    Point2 motion;                  // odom, the published velocity (0 when standing)
    double publishedSigma = NOT_A_NUMBER;   // the velocity sigma that was published
    int motionMode = -1;            // likeliest motion model (debug)
  };

  // Helpers of Update(), explained in cluster_tracker.cpp.
  double CalculateMeasurementVariance(const Cluster & cluster) const;
  double CalculateFalseAlarmProbability(const Cluster & cluster) const;
  void MergeClustersByTrack(std::vector<Cluster> & clusters, const Pose2 & robot) const;
  bool IsOccluded(const std::vector<Point2> & basePointsNow, const std::vector<double> & nearestRangePerBearing) const;
  std::vector<Point2> FindTrackPointsNow(const Track & track, const Pose2 & robot) const;
  std::optional<double> CalculateGatedDistance(const Track & track, const Point2 & measurement, double variance,
    double * logDeterminant = nullptr) const;

  TrackerParams params;
  std::vector<Track> tracks;     // all live tracks, confirmed or not
  int nextTrackId = 0;           // id given to the next new track
  double lastSeconds = -1.0;     // time of the last Update, to notice time jumping back
};

}  // namespace lidar_obstacles
