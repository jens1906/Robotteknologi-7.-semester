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
#pragma once

#include <array>

#include <Eigen/Dense>

#include "tmp_lidar/types.hpp"

namespace tmp_lidar
{

struct ImmParams
{
  double staticPositionNoise = 0.05;   // m / sqrt(s)
  double cvAccelerationNoise = 1.5;    // m/s^2
  double ctAccelerationNoise = 0.8;    // m/s^2
  double ctTurnNoise = 0.3;            // rad/s^2
  double modeSwitchRate = 0.3;         // 1/s, rate of leaving a model
  std::array<double, 3> initialModeProbability{0.6, 0.3, 0.1};
};

class ImmFilter
{
public:
  static constexpr int kStatic = 0;
  static constexpr int kConstantVelocity = 1;
  static constexpr int kTurning = 2;
  static constexpr int kModels = 3;

  using Vec5 = Eigen::Matrix<double, 5, 1>;
  using Mat5 = Eigen::Matrix<double, 5, 5>;

  ImmFilter() = default;

  // Starts at `position` (variance positionVariance per axis), velocity unknown with velocitySigma.
  void init(const Point2 & position, double positionVariance, double velocitySigma, const ImmParams & params);

  // Mixes the models and moves the state dt seconds ahead (dt <= 0 does nothing).
  void predict(double dt);

  // Squared Mahalanobis distance of z from the predicted position and ln det of its covariance, for a
  // measurement of variance measurementVariance per axis.
  double mahalanobis2(const Point2 & z, double measurementVariance, double * logDet = nullptr) const;

  // Measurement of the position (after predict()).
  void update(const Point2 & z, double measurementVariance);

  Point2 position() const;
  Point2 velocity() const;
  Eigen::Matrix2d positionCovariance() const;
  Eigen::Matrix2d velocityCovariance() const;
  double positionSigma() const;   // sqrt of the largest eigenvalue of the position covariance
  double velocitySigma() const;   // same, velocity
  // Velocity and its sigma given that the object moves (the static model left out of the mix): the mixed
  // velocity is pulled toward 0 by the static model's share; whether it moves at all is modeProbability(kStatic).
  Point2 movingVelocity() const;
  double movingVelocitySigma() const;
  double modeProbability(int model) const {return mu_[static_cast<std::size_t>(model)];}
  int likeliestMode() const;

private:
  void combine();

  ImmParams params_;
  std::array<Vec5, kModels> x_{};
  std::array<Mat5, kModels> P_{};
  std::array<double, kModels> mu_{};
  Vec5 x_combined_ = Vec5::Zero();
  Mat5 P_combined_ = Mat5::Identity();
};

}  // namespace tmp_lidar
