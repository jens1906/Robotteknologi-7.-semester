#include "tmp_lidar/motion_model.hpp"

#include <algorithm>
#include <cmath>

namespace tmp_lidar
{

namespace
{

using Vec5 = ImmFilter::Vec5;
using Mat5 = ImmFilter::Mat5;

constexpr double kHeld = 1e-6;   // variance added to components a model holds fixed (keeps P positive definite)

// Coordinated turn: constant speed, constant turn rate omega.
Vec5 turnStep(const Vec5 & x, double dt)
{
  const double w = x(4);
  const double wt = w * dt;
  double s, c;   // sin(w dt) / w, (1 - cos(w dt)) / w
  if (std::abs(w) < 1e-4) {
    s = dt - w * w * dt * dt * dt / 6.0;
    c = w * dt * dt / 2.0;
  } else {
    s = std::sin(wt) / w;
    c = (1.0 - std::cos(wt)) / w;
  }
  const double cw = std::cos(wt), sw = std::sin(wt);
  Vec5 out;
  out(0) = x(0) + s * x(2) - c * x(3);
  out(1) = x(1) + c * x(2) + s * x(3);
  out(2) = cw * x(2) - sw * x(3);
  out(3) = sw * x(2) + cw * x(3);
  out(4) = w;
  return out;
}

Mat5 turnJacobian(const Vec5 & x, double dt)
{
  Mat5 J;
  for (int k = 0; k < 5; ++k) {
    const double h = 1e-6 * std::max(1.0, std::abs(x(k)));
    Vec5 plus = x, minus = x;
    plus(k) += h;
    minus(k) -= h;
    J.col(k) = (turnStep(plus, dt) - turnStep(minus, dt)) / (2.0 * h);
  }
  return J;
}

// White acceleration noise of variance q (per axis) over dt on position and velocity.
void addAccelerationNoise(Mat5 & Q, double q, double dt)
{
  const double a = dt * dt * dt * dt / 4.0 * q, b = dt * dt * dt / 2.0 * q, d = dt * dt * q;
  for (int axis = 0; axis < 2; ++axis) {
    Q(axis, axis) += a;
    Q(axis, axis + 2) += b;
    Q(axis + 2, axis) += b;
    Q(axis + 2, axis + 2) += d;
  }
}

double gaussianLogLikelihood(const Eigen::Vector2d & nu, const Eigen::Matrix2d & S)
{
  return -0.5 * nu.dot(S.inverse() * nu) - 0.5 * std::log(std::max(S.determinant(), 1e-300)) -
         std::log(2.0 * M_PI);
}

}  // namespace

void ImmFilter::init(const Point2 & position, double positionVariance, double velocitySigma,
  const ImmParams & params)
{
  params_ = params;
  Vec5 x = Vec5::Zero();
  x(0) = position.x;
  x(1) = position.y;
  double total = 0.0;
  for (int j = 0; j < kModels; ++j) {
    x_[j] = x;
    P_[j] = Mat5::Zero();
    P_[j](0, 0) = P_[j](1, 1) = positionVariance;
    P_[j](2, 2) = P_[j](3, 3) = velocitySigma * velocitySigma;
    P_[j](4, 4) = 0.5 * 0.5;
    if (j == kStatic) {P_[j](2, 2) = P_[j](3, 3) = kHeld;}
    mu_[j] = std::max(1e-6, params.initialModeProbability[static_cast<std::size_t>(j)]);
    total += mu_[j];
  }
  for (auto & m : mu_) {m /= total;}
  combine();
}

void ImmFilter::predict(double dt)
{
  if (!(dt > 0.0)) {return;}

  // Markov model switching over dt: leave the current model with probability 1 - exp(-rate dt).
  const double leave = 1.0 - std::exp(-std::max(0.0, params_.modeSwitchRate) * dt);
  double p[kModels][kModels];
  for (int i = 0; i < kModels; ++i) {
    for (int j = 0; j < kModels; ++j) {p[i][j] = i == j ? 1.0 - leave : leave / (kModels - 1);}
  }

  // Mixing.
  std::array<double, kModels> c{};
  for (int j = 0; j < kModels; ++j) {
    for (int i = 0; i < kModels; ++i) {c[j] += p[i][j] * mu_[i];}
  }
  std::array<Vec5, kModels> x0;
  std::array<Mat5, kModels> P0;
  for (int j = 0; j < kModels; ++j) {
    x0[j] = Vec5::Zero();
    for (int i = 0; i < kModels; ++i) {x0[j] += p[i][j] * mu_[i] / c[j] * x_[i];}
    P0[j] = Mat5::Zero();
    for (int i = 0; i < kModels; ++i) {
      const Vec5 d = x_[i] - x0[j];
      P0[j] += p[i][j] * mu_[i] / c[j] * (P_[i] + d * d.transpose());
    }
  }

  // Model predictions.
  for (int j = 0; j < kModels; ++j) {
    Mat5 F = Mat5::Identity();
    Mat5 Q = Mat5::Zero();
    Vec5 x = x0[j];
    if (j == kStatic) {
      F(2, 2) = F(3, 3) = 0.0;   // omega is carried unchanged (not used), so mixing does not reset CT's
      x = F * x;
      const double q = params_.staticPositionNoise * params_.staticPositionNoise * dt;
      Q(0, 0) = Q(1, 1) = q;
      Q(2, 2) = Q(3, 3) = kHeld;
    } else if (j == kConstantVelocity) {
      F(0, 2) = F(1, 3) = dt;   // omega carried unchanged, as above
      x = F * x;
      addAccelerationNoise(Q, params_.cvAccelerationNoise * params_.cvAccelerationNoise, dt);
      Q(4, 4) = kHeld;
    } else {
      F = turnJacobian(x, dt);
      x = turnStep(x, dt);
      addAccelerationNoise(Q, params_.ctAccelerationNoise * params_.ctAccelerationNoise, dt);
      Q(4, 4) = params_.ctTurnNoise * params_.ctTurnNoise * dt * dt;
    }
    x_[j] = x;
    P_[j] = F * P0[j] * F.transpose() + Q;
    P_[j] = 0.5 * (P_[j] + P_[j].transpose());
  }
  mu_ = c;
  combine();
}

double ImmFilter::mahalanobis2(const Point2 & z, double measurementVariance, double * logDet) const
{
  const Eigen::Vector2d nu(z.x - x_combined_(0), z.y - x_combined_(1));
  const Eigen::Matrix2d S = P_combined_.topLeftCorner<2, 2>() +
    measurementVariance * Eigen::Matrix2d::Identity();
  if (logDet) {*logDet = std::log(std::max(S.determinant(), 1e-300));}
  return nu.dot(S.inverse() * nu);
}

void ImmFilter::update(const Point2 & z, double measurementVariance)
{
  const Eigen::Matrix2d R = measurementVariance * Eigen::Matrix2d::Identity();
  std::array<double, kModels> logLikelihood{};
  for (int j = 0; j < kModels; ++j) {
    const Eigen::Vector2d nu(z.x - x_[j](0), z.y - x_[j](1));
    const Eigen::Matrix2d S = P_[j].topLeftCorner<2, 2>() + R;
    const Eigen::Matrix<double, 5, 2> K = P_[j].leftCols<2>() * S.inverse();
    x_[j] += K * nu;
    Eigen::Matrix<double, 2, 5> H = Eigen::Matrix<double, 2, 5>::Zero();
    H(0, 0) = H(1, 1) = 1.0;
    const Mat5 IKH = Mat5::Identity() - K * H;
    P_[j] = IKH * P_[j] * IKH.transpose() + K * R * K.transpose();   // Joseph form
    logLikelihood[j] = gaussianLogLikelihood(nu, S);
  }
  const double best = *std::max_element(logLikelihood.begin(), logLikelihood.end());
  double total = 0.0;
  for (int j = 0; j < kModels; ++j) {
    mu_[j] *= std::exp(logLikelihood[j] - best);
    total += mu_[j];
  }
  for (auto & m : mu_) {m = std::max(1e-9, m / total);}
  combine();
}

void ImmFilter::combine()
{
  x_combined_ = Vec5::Zero();
  for (int j = 0; j < kModels; ++j) {x_combined_ += mu_[j] * x_[j];}
  P_combined_ = Mat5::Zero();
  for (int j = 0; j < kModels; ++j) {
    const Vec5 d = x_[j] - x_combined_;
    P_combined_ += mu_[j] * (P_[j] + d * d.transpose());
  }
}

Point2 ImmFilter::position() const {return {x_combined_(0), x_combined_(1)};}
Point2 ImmFilter::velocity() const {return {x_combined_(2), x_combined_(3)};}
Eigen::Matrix2d ImmFilter::positionCovariance() const {return P_combined_.topLeftCorner<2, 2>();}
Eigen::Matrix2d ImmFilter::velocityCovariance() const {return P_combined_.block<2, 2>(2, 2);}

namespace
{
double largestSigma(const Eigen::Matrix2d & m)
{
  // Largest eigenvalue of the symmetric part, closed form.
  const double a = m(0, 0), d = m(1, 1), b = 0.5 * (m(0, 1) + m(1, 0));
  const double largest = 0.5 * (a + d) + std::sqrt(0.25 * (a - d) * (a - d) + b * b);
  return std::sqrt(std::max(0.0, largest));
}
}  // namespace

double ImmFilter::positionSigma() const {return largestSigma(positionCovariance());}
double ImmFilter::velocitySigma() const {return largestSigma(velocityCovariance());}

namespace
{
// Mixture of the moving models' velocities: mean and covariance.
void movingMixture(const std::array<ImmFilter::Vec5, ImmFilter::kModels> & x,
  const std::array<ImmFilter::Mat5, ImmFilter::kModels> & P, const std::array<double, ImmFilter::kModels> & mu,
  Eigen::Vector2d & mean, Eigen::Matrix2d & cov)
{
  double total = 0.0;
  mean.setZero();
  for (int j = ImmFilter::kConstantVelocity; j < ImmFilter::kModels; ++j) {
    total += mu[j];
    mean += mu[j] * x[j].segment<2>(2);
  }
  mean /= total;
  cov.setZero();
  for (int j = ImmFilter::kConstantVelocity; j < ImmFilter::kModels; ++j) {
    const Eigen::Vector2d d = x[j].segment<2>(2) - mean;
    cov += mu[j] / total * (P[j].block<2, 2>(2, 2) + d * d.transpose());
  }
}
}  // namespace

Point2 ImmFilter::movingVelocity() const
{
  Eigen::Vector2d mean;
  Eigen::Matrix2d cov;
  movingMixture(x_, P_, mu_, mean, cov);
  return {mean(0), mean(1)};
}

double ImmFilter::movingVelocitySigma() const
{
  Eigen::Vector2d mean;
  Eigen::Matrix2d cov;
  movingMixture(x_, P_, mu_, mean, cov);
  return largestSigma(cov);
}

int ImmFilter::likeliestMode() const
{
  return static_cast<int>(std::max_element(mu_.begin(), mu_.end()) - mu_.begin());
}

}  // namespace tmp_lidar
