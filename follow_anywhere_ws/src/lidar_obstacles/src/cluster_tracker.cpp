// Tracking of the clusters over time. No ROS. See cluster_tracker.hpp.
#include "lidar_obstacles/cluster_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <optional>

namespace lidar_obstacles
{

// ===== Assignment ==================================================================================================
// Hungarian method (also called Kuhn-Munkres): finds the one-to-one pairing of rows and columns with the
// smallest total cost. Used by the tracker to pair clusters (rows) with tracks (columns).
//
// How it works, in short: every row and every column gets a "potential" (a number subtracted from the costs).
// The potentials are adjusted so that, for the pairs chosen, cost - rowPotential - columnPotential is 0 and is
// never negative for any other pair. When that holds for a complete pairing, no other pairing can be cheaper.
// Rows are added one at a time; each new row searches for the cheapest way to fit in, possibly moving rows that
// were already placed to other columns (an "augmenting path").

std::vector<int> SolveAssignment(const std::vector<std::vector<double>> & cost, double forbiddenCost)
{
  // No rows or no columns: nothing to assign.
  const int rowCount = static_cast<int>(cost.size());
  if (rowCount == 0) {
    return {};
  }
  const int columnCount = static_cast<int>(cost[0].size());
  std::vector<int> assignedColumns(static_cast<std::size_t>(rowCount), -1);
  if (columnCount == 0) {
    return assignedColumns;
  }

  // Square matrix padded with dummy entries; a forbidden pair costs more than leaving both unassigned.
  // The method needs as many rows as columns, so the smaller side is padded with dummy rows/columns that cost
  // half the forbidden cost. Pairing a real row with a dummy column means "not assigned", and that is always
  // cheaper than a forbidden pair, so forbidden pairs are only ever chosen when nothing else is possible
  // (and they are removed again at the end).
  const int size = std::max(rowCount, columnCount);
  auto paddedCost = [&](int row, int column) {
      if (row >= rowCount || column >= columnCount) {
        return 0.5 * forbiddenCost;
      }
      return std::min(cost[static_cast<std::size_t>(row)][static_cast<std::size_t>(column)], forbiddenCost);
    };

  // e-maxx formulation, 1-based.
  // Index 0 is a helper column used while placing a new row; real rows and columns are 1..size.
  // rowPotential / columnPotential: the potentials described at the top of the file.
  // rowOfColumn[column]: the row currently placed in that column (0 = free).
  // previousColumn[column]: on the search path, the column we came from, used to walk the path back.
  const double infinity = std::numeric_limits<double>::infinity();
  const auto vectorSize = static_cast<std::size_t>(size + 1);
  std::vector<double> rowPotential(vectorSize, 0.0);
  std::vector<double> columnPotential(vectorSize, 0.0);
  std::vector<int> rowOfColumn(vectorSize, 0);
  std::vector<int> previousColumn(vectorSize, 0);
  for (int row = 1; row <= size; ++row) {
    // Place this row: start from the helper column 0, which temporarily "holds" the new row.
    rowOfColumn[0] = row;
    int currentColumn = 0;
    // minimumSlack[column]: the smallest reduced cost found so far to reach that column on the search path.
    std::vector<double> minimumSlack(vectorSize, infinity);
    std::vector<bool> columnUsed(vectorSize, false);
    do {
      // Visit the row sitting in currentColumn and look at every column not yet on the path.
      columnUsed[static_cast<std::size_t>(currentColumn)] = true;
      const int currentRow = rowOfColumn[static_cast<std::size_t>(currentColumn)];
      double delta = infinity;
      int nextColumn = 0;
      for (int column = 1; column <= size; ++column) {
        const auto columnIndex = static_cast<std::size_t>(column);
        if (columnUsed[columnIndex]) {
          continue;
        }
        // Reduced cost: how much more than "free" (0) it costs to put currentRow in this column.
        const double reducedCost = paddedCost(currentRow - 1, column - 1) -
          rowPotential[static_cast<std::size_t>(currentRow)] - columnPotential[columnIndex];
        if (reducedCost < minimumSlack[columnIndex]) {
          minimumSlack[columnIndex] = reducedCost;
          previousColumn[columnIndex] = currentColumn;
        }
        // Remember the column that is cheapest to reach next.
        if (minimumSlack[columnIndex] < delta) {
          delta = minimumSlack[columnIndex];
          nextColumn = column;
        }
      }
      // Shift the potentials by delta so the cheapest column becomes reachable at reduced cost 0,
      // without making any reduced cost negative.
      for (int column = 0; column <= size; ++column) {
        const auto columnIndex = static_cast<std::size_t>(column);
        if (columnUsed[columnIndex]) {
          rowPotential[static_cast<std::size_t>(rowOfColumn[columnIndex])] += delta;
          columnPotential[columnIndex] -= delta;
        } else {
          minimumSlack[columnIndex] -= delta;
        }
      }
      currentColumn = nextColumn;
      // Stop when we reached a free column: the new row can be fitted in along this path.
    } while (rowOfColumn[static_cast<std::size_t>(currentColumn)] != 0);
    // Walk the path back, moving each row one step along it. This frees the way for the new row.
    do {
      const int nextColumn = previousColumn[static_cast<std::size_t>(currentColumn)];
      rowOfColumn[static_cast<std::size_t>(currentColumn)] = rowOfColumn[static_cast<std::size_t>(nextColumn)];
      currentColumn = nextColumn;
    } while (currentColumn != 0);
  }

  // Read the result back in 0-based indices. Pairs with a dummy row/column, or with a forbidden cost,
  // are left at -1 (not assigned).
  for (int column = 1; column <= size; ++column) {
    const int row = rowOfColumn[static_cast<std::size_t>(column)] - 1;
    const int costColumn = column - 1;
    if (row < rowCount && costColumn < columnCount &&
      cost[static_cast<std::size_t>(row)][static_cast<std::size_t>(costColumn)] < forbiddenCost)
    {
      assignedColumns[static_cast<std::size_t>(row)] = costColumn;
    }
  }
  return assignedColumns;
}

// ===== Motion model ================================================================================================
// IMM Kalman filter. See the motion model section of cluster_tracker.hpp.

// Helpers only used in this file.
namespace
{

// Short names for the per-model arrays.
using StateVector = ImmFilter::StateVector;
using StateMatrix = ImmFilter::StateMatrix;
using ModelStates = std::array<StateVector, ImmFilter::MODEL_COUNT>;
using ModelCovariances = std::array<StateMatrix, ImmFilter::MODEL_COUNT>;
using ModelProbabilities = std::array<double, ImmFilter::MODEL_COUNT>;

// variance added to components a model holds fixed (keeps the covariance positive definite)
// A covariance with an exact 0 on the diagonal cannot be inverted; this tiny value avoids that.
constexpr double HELD_VARIANCE = 1e-6;

// Coordinated turn: constant speed, constant turn rate omega.
// Moves the state elapsedSeconds ahead along a circular arc: the velocity vector rotates by omega * dt and the
// position follows the arc. For a turn rate near 0 the exact formulas divide by ~0, so a Taylor series
// (the same values without the division) is used instead; it reduces to a straight line.
StateVector PredictTurnStep(const StateVector & state, double elapsedSeconds)
{
  const double turnRate = state(4);
  const double turnAngle = turnRate * elapsedSeconds;
  double sineTerm;     // sin(w dt) / w
  double cosineTerm;   // (1 - cos(w dt)) / w
  if (std::abs(turnRate) < 1e-4) {
    sineTerm = elapsedSeconds - turnRate * turnRate * elapsedSeconds * elapsedSeconds * elapsedSeconds / 6.0;
    cosineTerm = turnRate * elapsedSeconds * elapsedSeconds / 2.0;
  } else {
    sineTerm = std::sin(turnAngle) / turnRate;
    cosineTerm = (1.0 - std::cos(turnAngle)) / turnRate;
  }
  const double cosineAngle = std::cos(turnAngle);
  const double sineAngle = std::sin(turnAngle);
  StateVector nextState;
  nextState(0) = state(0) + sineTerm * state(2) - cosineTerm * state(3);   // x along the arc
  nextState(1) = state(1) + cosineTerm * state(2) + sineTerm * state(3);   // y along the arc
  nextState(2) = cosineAngle * state(2) - sineAngle * state(3);            // velocity rotated by the turn
  nextState(3) = sineAngle * state(2) + cosineAngle * state(3);
  nextState(4) = turnRate;                                                 // turn rate stays the same
  return nextState;
}

// The turn model is not linear, so the filter needs its Jacobian (how each output changes with each input) to
// carry the uncertainty forward (extended Kalman filter). It is computed numerically: nudge each state
// component a tiny bit up and down and see how the predicted state changes (central difference).
StateMatrix CalculateTurnJacobian(const StateVector & state, double elapsedSeconds)
{
  StateMatrix jacobian;
  for (int component = 0; component < 5; ++component) {
    const double stepSize = 1e-6 * std::max(1.0, std::abs(state(component)));
    StateVector statePlus = state;
    StateVector stateMinus = state;
    statePlus(component) += stepSize;
    stateMinus(component) -= stepSize;
    jacobian.col(component) =
      (PredictTurnStep(statePlus, elapsedSeconds) - PredictTurnStep(stateMinus, elapsedSeconds)) / (2.0 * stepSize);
  }
  return jacobian;
}

// White acceleration noise of variance accelerationVariance (per axis) over elapsedSeconds on position and velocity.
// Models "the object may speed up or slow down a little at any moment". A random acceleration a over dt changes
// the velocity by a*dt and the position by a*dt^2/2, which gives these standard terms (and the cross terms,
// because position and velocity errors are linked).
void AddAccelerationNoise(StateMatrix & processNoise, double accelerationVariance, double elapsedSeconds)
{
  const double secondsSquared = elapsedSeconds * elapsedSeconds;
  const double positionNoise = secondsSquared * secondsSquared / 4.0 * accelerationVariance;
  const double crossNoise = secondsSquared * elapsedSeconds / 2.0 * accelerationVariance;
  const double velocityNoise = secondsSquared * accelerationVariance;
  for (int axis = 0; axis < 2; ++axis) {
    processNoise(axis, axis) += positionNoise;
    processNoise(axis, axis + 2) += crossNoise;
    processNoise(axis + 2, axis) += crossNoise;
    processNoise(axis + 2, axis + 2) += velocityNoise;
  }
}

// Log of the 2D Gaussian probability density of the innovation (measurement minus prediction).
// High = the model predicted the measurement well. Logs are used because the plain densities can be so small
// that they round to 0.
double CalculateGaussianLogLikelihood(const Eigen::Vector2d & innovation, const Eigen::Matrix2d & innovationCovariance)
{
  return -0.5 * innovation.dot(innovationCovariance.inverse() * innovation) -
         0.5 * std::log(std::max(innovationCovariance.determinant(), 1e-300)) - std::log(2.0 * M_PI);
}

// Turns a 2x2 covariance into one sigma: the square root of its largest eigenvalue, i.e. the standard deviation
// along the most uncertain direction (a worst-case single number for an elliptical uncertainty).
double CalculateLargestSigma(const Eigen::Matrix2d & covariance)
{
  // Largest eigenvalue of the symmetric part, closed form.
  const double firstVariance = covariance(0, 0);
  const double secondVariance = covariance(1, 1);
  const double crossVariance = 0.5 * (covariance(0, 1) + covariance(1, 0));
  const double largestEigenvalue = 0.5 * (firstVariance + secondVariance) +
    std::sqrt(0.25 * (firstVariance - secondVariance) * (firstVariance - secondVariance) +
    crossVariance * crossVariance);
  return std::sqrt(std::max(0.0, largestEigenvalue));
}

}  // namespace

void ImmFilter::Initialize(const Point2 & position, double positionVariance, double velocitySigma,
  const ImmParams & immParams)
{
  params = immParams;
  // All models start at the measured position with velocity 0 and turn rate 0.
  StateVector initialState = StateVector::Zero();
  initialState(0) = position.x;
  initialState(1) = position.y;
  double totalProbability = 0.0;
  for (int model = 0; model < MODEL_COUNT; ++model) {
    modelStates[model] = initialState;
    // Uncertainty: position as measured, velocity very uncertain (unknown), turn rate 0.5 rad/s.
    modelCovariances[model] = StateMatrix::Zero();
    modelCovariances[model](0, 0) = modelCovariances[model](1, 1) = positionVariance;
    modelCovariances[model](2, 2) = modelCovariances[model](3, 3) = velocitySigma * velocitySigma;
    modelCovariances[model](4, 4) = 0.5 * 0.5;
    // The static model is sure its velocity is 0.
    if (model == STATIC_MODEL) {
      modelCovariances[model](2, 2) = modelCovariances[model](3, 3) = HELD_VARIANCE;
    }
    // Starting probabilities (never exactly 0, so a model can always win back).
    modelProbabilities[model] = std::max(1e-6, params.initialModeProbability[static_cast<std::size_t>(model)]);
    totalProbability += modelProbabilities[model];
  }
  // Normalise so the probabilities sum to 1.
  for (auto & modelProbability : modelProbabilities) {
    modelProbability /= totalProbability;
  }
  CombineModels();
}

void ImmFilter::Predict(double elapsedSeconds)
{
  if (!(elapsedSeconds > 0.0)) {
    return;
  }

  // Markov model switching over elapsedSeconds: leave the current model with probability 1 - exp(-rate dt).
  // switchProbability[from][to]: chance the object changed from one kind of motion to another in this step
  // (staying is likely, switching to either of the other two shares the rest).
  const double leaveProbability = 1.0 - std::exp(-std::max(0.0, params.modeSwitchRate) * elapsedSeconds);
  double switchProbability[MODEL_COUNT][MODEL_COUNT];
  for (int fromModel = 0; fromModel < MODEL_COUNT; ++fromModel) {
    for (int toModel = 0; toModel < MODEL_COUNT; ++toModel) {
      switchProbability[fromModel][toModel] =
        fromModel == toModel ? 1.0 - leaveProbability : leaveProbability / (MODEL_COUNT - 1);
    }
  }

  // Mixing.
  // Before predicting, each model starts from a blend of all three models' estimates, weighted by how likely it
  // is that the object switched from that model to this one. This is the "interacting" part of the IMM: it lets
  // a model that was unlikely catch up quickly when the motion changes.
  // predictedProbabilities[to]: probability of each model after the possible switch.
  ModelProbabilities predictedProbabilities{};
  for (int toModel = 0; toModel < MODEL_COUNT; ++toModel) {
    for (int fromModel = 0; fromModel < MODEL_COUNT; ++fromModel) {
      predictedProbabilities[toModel] += switchProbability[fromModel][toModel] * modelProbabilities[fromModel];
    }
  }
  // The blended starting state and covariance of each model.
  ModelStates mixedStates;
  ModelCovariances mixedCovariances;
  for (int toModel = 0; toModel < MODEL_COUNT; ++toModel) {
    mixedStates[toModel] = StateVector::Zero();
    for (int fromModel = 0; fromModel < MODEL_COUNT; ++fromModel) {
      mixedStates[toModel] += switchProbability[fromModel][toModel] * modelProbabilities[fromModel] /
        predictedProbabilities[toModel] * modelStates[fromModel];
    }
    mixedCovariances[toModel] = StateMatrix::Zero();
    for (int fromModel = 0; fromModel < MODEL_COUNT; ++fromModel) {
      const StateVector stateOffset = modelStates[fromModel] - mixedStates[toModel];
      mixedCovariances[toModel] += switchProbability[fromModel][toModel] * modelProbabilities[fromModel] /
        predictedProbabilities[toModel] * (modelCovariances[fromModel] + stateOffset * stateOffset.transpose());
    }
  }

  // Model predictions.
  // Each model moves its blended state forward with its own motion assumption. transitionMatrix says how the
  // state changes over the step; processNoise how much uncertainty the step adds.
  for (int model = 0; model < MODEL_COUNT; ++model) {
    StateMatrix transitionMatrix = StateMatrix::Identity();
    StateMatrix processNoise = StateMatrix::Zero();
    StateVector predictedState = mixedStates[model];
    if (model == STATIC_MODEL) {
      // Static: velocity forced to 0, position stays but may drift a little.
      transitionMatrix(2, 2) = transitionMatrix(3, 3) = 0.0;   // omega is carried unchanged (not used), so mixing does not reset CT's
      predictedState = transitionMatrix * predictedState;
      const double positionNoise = params.staticPositionNoise * params.staticPositionNoise * elapsedSeconds;
      processNoise(0, 0) = processNoise(1, 1) = positionNoise;
      processNoise(2, 2) = processNoise(3, 3) = HELD_VARIANCE;
    } else if (model == CONSTANT_VELOCITY_MODEL) {
      // Constant velocity: position += velocity * dt, velocity stays (plus random acceleration).
      transitionMatrix(0, 2) = transitionMatrix(1, 3) = elapsedSeconds;   // omega carried unchanged, as above
      predictedState = transitionMatrix * predictedState;
      AddAccelerationNoise(processNoise, params.cvAccelerationNoise * params.cvAccelerationNoise, elapsedSeconds);
      processNoise(4, 4) = HELD_VARIANCE;
    } else {
      // Turning: move along an arc (non-linear, so the Jacobian is used for the covariance).
      transitionMatrix = CalculateTurnJacobian(predictedState, elapsedSeconds);
      predictedState = PredictTurnStep(predictedState, elapsedSeconds);
      AddAccelerationNoise(processNoise, params.ctAccelerationNoise * params.ctAccelerationNoise, elapsedSeconds);
      processNoise(4, 4) = params.ctTurnNoise * params.ctTurnNoise * elapsedSeconds * elapsedSeconds;
    }
    // Standard Kalman prediction of the covariance: P = F P F^T + Q, then made exactly symmetric again
    // (rounding errors can make it slightly unsymmetric over many steps).
    modelStates[model] = predictedState;
    modelCovariances[model] = transitionMatrix * mixedCovariances[model] * transitionMatrix.transpose() + processNoise;
    modelCovariances[model] = 0.5 * (modelCovariances[model] + modelCovariances[model].transpose());
  }
  modelProbabilities = predictedProbabilities;
  CombineModels();
}

double ImmFilter::CalculateMahalanobisSquared(const Point2 & measurement, double measurementVariance,
  double * logDeterminant) const
{
  // Innovation: how far the measurement is from where the filter expects the object.
  // Its covariance: the filter's own position uncertainty + the measurement's uncertainty.
  const Eigen::Vector2d innovation(measurement.x - combinedState(0), measurement.y - combinedState(1));
  const Eigen::Matrix2d innovationCovariance = combinedCovariance.topLeftCorner<2, 2>() +
    measurementVariance * Eigen::Matrix2d::Identity();
  // Optionally also return ln(det) of that covariance: the tracker adds it to the cost so that a very
  // uncertain track does not win every cluster just because its Mahalanobis distances are small.
  if (logDeterminant) {
    *logDeterminant = std::log(std::max(innovationCovariance.determinant(), 1e-300));
  }
  return innovation.dot(innovationCovariance.inverse() * innovation);
}

void ImmFilter::Update(const Point2 & measurement, double measurementVariance)
{
  const Eigen::Matrix2d measurementNoise = measurementVariance * Eigen::Matrix2d::Identity();
  ModelProbabilities logLikelihood{};
  for (int model = 0; model < MODEL_COUNT; ++model) {
    // Standard Kalman update of each model:
    // innovation = measurement - predicted position; innovationCovariance = how uncertain that difference is.
    const Eigen::Vector2d innovation(measurement.x - modelStates[model](0), measurement.y - modelStates[model](1));
    const Eigen::Matrix2d innovationCovariance = modelCovariances[model].topLeftCorner<2, 2>() + measurementNoise;
    // Kalman gain: how much of the innovation to apply to each state component (also corrects the velocity,
    // since a position error says something about the speed).
    const Eigen::Matrix<double, 5, 2> kalmanGain =
      modelCovariances[model].leftCols<2>() * innovationCovariance.inverse();
    modelStates[model] += kalmanGain * innovation;
    // The measurement sees x and y only (the first two state components).
    Eigen::Matrix<double, 2, 5> measurementMatrix = Eigen::Matrix<double, 2, 5>::Zero();
    measurementMatrix(0, 0) = measurementMatrix(1, 1) = 1.0;
    // New covariance in "Joseph form": numerically safer than the short formula, it always stays symmetric and
    // positive definite.
    const StateMatrix josephFactor = StateMatrix::Identity() - kalmanGain * measurementMatrix;
    modelCovariances[model] = josephFactor * modelCovariances[model] * josephFactor.transpose() +
      kalmanGain * measurementNoise * kalmanGain.transpose();   // Joseph form
    // How well this model predicted the measurement.
    logLikelihood[model] = CalculateGaussianLogLikelihood(innovation, innovationCovariance);
  }
  // Update the model probabilities: multiply each by its likelihood (Bayes) and normalise to sum 1.
  // The best log likelihood is subtracted first so exp() does not underflow to 0 for all models.
  const double bestLogLikelihood = *std::max_element(logLikelihood.begin(), logLikelihood.end());
  double totalProbability = 0.0;
  for (int model = 0; model < MODEL_COUNT; ++model) {
    modelProbabilities[model] *= std::exp(logLikelihood[model] - bestLogLikelihood);
    totalProbability += modelProbabilities[model];
  }
  // Never let a probability reach exactly 0, so every model can still come back.
  for (auto & modelProbability : modelProbabilities) {
    modelProbability = std::max(1e-9, modelProbability / totalProbability);
  }
  CombineModels();
}

// The combined estimate the rest of the code reads: the probability-weighted average of the three models, and
// a covariance that also includes how much the models disagree with each other.
void ImmFilter::CombineModels()
{
  combinedState = StateVector::Zero();
  for (int model = 0; model < MODEL_COUNT; ++model) {
    combinedState += modelProbabilities[model] * modelStates[model];
  }
  combinedCovariance = StateMatrix::Zero();
  for (int model = 0; model < MODEL_COUNT; ++model) {
    const StateVector stateOffset = modelStates[model] - combinedState;
    combinedCovariance += modelProbabilities[model] * (modelCovariances[model] + stateOffset * stateOffset.transpose());
  }
}

// Combined position (x, y) and its worst-direction sigma.
Point2 ImmFilter::GetPosition() const {return {combinedState(0), combinedState(1)};}
double ImmFilter::GetPositionSigma() const {return CalculateLargestSigma(combinedCovariance.topLeftCorner<2, 2>());}

// Velocity assuming the object moves, and its uncertainty as one sigma.
// Only the constant velocity and turning models are mixed (weighted by their probabilities, renormalised to
// sum to 1). The static model is left out, otherwise its "velocity 0" would pull the speed of a walking
// person down.
MovingVelocity ImmFilter::CalculateMovingVelocity() const
{
  double totalProbability = 0.0;
  Eigen::Vector2d meanVelocity = Eigen::Vector2d::Zero();
  for (int model = CONSTANT_VELOCITY_MODEL; model < MODEL_COUNT; ++model) {
    totalProbability += modelProbabilities[model];
    meanVelocity += modelProbabilities[model] * modelStates[model].segment<2>(2);
  }
  meanVelocity /= totalProbability;
  // Covariance of a mixture = each model's own covariance + how far its mean is from the mixed mean.
  Eigen::Matrix2d velocityCovariance = Eigen::Matrix2d::Zero();
  for (int model = CONSTANT_VELOCITY_MODEL; model < MODEL_COUNT; ++model) {
    const Eigen::Vector2d velocityOffset = modelStates[model].segment<2>(2) - meanVelocity;
    velocityCovariance += modelProbabilities[model] / totalProbability *
      (modelCovariances[model].block<2, 2>(2, 2) + velocityOffset * velocityOffset.transpose());
  }
  return {{meanVelocity(0), meanVelocity(1)}, CalculateLargestSigma(velocityCovariance)};
}

// Index of the model with the highest probability.
int ImmFilter::FindLikeliestMode() const
{
  return static_cast<int>(
    std::max_element(modelProbabilities.begin(), modelProbabilities.end()) - modelProbabilities.begin());
}

// ===== Cluster tracker =============================================================================================
// Cluster tracker: stable ids, velocities and flicker-free publishing. See cluster_tracker.hpp for the method.

// Helpers only used in this file.
namespace
{

// Tracker values
// BEARING_BINS: the 360 deg around the robot is split into 1 deg slices for the occlusion check.
// FORBIDDEN_COST: cost of a cluster/track pair that is outside the gate (never assigned).
constexpr int BEARING_BINS = 360;
constexpr double FORBIDDEN_COST = 1e9;

// Which 1 deg slice around the robot a base_link point falls in (0 = straight behind, going counter-clockwise).
int FindBearingBin(const Point2 & point)
{
  const double bearing = std::atan2(point.y, point.x);
  const int bin = static_cast<int>(std::floor((bearing + M_PI) / (2.0 * M_PI) * BEARING_BINS));
  return std::clamp(bin, 0, BEARING_BINS - 1);
}

// Bayes update of a track's existence probability when a cluster was matched to it ("seen").
// Being seen is likely for a real object (detectionProbability) and unlikely for noise (falseAlarmProbability),
// so the probability goes up.
double UpdateExistenceSeen(double existence, double detectionProbability, double falseAlarmProbability)
{
  return existence * detectionProbability /
         (existence * detectionProbability + (1.0 - existence) * falseAlarmProbability);
}

// Bayes update when nothing was matched to the track ("missed"). A miss is more likely for noise than for a
// real object the lidar should have seen, so the probability goes down. When the lidar could not have seen it
// anyway (detectionProbability ~0, e.g. hidden), the probability barely changes.
double UpdateExistenceMissed(double existence, double detectionProbability, double falseAlarmProbability)
{
  const double numerator = existence * (1.0 - detectionProbability);
  const double denominator = numerator + (1.0 - existence) * (1.0 - falseAlarmProbability);
  return denominator > 0.0 ? numerator / denominator : existence;
}

// The distance from the robot of the nearest of these base_link points.
double FindNearestRange(const std::vector<Point2> & basePoints)
{
  double nearestRange = std::numeric_limits<double>::infinity();
  for (const auto & point : basePoints) {
    nearestRange = std::min(nearestRange, CalculateLength(point));
  }
  return nearestRange;
}

}  // namespace

// Chance the lidar returns a cluster for a visible object at this distance: 0 outside the lidar's range,
// detectionProbabilityNear up to detectionNearRange, falling in a straight line to detectionProbabilityFar at
// detectionFarRange, and constant after that.
double ClusterTracker::CalculateDetectionProbability(double range) const
{
  if (!(range >= params.rangeMin && range <= params.rangeMax)) {
    return 0.0;
  }
  const double rangeSpan = params.detectionFarRange - params.detectionNearRange;
  const double fraction = rangeSpan > 0.0 ?
    std::clamp((range - params.detectionNearRange) / rangeSpan, 0.0, 1.0) :
    (range > params.detectionNearRange ? 1.0 : 0.0);
  return params.detectionProbabilityNear +
         fraction * (params.detectionProbabilityFar - params.detectionProbabilityNear);
}

// How uncertain a cluster's centroid is as a measurement of the object's position (variance, m^2):
// the lidar's own range noise (a + b * distance) plus a share of the cluster's size, because the lidar sees
// only the near side of an object and the centroid shifts as the view changes.
double ClusterTracker::CalculateMeasurementVariance(const Cluster & cluster) const
{
  const double sigma = params.lidarRangeStd[0] + params.lidarRangeStd[1] * CalculateLength(cluster.closest) +
    params.shapeSigmaScale * cluster.radius;
  return sigma * sigma;
}

// Chance a cluster is just noise: falseAlarmProbability, scaled up for clusters with fewer than
// falseAlarmPoints points (small clusters are more often noise), kept between 1e-6 and 0.5.
double ClusterTracker::CalculateFalseAlarmProbability(const Cluster & cluster) const
{
  const double pointCount = std::max<double>(1.0, static_cast<double>(cluster.points.size()));
  const double scale = std::max(1.0, params.falseAlarmPoints / pointCount);
  return std::clamp(params.falseAlarmProbability * scale, 1e-6, 0.5);
}

// Where a track's last seen points should be now, in base_link. While a track is not seen its points are moved
// (in odom) by the distance the filter's position moved since the last sighting, then converted to base_link.
// They are only moved when the published velocity says it moves: a standing object stays where it was seen.
std::vector<Point2> ClusterTracker::FindTrackPointsNow(const Track & track, const Pose2 & robot) const
{
  Point2 coastShift{0.0, 0.0};
  if (!track.isChunk && (track.motion.x != 0.0 || track.motion.y != 0.0)) {
    const Point2 positionNow = track.filter.GetPosition();
    coastShift = {positionNow.x - track.positionAtSighting.x, positionNow.y - track.positionAtSighting.y};
  }
  std::vector<Point2> basePointsNow;
  basePointsNow.reserve(track.points.size());
  for (const auto & point : track.points) {
    basePointsNow.push_back(robot.ConvertToBase({point.x + coastShift.x, point.y + coastShift.y}));
  }
  return basePointsNow;
}

// Gate between a track and a measurement: the squared Mahalanobis distance when the measurement may belong to
// the track, nothing when it may not. Never further than maximumGate; beyond minimumGate also the Mahalanobis
// test (gateChi2) must pass. logDeterminant, when given, receives ln det of the covariance (see the filter).
std::optional<double> ClusterTracker::CalculateGatedDistance(const Track & track, const Point2 & measurement,
  double variance, double * logDeterminant) const
{
  const double euclideanDistance = CalculateDistance(measurement, track.filter.GetPosition());
  if (euclideanDistance > params.maximumGate) {
    return std::nullopt;
  }
  const double distanceSquared =
    track.filter.CalculateMahalanobisSquared(measurement, variance * track.noiseScale, logDeterminant);
  if (distanceSquared > params.gateChi2 && euclideanDistance > params.minimumGate) {
    return std::nullopt;
  }
  return distanceSquared;
}

// Joins clusters that are pieces of one already tracked object (e.g. a person's two legs, which the
// clustering can see as two clusters). Without this the second piece would start a ghost track.
void ClusterTracker::MergeClustersByTrack(std::vector<Cluster> & clusters, const Pose2 & robot) const
{
  if (params.mergeRadius <= 0.0) {
    return;
  }
  // Each cluster -> the confirmed track it fits best (inside the gate).
  // clustersByTrack[trackIndex] = the clusters whose best fit is that track.
  std::map<std::size_t, std::vector<std::size_t>> clustersByTrack;
  for (std::size_t clusterIndex = 0; clusterIndex < clusters.size(); ++clusterIndex) {
    const Cluster & cluster = clusters[clusterIndex];
    if (cluster.isChunk) {
      continue;
    }
    const Point2 measurement = robot.ConvertToOdom(cluster.centroid);
    const double variance = CalculateMeasurementVariance(cluster);
    double bestDistanceSquared = std::numeric_limits<double>::infinity();
    std::size_t bestTrackIndex = tracks.size();
    for (std::size_t trackIndex = 0; trackIndex < tracks.size(); ++trackIndex) {
      const Track & track = tracks[trackIndex];
      if (!track.confirmed || track.isChunk) {
        continue;
      }
      const std::optional<double> distanceSquared = CalculateGatedDistance(track, measurement, variance);
      if (distanceSquared && *distanceSquared < bestDistanceSquared) {
        bestDistanceSquared = *distanceSquared;
        bestTrackIndex = trackIndex;
      }
    }
    if (bestTrackIndex < tracks.size()) {
      clustersByTrack[bestTrackIndex].push_back(clusterIndex);
    }
  }

  // For every track with 2 or more clusters: merge them into one, unless the result is too big to be one
  // object (larger than mergeRadius). The merged cluster replaces the first piece, the others are dropped.
  std::vector<bool> dropCluster(clusters.size(), false);
  for (const auto & trackEntry : clustersByTrack) {
    const std::vector<std::size_t> & members = trackEntry.second;
    if (members.size() < 2) {
      continue;
    }
    Cluster mergedCluster;
    for (auto clusterIndex : members) {
      const Cluster & cluster = clusters[clusterIndex];
      mergedCluster.points.insert(mergedCluster.points.end(), cluster.points.begin(), cluster.points.end());
    }
    FinalizeCluster(mergedCluster);
    if (mergedCluster.radius > params.mergeRadius) {
      continue;
    }
    clusters[members.front()] = std::move(mergedCluster);
    for (std::size_t memberIndex = 1; memberIndex < members.size(); ++memberIndex) {
      dropCluster[members[memberIndex]] = true;
    }
  }
  // Remove the dropped clusters by moving the kept ones to the front of the list (keeps their order).
  std::size_t keptCount = 0;
  for (std::size_t clusterIndex = 0; clusterIndex < clusters.size(); ++clusterIndex) {
    if (!dropCluster[clusterIndex]) {
      if (keptCount != clusterIndex) {
        clusters[keptCount] = std::move(clusters[clusterIndex]);
      }
      ++keptCount;
    }
  }
  clusters.resize(keptCount);
}

// True when at least half of the track's points (where they should be now, base_link) lie behind another
// cluster seen this cloud, i.e. something nearer (by at least occlusionMargin) is in the same direction. A
// hidden object cannot be seen, so missing it must not count as much against its existence.
bool ClusterTracker::IsOccluded(const std::vector<Point2> & basePointsNow,
  const std::vector<double> & nearestRangePerBearing) const
{
  if (basePointsNow.empty()) {
    return false;
  }
  int coveredCount = 0;
  for (const auto & basePoint : basePointsNow) {
    const double nearestRange = nearestRangePerBearing[static_cast<std::size_t>(FindBearingBin(basePoint))];
    if (nearestRange < CalculateLength(basePoint) - params.occlusionMargin) {
      ++coveredCount;
    }
  }
  return 2 * coveredCount >= static_cast<int>(basePointsNow.size());
}

void ClusterTracker::Update(std::vector<Cluster> & clusters, const Pose2 & robot, bool hasOdometry,
  double cloudSeconds)
{
  // Time went backwards (a looped rosbag): the tracks belong to another timeline, start over.
  if (cloudSeconds < lastSeconds) {
    tracks.clear();
  }
  lastSeconds = cloudSeconds;

  // --- predict every track to cloudSeconds
  // Each track's filter is moved forward to the time of this cloud, so tracks and clusters are compared at the
  // same moment.
  for (auto & track : tracks) {
    track.filter.Predict(cloudSeconds - track.lastPredicted);
    track.lastPredicted = cloudSeconds;
  }

  // --- join pieces of one object
  MergeClustersByTrack(clusters, robot);

  // Each cluster's measurement: its centroid in odom, and how uncertain that is.
  std::vector<Point2> measurements;
  std::vector<double> variances;
  measurements.reserve(clusters.size());
  for (const auto & cluster : clusters) {
    measurements.push_back(robot.ConvertToOdom(cluster.centroid));
    variances.push_back(CalculateMeasurementVariance(cluster));
  }

  // --- associate
  // trackOfCluster[cluster] = index of the track it belongs to (-1 = none yet);
  // trackSeen[track] = a cluster was matched to it this cloud.
  std::vector<int> trackOfCluster(clusters.size(), -1);
  std::vector<bool> trackSeen(tracks.size(), false);
  // wall pieces first: the track of the same odom cell
  std::map<std::int64_t, std::size_t> chunkTracks;
  for (std::size_t trackIndex = 0; trackIndex < tracks.size(); ++trackIndex) {
    if (tracks[trackIndex].isChunk) {
      chunkTracks[tracks[trackIndex].chunkKey] = trackIndex;
    }
  }
  for (std::size_t clusterIndex = 0; clusterIndex < clusters.size(); ++clusterIndex) {
    if (!clusters[clusterIndex].isChunk) {
      continue;
    }
    const auto chunkTrack = chunkTracks.find(clusters[clusterIndex].chunkKey);
    if (chunkTrack != chunkTracks.end() && !trackSeen[chunkTrack->second]) {
      trackOfCluster[clusterIndex] = static_cast<int>(chunkTrack->second);
      trackSeen[chunkTrack->second] = true;
    }
  }
  // everything else, wall pieces or not (a wall piece whose cell changed, a cluster that grew into a wall or
  // shrank out of one): minimum total cost inside the gates
  std::vector<std::size_t> freeClusters;
  std::vector<std::size_t> freeTracks;
  for (std::size_t clusterIndex = 0; clusterIndex < clusters.size(); ++clusterIndex) {
    if (trackOfCluster[clusterIndex] < 0) {
      freeClusters.push_back(clusterIndex);
    }
  }
  for (std::size_t trackIndex = 0; trackIndex < tracks.size(); ++trackIndex) {
    if (!trackSeen[trackIndex]) {
      freeTracks.push_back(trackIndex);
    }
  }
  if (!freeClusters.empty() && !freeTracks.empty()) {
    // Cost matrix: rows = free clusters, columns = free tracks. Pairs outside the gate stay FORBIDDEN_COST.
    // Allowed pairs cost the Mahalanobis distance squared (capped at the gate) + ln det of its covariance.
    std::vector<std::vector<double>> cost(freeClusters.size(), std::vector<double>(freeTracks.size(), FORBIDDEN_COST));
    for (std::size_t freeClusterIndex = 0; freeClusterIndex < freeClusters.size(); ++freeClusterIndex) {
      const std::size_t clusterIndex = freeClusters[freeClusterIndex];
      for (std::size_t freeTrackIndex = 0; freeTrackIndex < freeTracks.size(); ++freeTrackIndex) {
        double logDeterminant = 0.0;
        const std::optional<double> distanceSquared = CalculateGatedDistance(tracks[freeTracks[freeTrackIndex]],
          measurements[clusterIndex], variances[clusterIndex], &logDeterminant);
        if (distanceSquared) {
          cost[freeClusterIndex][freeTrackIndex] = std::min(*distanceSquared, params.gateChi2) + logDeterminant;
        }
      }
    }
    // logDeterminant may be negative: shift all allowed costs to be positive.
    double lowestCost = 0.0;
    for (const auto & costRow : cost) {
      for (double entryCost : costRow) {
        if (entryCost < FORBIDDEN_COST) {
          lowestCost = std::min(lowestCost, entryCost);
        }
      }
    }
    for (auto & costRow : cost) {
      for (double & entryCost : costRow) {
        if (entryCost < FORBIDDEN_COST) {
          entryCost -= lowestCost;
        }
      }
    }
    // Best overall pairing (Hungarian method), then write it back into trackOfCluster / trackSeen.
    const std::vector<int> assignedTracks = SolveAssignment(cost, FORBIDDEN_COST);
    for (std::size_t freeClusterIndex = 0; freeClusterIndex < freeClusters.size(); ++freeClusterIndex) {
      if (assignedTracks[freeClusterIndex] < 0) {
        continue;
      }
      const std::size_t trackIndex = freeTracks[static_cast<std::size_t>(assignedTracks[freeClusterIndex])];
      trackOfCluster[freeClusters[freeClusterIndex]] = static_cast<int>(trackIndex);
      trackSeen[trackIndex] = true;
    }
  }

  // --- missed tracks: existence down, by how likely the lidar was to see them
  // nearestRangePerBearing[bin]: distance of the nearest cluster point in each 1 deg direction this cloud,
  // used to check whether a missed track was hidden behind something.
  std::vector<double> nearestRangePerBearing(BEARING_BINS, std::numeric_limits<double>::infinity());
  for (const auto & cluster : clusters) {
    for (const auto & point : cluster.points) {
      double & binRange = nearestRangePerBearing[static_cast<std::size_t>(FindBearingBin(point))];
      binRange = std::min(binRange, CalculateLength(point));
    }
  }
  for (std::size_t trackIndex = 0; trackIndex < tracks.size(); ++trackIndex) {
    if (trackSeen[trackIndex]) {
      continue;
    }
    // Where the missed track's points should be now (base_link), and how likely the lidar was to see them.
    Track & track = tracks[trackIndex];
    const std::vector<Point2> basePointsNow = FindTrackPointsNow(track, robot);
    double detectionProbability = CalculateDetectionProbability(FindNearestRange(basePointsNow));
    // Hidden behind a nearer cluster: much less likely to be seen, so the miss counts much less.
    track.occluded = IsOccluded(basePointsNow, nearestRangePerBearing);
    if (track.occluded) {
      detectionProbability *= params.occludedDetectionFactor;
    }
    // A miss only counts once the track was missed in every cloud of a window of missWindow clouds.
    if (++track.missStreak % std::max(1, params.missWindow) == 0) {
      track.existence =
        UpdateExistenceMissed(track.existence, detectionProbability, params.falseAlarmProbability);
    }
  }

  // --- seen tracks and new tracks
  for (std::size_t clusterIndex = 0; clusterIndex < clusters.size(); ++clusterIndex) {
    Cluster & cluster = clusters[clusterIndex];
    const double falseAlarmProbability = CalculateFalseAlarmProbability(cluster);
    Track * track = nullptr;
    if (trackOfCluster[clusterIndex] >= 0) {
      // Matched: update the track's filter with the cluster's centroid.
      track = &tracks[static_cast<std::size_t>(trackOfCluster[clusterIndex])];
      const double variance = variances[clusterIndex] * track->noiseScale;
      const double normalizedInnovation =
        track->filter.CalculateMahalanobisSquared(measurements[clusterIndex], variance);
      track->filter.Update(measurements[clusterIndex], variance);
      // Adaptive noise: if this track's centroid jumps more than the model expects (normalized innovation / 2
      // above 1), slowly raise its measurement noise, so the jumps are not taken for real motion.
      if (track->sightings >= 2) {   // the first innovation includes the unknown velocity
        const double adaptRate = params.noiseAdaptRate;
        track->noiseScale = std::clamp(
          track->noiseScale * (1.0 - adaptRate + adaptRate * std::min(normalizedInnovation, 50.0) / 2.0), 1.0,
          params.maxNoiseScale);
      }
      // Seen: existence up.
      const double detectionProbability =
        std::max(1e-3, CalculateDetectionProbability(CalculateLength(cluster.closest)));
      track->existence = UpdateExistenceSeen(track->existence, detectionProbability, falseAlarmProbability);
      ++track->sightings;
      track->missStreak = 0;
    } else {
      // Not matched: start a new track at the cluster's centroid with an unknown velocity.
      Track newTrack;
      newTrack.id = nextTrackId++;
      newTrack.isChunk = cluster.isChunk;
      newTrack.chunkKey = cluster.chunkKey;
      newTrack.filter.Initialize(measurements[clusterIndex], variances[clusterIndex], params.unknownVelocitySigma,
        params.imm);
      newTrack.existence = params.birthExistence;
      newTrack.sightings = 1;
      newTrack.lastPredicted = cloudSeconds;
      tracks.push_back(std::move(newTrack));
      trackSeen.push_back(true);
      track = &tracks.back();
    }
    // Common to both: remember this sighting (time, points in odom, filter position and sigma right now).
    track->existence = std::min(track->existence, params.maxExistence);
    track->isChunk = cluster.isChunk;
    track->chunkKey = cluster.chunkKey;
    track->lastSeen = cloudSeconds;
    track->occluded = false;
    track->points.clear();
    for (const auto & point : cluster.points) {
      track->points.push_back(robot.ConvertToOdom(point));
    }
    track->positionAtSighting = track->filter.GetPosition();
    track->sigmaAtSighting = track->isChunk ? 0.0 : track->filter.GetPositionSigma();

    // Decide the velocity to publish for this cluster.
    cluster.trackId = track->id;
    if (cluster.isChunk) {
      // Wall pieces never move.
      track->motion = {0.0, 0.0};
      track->motionMode = ImmFilter::STATIC_MODEL;
      cluster.velocitySigma = params.staticVelocitySigma;
    } else {
      track->motionMode = track->filter.FindLikeliestMode();
      if (!hasOdometry || track->sightings < std::max(2, params.velocitySightings)) {
        // Too new, or no odometry (then the robot's own motion is mixed in): velocity unknown.
        track->motion = {0.0, 0.0};
        cluster.velocitySigma = params.unknownVelocitySigma;
      } else {
        // Report 0 when the static model is likely or the speed is too small to trust (below minimumSpeed or
        // 2 sigma); otherwise the moving velocity.
        const MovingVelocity moving = track->filter.CalculateMovingVelocity();
        const double velocitySigma = std::max(params.minimumVelocitySigma, moving.sigma);
        const Point2 movingVelocity = moving.velocity;
        const bool standing =
          track->filter.GetModeProbability(ImmFilter::STATIC_MODEL) > params.staticPublishProbability ||
          CalculateLength(movingVelocity) < std::max(params.minimumSpeed, 2.0 * velocitySigma);
        track->motion = standing ? Point2{0.0, 0.0} : movingVelocity;
        cluster.velocitySigma = velocitySigma;
      }
    }
    // The velocity is kept in odom on the track and given to the cluster in base_link axes.
    cluster.velocity = robot.RotateToBase(track->motion);
    track->publishedSigma = cluster.velocitySigma;
  }

  // --- confirm with hysteresis, drop the dead
  // Confirmed from confirmExistence up, unconfirmed again only below keepExistence (a lower value), so a track
  // near the threshold does not blink on and off.
  for (auto & track : tracks) {
    if (track.existence >= params.confirmExistence) {
      track.confirmed = true;
    }
    if (track.existence < params.keepExistence) {
      track.confirmed = false;
    }
  }
  // Delete tracks that are probably not real, not seen for too long, or too uncertain to be useful.
  tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [&](const Track & track) {
      const double timeout = track.occluded ? params.occludedTimeout : params.trackTimeout;
      return track.existence < params.deleteExistence || cloudSeconds - track.lastSeen > timeout ||
             track.filter.GetPositionSigma() > params.maxPositionSigma;
    }), tracks.end());

  // Copy the debug values (existence, motion model) onto the clusters for the RViz markers.
  for (auto & cluster : clusters) {
    for (const auto & track : tracks) {
      if (track.id == cluster.trackId) {
        cluster.existence = track.existence;
        cluster.motionMode = track.motionMode;
        break;
      }
    }
  }
}

std::vector<Cluster> ClusterTracker::FindPublishable(const std::vector<Cluster> & clusters, const Pose2 & robot,
  double cloudSeconds) const
{
  std::vector<Cluster> publishableClusters;
  publishableClusters.reserve(clusters.size());
  for (const auto & track : tracks) {
    // Unconfirmed tracks (maybe noise) are never published.
    if (!track.confirmed) {
      continue;
    }
    // Seen in this cloud: publish its cluster as it is.
    if (cloudSeconds - track.lastSeen <= 1e-9) {
      for (const auto & cluster : clusters) {
        if (cluster.trackId == track.id) {
          publishableClusters.push_back(cluster);
          break;
        }
      }
      continue;
    }
    // Missed in this cloud: last points, moved by the prediction. The filter was predicted to the time of the
    // last Update(); a later cloudSeconds (publishing between clouds) is not predicted further.
    // This "coasting" cluster keeps the obstacle on the output while the lidar briefly misses it.
    Cluster coastingCluster;
    coastingCluster.isChunk = track.isChunk;
    coastingCluster.chunkKey = track.chunkKey;
    coastingCluster.isCoasting = true;
    coastingCluster.trackId = track.id;
    coastingCluster.points = FindTrackPointsNow(track, robot);
    FinalizeCluster(coastingCluster);
    coastingCluster.velocity = robot.RotateToBase(track.motion);
    coastingCluster.velocitySigma = track.publishedSigma;
    // Extra position uncertainty: how much the filter's uncertainty grew since it was last seen.
    coastingCluster.extraSigma =
      track.isChunk ? 0.0 : std::max(0.0, track.filter.GetPositionSigma() - track.sigmaAtSighting);
    coastingCluster.existence = track.existence;
    coastingCluster.motionMode = track.motionMode;
    publishableClusters.push_back(std::move(coastingCluster));
  }
  return publishableClusters;
}

}  // namespace lidar_obstacles
