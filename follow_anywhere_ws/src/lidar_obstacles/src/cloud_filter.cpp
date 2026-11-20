// Cleaning of the incoming lidar cloud. No ROS. See cloud_filter.hpp.
#include "lidar_obstacles/cloud_filter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace lidar_obstacles
{

// ===== Cloud gate ==================================================================================================
// Empty-cloud gate. See the cloud gate section of cloud_filter.hpp.

// Median of the remembered point counts. The median (middle value) is used instead of the average so that a
// few odd clouds do not change what "typical" means. nth_element only sorts enough to find the middle value.
double CloudGate::CalculateTypicalCount() const
{
  if (acceptedCounts.empty()) {
    return 0.0;
  }
  std::vector<std::size_t> sortedCounts(acceptedCounts.begin(), acceptedCounts.end());
  const std::size_t middle = sortedCounts.size() / 2;
  std::nth_element(sortedCounts.begin(), sortedCounts.begin() + static_cast<long>(middle), sortedCounts.end());
  return static_cast<double>(sortedCounts[middle]);
}

bool CloudGate::Accept(std::size_t pointCount)
{
  // "Degraded" = too few points in absolute terms, or much fewer than usual.
  const bool degraded = pointCount < params.minPoints ||
    (!acceptedCounts.empty() && static_cast<double>(pointCount) < params.minFraction * CalculateTypicalCount());
  // Skip a degraded cloud, but never more than maxSkipped in a row.
  if (degraded && skippedInARow < params.maxSkipped) {
    ++skippedInARow;
    return false;
  }
  // Accepted. Only normal clouds go into the history, so a lidar that stays quiet does not lower "typical".
  skippedInARow = 0;
  if (!degraded) {
    acceptedCounts.push_back(pointCount);
    while (acceptedCounts.size() > params.history) {
      acceptedCounts.pop_front();
    }
  }
  return true;
}

// ===== Cloud filter ================================================================================================
// Floor plane fit (RANSAC + least squares) and the obstacle point filter. See cloud_filter.hpp.

// Helpers only used in this file (an unnamed namespace hides them from the other files).
namespace
{

using Matrix3 = std::array<std::array<double, 3>, 3>;
using Vector3 = std::array<double, 3>;

// True when (x, y) lies inside the rectangle the robot's own body covers in base_link.
// The lidar sees the robot's legs and back; those points must never become obstacles.
bool IsInBody(double x, double y, const FilterParams & params)
{
  return x >= params.bodyMin.x && x <= params.bodyMax.x && y >= params.bodyMin.y && y <= params.bodyMax.y;
}

// True when x, y and z are all real numbers. The Go2 sends NaN for beams that hit nothing.
bool IsFinitePoint3(const Point3 & point)
{
  return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

// Determinant of a 3x3 matrix (expansion along the first row).
double CalculateDeterminant(const Matrix3 & matrix)
{
  return matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
         matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
         matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
}

// Solves the 3x3 system matrix * solution = rightSide (Cramer); false when singular.
// Cramer's rule: each unknown = determinant of the matrix with that column replaced by rightSide, divided by
// the determinant of the matrix. A determinant near 0 means the points are on a line (no unique plane).
bool SolveLinearSystem3(const Matrix3 & matrix, const Vector3 & rightSide, Vector3 & solution)
{
  const double determinant = CalculateDeterminant(matrix);
  if (std::abs(determinant) < 1e-9) {
    return false;
  }
  for (int column = 0; column < 3; ++column) {
    Matrix3 replacedMatrix = matrix;
    for (int row = 0; row < 3; ++row) {
      replacedMatrix[row][column] = rightSide[row];
    }
    solution[column] = CalculateDeterminant(replacedMatrix) / determinant;
  }
  return true;
}

// Copies the solved [slopeX, slopeY, heightAtOrigin] into the plane.
void SetPlaneCoefficients(const Vector3 & coefficients, GroundPlane & plane)
{
  plane.slopeX = coefficients[0];
  plane.slopeY = coefficients[1];
  plane.heightAtOrigin = coefficients[2];
}

// Plane through three points: z = slopeX x + slopeY y + heightAtOrigin.
// Each point gives one equation (x * slopeX + y * slopeY + 1 * heightAtOrigin = z): three equations, three unknowns.
bool FitPlaneThroughThreePoints(const Point3 & firstPoint, const Point3 & secondPoint, const Point3 & thirdPoint,
  GroundPlane & plane)
{
  Vector3 coefficients;
  const Matrix3 matrix{{{firstPoint.x, firstPoint.y, 1.0}, {secondPoint.x, secondPoint.y, 1.0},
    {thirdPoint.x, thirdPoint.y, 1.0}}};
  if (!SolveLinearSystem3(matrix, {firstPoint.z, secondPoint.z, thirdPoint.z}, coefficients)) {
    return false;
  }
  SetPlaneCoefficients(coefficients, plane);
  return true;
}

// Least squares plane z = slopeX x + slopeY y + heightAtOrigin through points.
// With many points the equations cannot all hold exactly; least squares finds the plane with the smallest sum
// of squared height errors. It builds the "normal equations" (A^T A) c = A^T z, where every row of A is
// [x, y, 1], and solves that 3x3 system.
bool FitLeastSquaresPlane(const std::vector<const Point3 *> & points, GroundPlane & plane)
{
  Matrix3 normalMatrix{};
  Vector3 rightSide{};
  for (const auto * point : points) {
    const double row[3] = {point->x, point->y, 1.0};
    for (int rowIndex = 0; rowIndex < 3; ++rowIndex) {
      for (int columnIndex = 0; columnIndex < 3; ++columnIndex) {
        normalMatrix[rowIndex][columnIndex] += row[rowIndex] * row[columnIndex];
      }
      rightSide[rowIndex] += row[rowIndex] * point->z;
    }
  }
  Vector3 coefficients;
  if (!SolveLinearSystem3(normalMatrix, rightSide, coefficients)) {
    return false;
  }
  SetPlaneCoefficients(coefficients, plane);
  return true;
}

// A plane is believable as the floor when it is near the expected floor height (floorZ +- floorSearch)
// and not tilted more than floorMaxTiltDeg. This rejects planes fitted to a table top or a ramp-like wall.
bool IsPlaneAcceptable(const GroundPlane & plane, const FilterParams & params)
{
  return std::abs(plane.heightAtOrigin - params.floorZ) <= params.floorSearch &&
         plane.CalculateTiltDegrees() <= params.floorMaxTiltDeg;
}

}  // namespace

GroundPlane GroundEstimator::Update(const std::vector<Point3> & cloud, const FilterParams & params)
{
  // The reference is where we expect the floor: last cloud's plane, or a flat floor at floorZ at the start.
  // It is also what is returned when this cloud's fit fails.
  GroundPlane flatPlane;
  flatPlane.heightAtOrigin = params.floorZ;
  const GroundPlane referencePlane = hasLastPlane ? lastPlane : flatPlane;

  // Candidates: near the robot and close to the floor we expect (the last plane, so a tilted floor
  // far out stays a candidate).
  // Only pointers are stored, so the cloud is not copied.
  std::vector<const Point3 *> candidates;
  candidates.reserve(cloud.size());
  for (const auto & point : cloud) {
    if (!IsFinitePoint3(point) || IsInBody(point.x, point.y, params)) {
      continue;
    }
    const double horizontalDistance = std::hypot(point.x, point.y);
    if (horizontalDistance < params.rangeMin || horizontalDistance > params.floorFitRange) {
      continue;
    }
    if (std::abs(referencePlane.CalculateHeightAbove(point)) > params.floorSearch) {
      continue;
    }
    candidates.push_back(&point);
  }
  // Too few floor points to fit anything reliable: keep the previous floor.
  if (static_cast<int>(candidates.size()) < params.floorMinimumPoints) {
    return referencePlane;
  }

  // RANSAC: try floorIterations random planes through 3 candidates and keep the one most candidates agree with
  // (within floorInlierBand). Obstacles and noise disagree with the floor plane, so they cannot win.
  std::uniform_int_distribution<std::size_t> pickCandidate(0, candidates.size() - 1);
  GroundPlane bestPlane;
  int bestInlierCount = -1;
  for (int iteration = 0; iteration < params.floorIterations; ++iteration) {
    GroundPlane trialPlane;
    const Point3 & firstPoint = *candidates[pickCandidate(randomGenerator)];
    const Point3 & secondPoint = *candidates[pickCandidate(randomGenerator)];
    const Point3 & thirdPoint = *candidates[pickCandidate(randomGenerator)];
    if (!FitPlaneThroughThreePoints(firstPoint, secondPoint, thirdPoint, trialPlane)) {
      continue;
    }
    if (!IsPlaneAcceptable(trialPlane, params)) {
      continue;
    }
    int inlierCount = 0;
    for (const auto * candidate : candidates) {
      if (std::abs(trialPlane.CalculateHeightAbove(*candidate)) <= params.floorInlierBand) {
        ++inlierCount;
      }
    }
    if (inlierCount > bestInlierCount) {
      bestInlierCount = inlierCount;
      bestPlane = trialPlane;
    }
  }
  // Even the best plane has too few supporting points: keep the previous floor.
  if (bestInlierCount < params.floorMinimumPoints) {
    return referencePlane;
  }

  // Refine: least squares through all the points that agree with the best RANSAC plane. This averages out the
  // noise of the 3 random points. If the refined plane fails (singular or not acceptable), keep the RANSAC one.
  std::vector<const Point3 *> inliers;
  inliers.reserve(static_cast<std::size_t>(bestInlierCount));
  for (const auto * candidate : candidates) {
    if (std::abs(bestPlane.CalculateHeightAbove(*candidate)) <= params.floorInlierBand) {
      inliers.push_back(candidate);
    }
  }
  GroundPlane refinedPlane = bestPlane;
  if (!FitLeastSquaresPlane(inliers, refinedPlane) || !IsPlaneAcceptable(refinedPlane, params)) {
    refinedPlane = bestPlane;
  }
  // Store the result as the reference for the next cloud.
  refinedPlane.inliers = static_cast<int>(inliers.size());
  refinedPlane.fitted = true;
  lastPlane = refinedPlane;
  hasLastPlane = true;
  return refinedPlane;
}

std::vector<Point2> FilterCloud(const std::vector<Point3> & cloud, const FilterParams & params,
  const GroundPlane & plane)
{
  std::vector<Point2> obstaclePoints;
  obstaclePoints.reserve(cloud.size());
  for (const auto & point : cloud) {
    // Drop beams that returned nothing.
    if (!IsFinitePoint3(point)) {
      continue;
    }
    // Drop the floor (at most groundHeight above the plane) and anything above maxHeight (ceiling, lamps).
    const double horizontalDistance = std::hypot(point.x, point.y);
    const double floorHeight =
      horizontalDistance < params.nearBodyRange ? params.nearBodyGroundHeight : params.groundHeight;
    const double heightAboveFloor = plane.CalculateHeightAbove(point);
    if (heightAboveFloor <= floorHeight || heightAboveFloor > params.maxHeight) {
      continue;
    }
    // Drop the robot's own body.
    if (IsInBody(point.x, point.y, params)) {
      continue;
    }
    // Drop points too close (noise next to the sensor) or too far (sparse and not relevant).
    if (horizontalDistance < params.rangeMin || horizontalDistance > params.rangeMax) {
      continue;
    }
    // Keep it as a 2D point on the floor.
    obstaclePoints.push_back({point.x, point.y});
  }
  return obstaclePoints;
}

}  // namespace lidar_obstacles
