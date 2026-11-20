// Cleaning of the incoming lidar cloud. No ROS.
// Two parts, in pipeline order:
// - Cloud gate: skips nearly empty clouds.
// - Cloud filter: fits the floor plane and keeps only the obstacle points.
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <random>
#include <vector>

#include "lidar_obstacles/types.hpp"

namespace lidar_obstacles
{

// ===== Cloud gate ==================================================================================================
// Guard against empty or nearly empty clouds. No ROS.
//
// The L1 sometimes delivers a cloud with no or very few points (a dropped packet, a driver hiccup). Processed
// normally, such a cloud is a miss for every object at once and every obstacle blinks out. A cloud with fewer
// than minFraction of the typical count (median of the last `history` accepted clouds), or fewer than minPoints,
// is rejected: the caller skips it and repeats its last output. After maxSkipped rejected clouds in a row the
// next one is accepted anyway, so a lidar that really went quiet does not freeze the old obstacles.

// Settings of the gate (emptyCloudFraction, emptyCloudPoints, maxSkippedClouds in the yaml).
struct CloudGateParams
{
  double minFraction = 0.3;
  std::size_t minPoints = 50;
  int maxSkipped = 4;
  std::size_t history = 30;
};

class CloudGate
{
public:
  explicit CloudGate(CloudGateParams gateParams = {})
  : params(gateParams) {}

  // true: process the cloud of `pointCount` points; false: skip it.
  bool Accept(std::size_t pointCount);

  double CalculateTypicalCount() const;   // median of the accepted counts, 0 when none

  // Forget the history (called when odometry comes back after a gap and everything restarts).
  void Reset()
  {
    acceptedCounts.clear();
    skippedInARow = 0;
  }

private:
  CloudGateParams params;
  std::deque<std::size_t> acceptedCounts;   // point counts of the last `history` normal clouds
  int skippedInARow = 0;                    // clouds rejected since the last accepted one
};

// ===== Cloud filter ================================================================================================
// Lidar points (base_link, 3D) -> obstacle points on the floor plane (base_link, 2D). No ROS.
//
// The floor is a plane fitted each cloud (GroundEstimator), not a fixed height: the Go2 pitches and rolls
// by up to ~6 deg while walking, which is +-0.25 m of floor height at 5 m in base_link (measured on the
// bag go2_all_topics_6_10_11_28: floor at z ~ -0.38 m, tilted 1-4 deg, residual std 2.3 cm, 99 % within
// 8.6 cm). A point is an obstacle when it is groundHeight .. maxHeight above that plane.
// Removed as well: NaN, the robot's own body box, points outside rangeMin..rangeMax (horizontal distance).
//
// This is the first processing step of every cloud: GroundEstimator::Update finds the floor, then FilterCloud
// keeps only the points that stand up from that floor (legs, walls, boxes) and drops their height.

// Settings of the floor fit and the point filter. The defaults are overwritten by the node's parameters
// (lidar_obstacles_params.yaml).
struct FilterParams
{
  double floorZ = -0.38;            // m, floor height in base_link when no plane was fitted yet
  double floorSearch = 0.15;        // m, plane height at base_link must stay within floorZ +- this
  double floorFitRange = 4.0;       // m, points nearer than this are used for the fit
  double floorMaxTiltDeg = 10.0;    // steeper planes are rejected
  double floorInlierBand = 0.04;    // m, RANSAC inlier distance
  int floorIterations = 100;
  int floorMinimumPoints = 50;      // inliers needed to accept a plane
  double groundHeight = 0.10;       // m above the floor plane that still counts as floor
  double maxHeight = 1.9;           // m above the floor plane; higher points are ignored
  Point2 bodyMin{-0.45, -0.2};      // base_link box of the robot's own body
  Point2 bodyMax{0.45, 0.2};
  double rangeMin = 0.35;
  double rangeMax = 6.0;
};

// z = slopeX x + slopeY y + heightAtOrigin in base_link.
// slopeX / slopeY: how much the floor rises per metre forward / to the left (non-zero when the robot pitches
// or rolls). heightAtOrigin: the floor height right under base_link.
struct GroundPlane
{
  double slopeX = 0.0;
  double slopeY = 0.0;
  double heightAtOrigin = -0.38;
  int inliers = 0;
  bool fitted = false;   // false: flat fallback

  // How high a point is above this floor plane (negative = below the floor).
  double CalculateHeightAbove(const Point3 & point) const
  {
    return point.z - (slopeX * point.x + slopeY * point.y + heightAtOrigin);
  }

  // Tilt of the plane against base_link's horizontal, in degrees (shown in the RViz markers).
  double CalculateTiltDegrees() const {return std::atan(std::hypot(slopeX, slopeY)) * 180.0 / M_PI;}
};

// Fits the floor plane each cloud; keeps the last good plane when a fit fails.
// Uses RANSAC: pick 3 random floor candidates, fit a plane through them, count how many candidates lie on it,
// repeat floorIterations times and keep the plane with the most points. Then refine it with a least squares
// fit through all those points. The random generator has a fixed seed so runs are repeatable.
class GroundEstimator
{
public:
  GroundPlane Update(const std::vector<Point3> & cloud, const FilterParams & params);

private:
  GroundPlane lastPlane;
  bool hasLastPlane = false;
  std::mt19937 randomGenerator{12345};
};

// Keeps the points that are obstacles (between groundHeight and maxHeight above the plane, outside the robot's
// body box, within rangeMin..rangeMax) and returns them as 2D points in base_link.
std::vector<Point2> FilterCloud(const std::vector<Point3> & cloud, const FilterParams & params, const GroundPlane & plane);

}  // namespace lidar_obstacles
