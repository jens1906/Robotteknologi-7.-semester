// Lidar points (base_link, 3D) -> obstacle points on the floor plane (base_link, 2D). No ROS.
//
// The floor is a plane fitted each cloud (GroundEstimator), not a fixed height: the Go2 pitches and rolls
// by up to ~6 deg while walking, which is +-0.25 m of floor height at 5 m in base_link (measured on the
// bag go2_all_topics_6_10_11_28: floor at z ~ -0.38 m, tilted 1-4 deg, residual std 2.3 cm, 99 % within
// 8.6 cm). A point is an obstacle when it is groundHeight .. maxHeight above that plane.
// Removed as well: NaN, the robot's own body box, points outside rangeMin..rangeMax (horizontal distance).
#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "tmp_lidar/types.hpp"

namespace tmp_lidar
{

struct FilterParams
{
  double floorZ = -0.38;            // m, floor height in base_link when no plane was fitted yet
  bool fitFloorPlane = true;        // false: flat floor at floorZ
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

// z = a x + b y + c in base_link.
struct GroundPlane
{
  double a = 0.0;
  double b = 0.0;
  double c = -0.38;
  int inliers = 0;
  bool fitted = false;   // false: flat fallback

  double heightAbove(const Point3 & p) const {return p.z - (a * p.x + b * p.y + c);}
  double tiltDeg() const {return std::atan(std::hypot(a, b)) * 180.0 / M_PI;}
};

// Fits the floor plane each cloud; keeps the last good plane when a fit fails.
class GroundEstimator
{
public:
  GroundPlane update(const std::vector<Point3> & cloud, const FilterParams & params);
  void reset() {haveLast_ = false;}

private:
  GroundPlane last_;
  bool haveLast_ = false;
  std::mt19937 rng_{12345};
};

struct FilterResult
{
  std::vector<Point2> points;
  GroundPlane plane;
};

FilterResult filterCloud(const std::vector<Point3> & cloud, const FilterParams & params, const GroundPlane & plane);

}  // namespace tmp_lidar
