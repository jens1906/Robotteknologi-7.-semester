// Lidar points (base_link, 3D) -> obstacle points on the floor plane (base_link, 2D). No ROS.
//
// Removed: NaN, the floor (within groundHeight of floorZ), anything above maxHeight over the floor
// (ceilings, overhangs the robot fits under), the robot's own body box, and points outside
// rangeMin..rangeMax (horizontal distance from base_link).
// The floor height in base_link is floorZ (standing Go2: base ~0.32 m above the floor). With
// estimateFloor it is measured each frame: a low percentile of z near the robot, kept within
// floorSearch of floorZ (so a frame of only boxes cannot move the floor onto their tops).
#pragma once

#include <vector>

#include "tmp_lidar/types.hpp"

namespace tmp_lidar
{

struct FilterParams
{
  double floorZ = -0.32;
  bool estimateFloor = true;
  double floorSearch = 0.15;        // m, estimated floor stays within floorZ +- this
  double floorEstimateRange = 3.0;  // m, points nearer than this are used for the estimate
  double floorPercentile = 0.05;
  int floorMinimumPoints = 50;
  double groundHeight = 0.05;       // m above the floor that still counts as floor
  double maxHeight = 1.9;           // m above the floor; higher points are ignored
  Point2 bodyMin{-0.45, -0.2};      // base_link box of the robot's own body
  Point2 bodyMax{0.45, 0.2};
  double rangeMin = 0.35;
  double rangeMax = 6.0;
};

struct FilterResult
{
  std::vector<Point2> points;
  double floorZ = 0.0;   // the floor height used
};

FilterResult filterCloud(const std::vector<Point3> & cloud, const FilterParams & params);

}  // namespace tmp_lidar
