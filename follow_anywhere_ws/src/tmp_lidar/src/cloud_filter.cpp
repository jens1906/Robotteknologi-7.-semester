#include "tmp_lidar/cloud_filter.hpp"

#include <algorithm>
#include <cmath>

namespace tmp_lidar
{

namespace
{

bool inBody(double x, double y, const FilterParams & p)
{
  return x >= p.bodyMin.x && x <= p.bodyMax.x && y >= p.bodyMin.y && y <= p.bodyMax.y;
}

double estimateFloor(const std::vector<Point3> & cloud, const FilterParams & p)
{
  std::vector<double> heights;
  heights.reserve(cloud.size());
  for (const auto & q : cloud) {
    if (!std::isfinite(q.x) || !std::isfinite(q.y) || !std::isfinite(q.z)) {continue;}
    if (inBody(q.x, q.y, p) || std::hypot(q.x, q.y) > p.floorEstimateRange) {continue;}
    // Only points that could be floor: anything far from floorZ is an object or noise.
    if (std::abs(q.z - p.floorZ) > 2.0 * p.floorSearch) {continue;}
    heights.push_back(q.z);
  }
  if (static_cast<int>(heights.size()) < p.floorMinimumPoints) {return p.floorZ;}
  const auto index = static_cast<std::size_t>(p.floorPercentile * static_cast<double>(heights.size() - 1));
  std::nth_element(heights.begin(), heights.begin() + static_cast<std::ptrdiff_t>(index), heights.end());
  return std::clamp(heights[index], p.floorZ - p.floorSearch, p.floorZ + p.floorSearch);
}

}  // namespace

FilterResult filterCloud(const std::vector<Point3> & cloud, const FilterParams & params)
{
  FilterResult result;
  result.floorZ = params.estimateFloor ? estimateFloor(cloud, params) : params.floorZ;
  const double zLow = result.floorZ + params.groundHeight;
  const double zHigh = result.floorZ + params.maxHeight;
  result.points.reserve(cloud.size());
  for (const auto & q : cloud) {
    if (!std::isfinite(q.x) || !std::isfinite(q.y) || !std::isfinite(q.z)) {continue;}
    if (q.z <= zLow || q.z > zHigh) {continue;}
    if (inBody(q.x, q.y, params)) {continue;}
    const double r = std::hypot(q.x, q.y);
    if (r < params.rangeMin || r > params.rangeMax) {continue;}
    result.points.push_back({q.x, q.y});
  }
  return result;
}

}  // namespace tmp_lidar
