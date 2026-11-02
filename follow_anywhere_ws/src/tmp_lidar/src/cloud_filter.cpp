#include "tmp_lidar/cloud_filter.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace tmp_lidar
{

namespace
{

bool inBody(double x, double y, const FilterParams & p)
{
  return x >= p.bodyMin.x && x <= p.bodyMax.x && y >= p.bodyMin.y && y <= p.bodyMax.y;
}

bool isFinite3(const Point3 & q)
{
  return std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z);
}

// Solves the 3x3 system m * s = v (Cramer); false when singular.
bool solve3(const std::array<std::array<double, 3>, 3> & m, const std::array<double, 3> & v, std::array<double, 3> & s)
{
  auto det = [](const std::array<std::array<double, 3>, 3> & a) {
      return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
             a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
             a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    };
  const double d = det(m);
  if (std::abs(d) < 1e-9) {return false;}
  for (int k = 0; k < 3; ++k) {
    auto mk = m;
    for (int r = 0; r < 3; ++r) {mk[r][k] = v[r];}
    s[k] = det(mk) / d;
  }
  return true;
}

// Plane through three points: z = a x + b y + c.
bool planeThrough(const Point3 & p, const Point3 & q, const Point3 & r, GroundPlane & out)
{
  std::array<double, 3> s;
  if (!solve3({{{p.x, p.y, 1.0}, {q.x, q.y, 1.0}, {r.x, r.y, 1.0}}}, {p.z, q.z, r.z}, s)) {return false;}
  out.a = s[0];
  out.b = s[1];
  out.c = s[2];
  return true;
}

// Least squares plane z = a x + b y + c through points.
bool fitLeastSquares(const std::vector<const Point3 *> & points, GroundPlane & out)
{
  std::array<std::array<double, 3>, 3> m{};
  std::array<double, 3> v{};
  for (const auto * q : points) {
    const double row[3] = {q->x, q->y, 1.0};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {m[i][j] += row[i] * row[j];}
      v[i] += row[i] * q->z;
    }
  }
  std::array<double, 3> s;
  if (!solve3(m, v, s)) {return false;}
  out.a = s[0];
  out.b = s[1];
  out.c = s[2];
  return true;
}

bool acceptable(const GroundPlane & plane, const FilterParams & p)
{
  return std::abs(plane.c - p.floorZ) <= p.floorSearch && plane.tiltDeg() <= p.floorMaxTiltDeg;
}

}  // namespace

GroundPlane GroundEstimator::update(const std::vector<Point3> & cloud, const FilterParams & params)
{
  GroundPlane flat;
  flat.c = params.floorZ;
  const GroundPlane reference = haveLast_ ? last_ : flat;
  if (!params.fitFloorPlane) {return flat;}

  // Candidates: near the robot and close to the floor we expect (the last plane, so a tilted floor
  // far out stays a candidate).
  std::vector<const Point3 *> candidates;
  candidates.reserve(cloud.size());
  for (const auto & q : cloud) {
    if (!isFinite3(q) || inBody(q.x, q.y, params)) {continue;}
    const double r = std::hypot(q.x, q.y);
    if (r < params.rangeMin || r > params.floorFitRange) {continue;}
    if (std::abs(reference.heightAbove(q)) > params.floorSearch) {continue;}
    candidates.push_back(&q);
  }
  if (static_cast<int>(candidates.size()) < params.floorMinimumPoints) {return reference;}

  std::uniform_int_distribution<std::size_t> pick(0, candidates.size() - 1);
  GroundPlane best;
  int bestCount = -1;
  for (int it = 0; it < params.floorIterations; ++it) {
    GroundPlane trial;
    if (!planeThrough(*candidates[pick(rng_)], *candidates[pick(rng_)], *candidates[pick(rng_)], trial)) {continue;}
    if (!acceptable(trial, params)) {continue;}
    int count = 0;
    for (const auto * q : candidates) {
      if (std::abs(trial.heightAbove(*q)) <= params.floorInlierBand) {++count;}
    }
    if (count > bestCount) {
      bestCount = count;
      best = trial;
    }
  }
  if (bestCount < params.floorMinimumPoints) {return reference;}

  std::vector<const Point3 *> inliers;
  inliers.reserve(static_cast<std::size_t>(bestCount));
  for (const auto * q : candidates) {
    if (std::abs(best.heightAbove(*q)) <= params.floorInlierBand) {inliers.push_back(q);}
  }
  GroundPlane refined = best;
  if (!fitLeastSquares(inliers, refined) || !acceptable(refined, params)) {refined = best;}
  refined.inliers = static_cast<int>(inliers.size());
  refined.fitted = true;
  last_ = refined;
  haveLast_ = true;
  return refined;
}

FilterResult filterCloud(const std::vector<Point3> & cloud, const FilterParams & params, const GroundPlane & plane)
{
  FilterResult result;
  result.plane = plane;
  result.points.reserve(cloud.size());
  for (const auto & q : cloud) {
    if (!isFinite3(q)) {continue;}
    const double h = plane.heightAbove(q);
    if (h <= params.groundHeight || h > params.maxHeight) {continue;}
    if (inBody(q.x, q.y, params)) {continue;}
    const double r = std::hypot(q.x, q.y);
    if (r < params.rangeMin || r > params.rangeMax) {continue;}
    result.points.push_back({q.x, q.y});
  }
  return result;
}

}  // namespace tmp_lidar
