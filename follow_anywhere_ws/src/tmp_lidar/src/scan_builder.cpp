#include "tmp_lidar/scan_builder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tmp_lidar
{

std::vector<float> buildScan(const std::vector<Point2> & points, int rays, double rangeMin, double rangeMax)
{
  std::vector<float> ranges(static_cast<std::size_t>(std::max(rays, 1)), std::numeric_limits<float>::infinity());
  const double step = 2.0 * M_PI / static_cast<double>(ranges.size());
  for (const auto & p : points) {
    const double r = norm(p);
    if (r < rangeMin || r > rangeMax) {continue;}
    // Nearest ray: rays sit at -pi + i * step.
    auto i = static_cast<long>(std::lround((std::atan2(p.y, p.x) + M_PI) / step));
    i %= static_cast<long>(ranges.size());
    auto & slot = ranges[static_cast<std::size_t>(i)];
    slot = std::min(slot, static_cast<float>(r));
  }
  return ranges;
}

std::vector<float> buildScan(const std::vector<Cluster> & clusters, int rays, double rangeMin, double rangeMax)
{
  std::vector<float> ranges(static_cast<std::size_t>(std::max(rays, 1)), std::numeric_limits<float>::infinity());
  const auto n = static_cast<long>(ranges.size());
  const double step = 2.0 * M_PI / static_cast<double>(n);
  auto put = [&](long i, double r) {
      if (r < rangeMin || r > rangeMax) {return;}
      auto & slot = ranges[static_cast<std::size_t>(((i % n) + n) % n)];
      slot = std::min(slot, static_cast<float>(r));
    };
  for (const auto & c : clusters) {
    if (c.points.empty()) {continue;}
    // Bearings unwrapped around the centroid's, so a cluster behind the robot (across +-pi) stays in one piece.
    const double a0 = std::atan2(c.centroid.y, c.centroid.x);
    std::vector<std::pair<double, double>> polar;   // ray index (continuous), range
    polar.reserve(c.points.size());
    for (const auto & p : c.points) {
      const double a = a0 + std::remainder(std::atan2(p.y, p.x) - a0, 2.0 * M_PI);
      polar.emplace_back((a + M_PI) / step, norm(p));
    }
    std::sort(polar.begin(), polar.end());
    for (std::size_t k = 0; k < polar.size(); ++k) {
      put(std::lround(polar[k].first), polar[k].second);
      if (k + 1 == polar.size()) {continue;}
      const auto & [fa, ra] = polar[k];
      const auto & [fb, rb] = polar[k + 1];
      for (auto i = static_cast<long>(std::ceil(fa)); i <= static_cast<long>(std::floor(fb)); ++i) {
        const double s = fb > fa ? (static_cast<double>(i) - fa) / (fb - fa) : 0.0;
        put(i, ra + s * (rb - ra));
      }
    }
  }
  return ranges;
}

}  // namespace tmp_lidar
