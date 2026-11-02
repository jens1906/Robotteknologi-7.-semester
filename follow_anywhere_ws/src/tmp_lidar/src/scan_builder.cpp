#include "tmp_lidar/scan_builder.hpp"

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

}  // namespace tmp_lidar
