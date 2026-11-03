#include "tmp_lidar/cloud_gate.hpp"

#include <algorithm>
#include <vector>

namespace tmp_lidar
{

double CloudGate::typical() const
{
  if (counts_.empty()) {return 0.0;}
  std::vector<std::size_t> sorted(counts_.begin(), counts_.end());
  std::nth_element(sorted.begin(), sorted.begin() + static_cast<long>(sorted.size() / 2), sorted.end());
  return static_cast<double>(sorted[sorted.size() / 2]);
}

bool CloudGate::accept(std::size_t points)
{
  const bool degraded = points < params_.minPoints ||
    (!counts_.empty() && static_cast<double>(points) < params_.minFraction * typical());
  if (degraded && skipped_ < params_.maxSkipped) {
    ++skipped_;
    return false;
  }
  skipped_ = 0;
  if (!degraded) {
    counts_.push_back(points);
    while (counts_.size() > params_.history) {counts_.pop_front();}
  }
  return true;
}

}  // namespace tmp_lidar
