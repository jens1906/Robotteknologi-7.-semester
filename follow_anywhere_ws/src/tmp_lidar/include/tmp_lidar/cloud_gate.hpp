// Guard against empty or nearly empty clouds. No ROS.
//
// The L1 sometimes delivers a cloud with no or very few points (a dropped packet, a driver hiccup). Processed
// normally, such a cloud is a miss for every object at once and every obstacle blinks out. A cloud with fewer
// than minFraction of the typical count (median of the last `history` accepted clouds), or fewer than minPoints,
// is rejected: the caller skips it and repeats its last output. After maxSkipped rejected clouds in a row the
// next one is accepted anyway, so a lidar that really went quiet does not freeze the old obstacles.
#pragma once

#include <cstddef>
#include <deque>

namespace tmp_lidar
{

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
  explicit CloudGate(CloudGateParams params = {})
  : params_(params) {}

  // true: process the cloud of `points` points; false: skip it.
  bool accept(std::size_t points);

  double typical() const;   // median of the accepted counts, 0 when none
  int skippedInARow() const {return skipped_;}
  void reset()
  {
    counts_.clear();
    skipped_ = 0;
  }

private:
  CloudGateParams params_;
  std::deque<std::size_t> counts_;
  int skipped_ = 0;
};

}  // namespace tmp_lidar
