#include "tmp_lidar/hit_grid.hpp"

#include <algorithm>
#include <cmath>

namespace tmp_lidar
{

namespace
{

std::int64_t cellKey(const Point2 & p, double size)
{
  const auto ix = static_cast<std::int64_t>(std::floor(p.x / size));
  const auto iy = static_cast<std::int64_t>(std::floor(p.y / size));
  return (ix << 32) ^ (iy & 0xffffffffLL);
}

}  // namespace

void HitGrid::add(const std::vector<Point2> & odomPoints, double t)
{
  if (t < lastTime_) {cells_.clear();}   // time jumped back (bag loop)
  lastTime_ = t;

  // Forget what is older than the memory.
  const double oldest = t - params_.memorySeconds;
  for (auto it = cells_.begin(); it != cells_.end(); ) {
    auto & cell = it->second;
    cell.hitTimes.erase(std::remove_if(cell.hitTimes.begin(), cell.hitTimes.end(),
      [&](double h) {return h < oldest;}), cell.hitTimes.end());
    cell.points.erase(std::remove_if(cell.points.begin(), cell.points.end(),
      [&](const auto & p) {return p.first < oldest;}), cell.points.end());
    it = cell.hitTimes.empty() ? cells_.erase(it) : std::next(it);
  }

  for (const auto & p : odomPoints) {
    auto & cell = cells_[cellKey(p, params_.cellSize)];
    if (cell.hitTimes.empty() || cell.hitTimes.back() != t) {cell.hitTimes.push_back(t);}
    cell.points.emplace_back(t, p);
    if (static_cast<int>(cell.points.size()) > params_.maxPointsPerCell) {cell.points.erase(cell.points.begin());}
  }
}

bool HitGrid::occupied(const Cell & cell, double t, int minHits) const
{
  const double oldest = t - params_.memorySeconds;
  int hits = 0;
  for (double h : cell.hitTimes) {
    if (h >= oldest) {++hits;}
  }
  return hits >= minHits;
}

std::vector<StampedPoint> HitGrid::occupiedStampedPoints(double t, int minHits) const
{
  std::vector<StampedPoint> out;
  const double oldest = t - params_.memorySeconds;
  for (const auto & [key, cell] : cells_) {
    (void)key;
    if (!occupied(cell, t, minHits)) {continue;}
    for (const auto & [time, p] : cell.points) {
      if (time >= oldest) {out.push_back({p, time});}
    }
  }
  return out;
}

std::vector<Point2> HitGrid::occupiedPoints(double t, int minHits) const
{
  std::vector<Point2> out;
  for (const auto & s : occupiedStampedPoints(t, minHits)) {out.push_back(s.p);}
  return out;
}

std::size_t HitGrid::occupiedCells(double t) const
{
  std::size_t n = 0;
  for (const auto & [key, cell] : cells_) {
    (void)key;
    if (occupied(cell, t, params_.minCellHits)) {++n;}
  }
  return n;
}

}  // namespace tmp_lidar
