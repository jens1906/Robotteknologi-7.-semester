// Short-memory grid of lidar hits in odom. No ROS.
//
// The Go2's L1 gives only ~1400 points per cloud in a non-repeating pattern, so one cloud sees an object
// only partly and a second cloud sees other parts: clusters made from single clouds come and go. Here each
// cell (cellSize, keyed in odom so cell borders stay on the floor while the robot moves) remembers the clouds
// that hit it during the last memorySeconds. A cell is occupied when at least minCellHits different clouds
// hit it: single returns (floor speckles, noise) never are. The points of occupied cells, from the
// remembered clouds, are the input of the clustering.
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "tmp_lidar/types.hpp"

namespace tmp_lidar
{

struct HitGridParams
{
  double cellSize = 0.1;
  double memorySeconds = 0.4;
  int minCellHits = 2;
  int maxPointsPerCell = 8;
};

class HitGrid
{
public:
  explicit HitGrid(HitGridParams params = {})
  : params_(params) {}

  // Points of one cloud (odom) at time t.
  void add(const std::vector<Point2> & odomPoints, double t);

  // Points (odom) of the cells occupied at time t.
  std::vector<Point2> occupiedPoints(double t, int minHits) const;
  std::vector<Point2> occupiedPoints(double t) const {return occupiedPoints(t, params_.minCellHits);}

  std::size_t occupiedCells(double t) const;
  void clear() {cells_.clear();}

private:
  struct Cell
  {
    std::vector<double> hitTimes;                 // one per cloud that hit the cell
    std::vector<std::pair<double, Point2>> points;  // newest last
  };

  bool occupied(const Cell & cell, double t, int minHits) const;

  HitGridParams params_;
  std::unordered_map<std::int64_t, Cell> cells_;
  double lastTime_ = -1.0;
};

}  // namespace tmp_lidar
