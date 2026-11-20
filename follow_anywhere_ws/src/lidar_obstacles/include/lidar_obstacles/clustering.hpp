// Obstacle points -> clusters. No ROS.
// Two parts, in pipeline order:
// - Hit grid: short memory of the last clouds, keeps only cells hit by several clouds.
// - Clustering: groups the remembered points into clusters (walls into pieces).
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "lidar_obstacles/types.hpp"

namespace lidar_obstacles
{

// ===== Hit grid ====================================================================================================
// Short-memory grid of lidar hits in odom. No ROS.
//
// The Go2's L1 gives only ~1400 points per cloud in a non-repeating pattern, so one cloud sees an object
// only partly and a second cloud sees other parts: clusters made from single clouds come and go. Here each
// cell (cellSize, keyed in odom so cell borders stay on the floor while the robot moves) remembers the clouds
// that hit it during the last memorySeconds. A cell is occupied when at least minCellHits different clouds
// hit it: single returns (floor speckles, noise) never are. The points of occupied cells, from the
// remembered clouds, are the input of the clustering.
//
// In the pipeline: FilterCloud (obstacle points, moved to odom) -> HitGrid::Add -> FindOccupiedPoints -> clustering.

// Settings of the grid (hitCellSize, memorySeconds, minCellHits in the yaml).
struct HitGridParams
{
  double cellSize = 0.1;
  double memorySeconds = 0.4;
  int minCellHits = 2;
  int maxPointsPerCell = 8;   // points kept per cell; older ones are dropped first (limits memory and work)
};

class HitGrid
{
public:
  explicit HitGrid(HitGridParams gridParams = {})
  : params(gridParams) {}

  // Points of one cloud (odom) at time cloudSeconds.
  void Add(const std::vector<Point2> & odomPoints, double cloudSeconds);

  // Points (odom) of the cells occupied at time nowSeconds.
  // minHits: how many different clouds must have hit a cell. The second version uses minCellHits from the
  // settings; the node passes 1 when it has no odometry (then the grid only holds the current cloud).
  std::vector<Point2> FindOccupiedPoints(double nowSeconds, int minHits) const;
  std::vector<Point2> FindOccupiedPoints(double nowSeconds) const
  {
    return FindOccupiedPoints(nowSeconds, params.minCellHits);
  }

  // Forget everything (no odometry, or odometry restarted).
  void Clear() {cells.clear();}

private:
  // What one grid cell remembers.
  struct Cell
  {
    std::vector<double> hitSeconds;      // one per cloud that hit the cell
    std::vector<StampedPoint> points;    // newest last
  };

  bool IsOccupied(const Cell & cell, double nowSeconds, int minHits) const;

  HitGridParams params;
  std::unordered_map<std::int64_t, Cell> cells;   // only cells that were hit exist (a sparse grid)
  double lastSeconds = -1.0;                      // time of the last Add, to notice time jumping back
};

// ===== Clustering ==================================================================================================
// 2D obstacle points -> clusters. No ROS.
//
// Two points belong to the same cluster when they are closer than the link distance, which grows with range:
// max(linkDistance, range * linkAngle). The L1's points get sparser with distance, and a fixed distance cuts far
// objects into several pieces that come and go from cloud to cloud. Points are found through a hash grid of
// cellSize and joined with union-find.
// Clusters with fewer than minPoints(range) points are dropped (stray returns); minPoints falls linearly from
// minClusterPoints at the robot to minClusterPointsFar at farRange, as far objects return fewer points.
// A cluster larger than maxClusterRadius (a wall, a row of boxes) is cut into pieces on a grid of chunkSize, each
// reported as its own obstacle with isChunk set and chunkKey its grid cell: navigation sees points and circles,
// and one circle around a whole wall would block the room.
// The grids are laid in the frame the points are given in. Give odom points and the robot pose: cells and wall
// pieces then stay in place on the floor while the robot moves (a wall piece keeps its chunkKey). The clusters
// come out in base_link of `robot`.
//
// In the pipeline: hit grid (occupied points) -> ClusterPoints -> tracker.

// Settings of the clustering (cellSize, linkDistance, ... in the yaml; farRange is set to obstacleMaxRange).
struct ClusterParams
{
  double cellSize = 0.1;           // hash grid of the neighbour search
  double linkDistance = 0.15;      // m, points closer than this are one obstacle ...
  double linkAngle = 0.04;         // rad, ... or closer than range * this (far away)
  int minClusterPoints = 3;
  int minClusterPointsFar = 2;
  double farRange = 6.0;
  double maxClusterRadius = 0.6;
  double chunkSize = 0.5;
};

// Groups the points into clusters as described above. `robot` is the robot pose in the frame of the points
// (Pose2{} = the points are already in base_link).
std::vector<Cluster> ClusterPoints(const std::vector<Point2> & points, const ClusterParams & params,
  const Pose2 & robot = Pose2{});

}  // namespace lidar_obstacles
