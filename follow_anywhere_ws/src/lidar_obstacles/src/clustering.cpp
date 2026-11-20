// Obstacle points -> clusters. No ROS. See clustering.hpp.
#include "lidar_obstacles/clustering.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <unordered_map>

namespace lidar_obstacles
{

// ===== Hit grid ====================================================================================================
// Short-memory hit grid. See the hit grid section of clustering.hpp.

// Helpers only used in this file.
namespace
{

// The cell a point falls in, packed into one 64-bit key (x index in the upper 32 bits, y in the lower 32).
std::int64_t MakeCellKey(const Point2 & point, double cellSize)
{
  const auto cellX = static_cast<std::int64_t>(std::floor(point.x / cellSize));
  const auto cellY = static_cast<std::int64_t>(std::floor(point.y / cellSize));
  return (cellX << 32) ^ (cellY & 0xffffffffLL);
}

}  // namespace

void HitGrid::Add(const std::vector<Point2> & odomPoints, double cloudSeconds)
{
  // Time went backwards (a rosbag that loops): the old memory belongs to another timeline, drop it.
  if (cloudSeconds < lastSeconds) {
    cells.clear();
  }
  lastSeconds = cloudSeconds;

  // Forget what is older than the memory.
  // Old hit times and old points are removed from every cell; a cell with no hits left is deleted.
  const double oldestSeconds = cloudSeconds - params.memorySeconds;
  for (auto cellEntry = cells.begin(); cellEntry != cells.end(); ) {
    Cell & cell = cellEntry->second;
    cell.hitSeconds.erase(std::remove_if(cell.hitSeconds.begin(), cell.hitSeconds.end(),
      [&](double hitSecond) {return hitSecond < oldestSeconds;}), cell.hitSeconds.end());
    cell.points.erase(std::remove_if(cell.points.begin(), cell.points.end(),
      [&](const StampedPoint & stampedPoint) {return stampedPoint.seconds < oldestSeconds;}), cell.points.end());
    cellEntry = cell.hitSeconds.empty() ? cells.erase(cellEntry) : std::next(cellEntry);
  }

  // Add this cloud's points. A cell hit by several points of the same cloud still counts as one hit
  // (hitSeconds gets this cloud's time only once), so "hits" really means "different clouds".
  for (const auto & point : odomPoints) {
    Cell & cell = cells[MakeCellKey(point, params.cellSize)];
    if (cell.hitSeconds.empty() || cell.hitSeconds.back() != cloudSeconds) {
      cell.hitSeconds.push_back(cloudSeconds);
    }
    cell.points.push_back({point, cloudSeconds});
    // Keep at most maxPointsPerCell points, dropping the oldest.
    if (static_cast<int>(cell.points.size()) > params.maxPointsPerCell) {
      cell.points.erase(cell.points.begin());
    }
  }
}

// A cell is occupied when at least minHits different clouds hit it within the memory window.
bool HitGrid::IsOccupied(const Cell & cell, double nowSeconds, int minHits) const
{
  const double oldestSeconds = nowSeconds - params.memorySeconds;
  int hitCount = 0;
  for (double hitSecond : cell.hitSeconds) {
    if (hitSecond >= oldestSeconds) {
      ++hitCount;
    }
  }
  return hitCount >= minHits;
}

// All remembered points (still inside the memory window) of the occupied cells.
std::vector<Point2> HitGrid::FindOccupiedPoints(double nowSeconds, int minHits) const
{
  std::vector<Point2> occupiedPoints;
  const double oldestSeconds = nowSeconds - params.memorySeconds;
  for (const auto & cellEntry : cells) {
    const Cell & cell = cellEntry.second;
    if (!IsOccupied(cell, nowSeconds, minHits)) {
      continue;
    }
    for (const auto & stampedPoint : cell.points) {
      if (stampedPoint.seconds >= oldestSeconds) {
        occupiedPoints.push_back(stampedPoint.point);
      }
    }
  }
  return occupiedPoints;
}

// ===== Clustering ==================================================================================================
// Point clustering. See clustering.hpp.

// Helpers only used in this file.
namespace
{

// A grid cell is identified by one 64-bit number: the x index in the upper 32 bits, the y index in the lower 32.
using CellKey = std::int64_t;

// Packs the cell's x and y index into one key (used as the key of a hash map).
CellKey MakeCellKey(std::int64_t cellX, std::int64_t cellY)
{
  return (cellX << 32) ^ (cellY & 0xffffffffLL);
}

// Which cell a coordinate falls in. floor() so negative coordinates go to the correct cell (-0.05 -> cell -1).
std::int64_t CalculateCellIndex(double value, double cellSize)
{
  return static_cast<std::int64_t>(std::floor(value / cellSize));
}

// The key of the cell a point falls in.
CellKey FindCellKey(const Point2 & point, double cellSize)
{
  return MakeCellKey(CalculateCellIndex(point.x, cellSize), CalculateCellIndex(point.y, cellSize));
}

// Union-find (disjoint sets): keeps track of which points belong together.
// Every point starts as its own group. Join() merges two groups; FindRoot() returns one representative point
// of a group, so two points are in the same cluster exactly when they have the same root.
struct UnionFind
{
  std::vector<std::size_t> parent;   // parent[i] == i: i is a root

  // Every element is its own parent at the start (0, 1, 2, ...).
  explicit UnionFind(std::size_t elementCount)
  : parent(elementCount)
  {
    std::iota(parent.begin(), parent.end(), std::size_t{0});
  }

  // Follows the parents up to the root. On the way it points each element at its grandparent
  // ("path halving"), which keeps the chains short so later lookups are fast.
  std::size_t FindRoot(std::size_t element)
  {
    while (parent[element] != element) {
      parent[element] = parent[parent[element]];
      element = parent[element];
    }
    return element;
  }

  // Merges the two groups by hanging one root under the other.
  void Join(std::size_t firstElement, std::size_t secondElement)
  {
    parent[FindRoot(firstElement)] = FindRoot(secondElement);
  }
};

// Points (frame of the input) -> base_link cluster with centroid, closest point and radius.
// `members` are the indices of this cluster's points in `points`.
Cluster MakeCluster(const std::vector<Point2> & points, const std::vector<std::size_t> & members,
  const Pose2 & robot)
{
  Cluster cluster;
  cluster.points.reserve(members.size());
  for (auto memberIndex : members) {
    cluster.points.push_back(robot.ConvertToBase(points[memberIndex]));
  }
  FinalizeCluster(cluster);
  return cluster;
}

// Large cluster -> pieces on a coarse grid of the input frame (every piece keeps at least 2 points).
// Each chunkSize x chunkSize cell of the wall becomes its own small obstacle. Because the grid is in odom,
// a piece keeps the same cell (chunkKey) while the robot moves, so the tracker can follow it by that key.
void SplitIntoChunks(const std::vector<Point2> & points, const std::vector<std::size_t> & members,
  const ClusterParams & params, const Pose2 & robot, std::vector<Cluster> & clusters)
{
  // Sort the wall's points into chunk cells.
  std::unordered_map<CellKey, std::vector<std::size_t>> chunkGrid;
  for (auto memberIndex : members) {
    chunkGrid[FindCellKey(points[memberIndex], params.chunkSize)].push_back(memberIndex);
  }
  // Every cell with at least 2 points becomes a chunk cluster (a single point is likely noise).
  for (const auto & [chunkKey, chunkMembers] : chunkGrid) {
    if (chunkMembers.size() < 2) {
      continue;
    }
    Cluster chunk = MakeCluster(points, chunkMembers, robot);
    chunk.isChunk = true;
    chunk.chunkKey = chunkKey;
    clusters.push_back(std::move(chunk));
  }
}

// The minimum number of points a cluster at `range` needs to count as an obstacle: minClusterPoints next to the
// robot, falling in a straight line to minClusterPointsFar at farRange (far objects get fewer lidar hits).
int CalculateMinimumPoints(double range, const ClusterParams & params)
{
  if (params.farRange <= 0.0) {
    return params.minClusterPoints;
  }
  const double fraction = std::clamp(range / params.farRange, 0.0, 1.0);
  return static_cast<int>(std::lround(params.minClusterPoints +
         fraction * (params.minClusterPointsFar - params.minClusterPoints)));
}

}  // namespace

std::vector<Cluster> ClusterPoints(const std::vector<Point2> & points, const ClusterParams & params,
  const Pose2 & robot)
{
  std::vector<Cluster> clusters;
  if (points.empty() || params.cellSize <= 0.0) {
    return clusters;
  }

  // Hash grid: for every cell, the indices of the points in it. Finding a point's neighbours then means
  // looking only in the cells around it instead of comparing it with every other point.
  std::unordered_map<CellKey, std::vector<std::size_t>> grid;
  grid.reserve(points.size());
  for (std::size_t pointIndex = 0; pointIndex < points.size(); ++pointIndex) {
    grid[FindCellKey(points[pointIndex], params.cellSize)].push_back(pointIndex);
  }

  // Link every point to its neighbours within its link distance.
  const Point2 robotPosition{robot.x, robot.y};
  UnionFind linkedSets(points.size());
  for (std::size_t pointIndex = 0; pointIndex < points.size(); ++pointIndex) {
    const Point2 & point = points[pointIndex];
    // Link distance grows with the distance from the robot, because far points are further apart.
    const double linkDistance =
      std::max(params.linkDistance, CalculateDistance(point, robotPosition) * params.linkAngle);
    // How many cells in each direction can hold a point within that distance.
    const auto cellReach = static_cast<std::int64_t>(std::ceil(linkDistance / params.cellSize));
    const auto cellX = CalculateCellIndex(point.x, params.cellSize);
    const auto cellY = CalculateCellIndex(point.y, params.cellSize);
    for (auto stepX = -cellReach; stepX <= cellReach; ++stepX) {
      for (auto stepY = -cellReach; stepY <= cellReach; ++stepY) {
        const auto neighbourCell = grid.find(MakeCellKey(cellX + stepX, cellY + stepY));
        if (neighbourCell == grid.end()) {
          continue;
        }
        // Join with every close point. "neighbourIndex > pointIndex" checks each pair only once.
        for (auto neighbourIndex : neighbourCell->second) {
          if (neighbourIndex > pointIndex && CalculateDistance(point, points[neighbourIndex]) <= linkDistance) {
            linkedSets.Join(pointIndex, neighbourIndex);
          }
        }
      }
    }
  }

  // Collect the groups: all points with the same root are one cluster.
  std::unordered_map<std::size_t, std::vector<std::size_t>> groups;
  for (std::size_t pointIndex = 0; pointIndex < points.size(); ++pointIndex) {
    groups[linkedSets.FindRoot(pointIndex)].push_back(pointIndex);
  }

  // Turn each group into a cluster: drop groups with too few points, cut groups that are too large (walls)
  // into chunks, keep the rest as they are.
  for (const auto & group : groups) {
    const std::vector<std::size_t> & members = group.second;
    Cluster cluster = MakeCluster(points, members, robot);
    if (static_cast<int>(cluster.points.size()) < CalculateMinimumPoints(CalculateLength(cluster.closest), params)) {
      continue;
    }
    if (params.maxClusterRadius > 0.0 && cluster.radius > params.maxClusterRadius) {
      SplitIntoChunks(points, members, params, robot, clusters);
    } else {
      clusters.push_back(std::move(cluster));
    }
  }
  return clusters;
}

}  // namespace lidar_obstacles
