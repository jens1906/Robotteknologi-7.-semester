#include "tmp_lidar/clustering.hpp"

#include <cmath>
#include <cstdint>
#include <unordered_map>

namespace tmp_lidar
{

namespace
{

using CellKey = std::int64_t;

CellKey makeKey(std::int64_t ix, std::int64_t iy)
{
  return (ix << 32) ^ (iy & 0xffffffffLL);
}

std::int64_t cellIndex(double value, double size)
{
  return static_cast<std::int64_t>(std::floor(value / size));
}

// Groups point indices by grid cell.
std::unordered_map<CellKey, std::vector<std::size_t>> buildGrid(
  const std::vector<Point2> & points, double size)
{
  std::unordered_map<CellKey, std::vector<std::size_t>> grid;
  grid.reserve(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    grid[makeKey(cellIndex(points[i].x, size), cellIndex(points[i].y, size))].push_back(i);
  }
  return grid;
}

// Large cluster -> pieces on a coarse grid (every piece keeps at least 2 points).
void splitIntoChunks(const Cluster & big, const ClusterParams & params, std::vector<Cluster> & out)
{
  const auto grid = buildGrid(big.points, params.chunkSize);
  for (const auto & cell : grid) {
    if (cell.second.size() < 2) {continue;}
    Cluster piece;
    piece.isChunk = true;
    for (auto i : cell.second) {piece.points.push_back(big.points[i]);}
    finalizeCluster(piece);
    out.push_back(std::move(piece));
  }
}

}  // namespace

std::vector<Cluster> clusterPoints(const std::vector<Point2> & points, const ClusterParams & params)
{
  std::vector<Cluster> clusters;
  if (points.empty() || params.cellSize <= 0.0) {return clusters;}

  const auto grid = buildGrid(points, params.cellSize);
  std::unordered_map<CellKey, bool> visited;
  visited.reserve(grid.size());

  for (const auto & seed : grid) {
    if (visited[seed.first]) {continue;}
    Cluster cluster;
    std::vector<CellKey> stack{seed.first};
    visited[seed.first] = true;
    while (!stack.empty()) {
      const CellKey key = stack.back();
      stack.pop_back();
      const auto & cellPoints = grid.at(key);
      for (auto i : cellPoints) {cluster.points.push_back(points[i]);}
      // Recover the cell indices from the first point of the cell.
      const auto & p = points[cellPoints.front()];
      const auto ix = cellIndex(p.x, params.cellSize);
      const auto iy = cellIndex(p.y, params.cellSize);
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          const CellKey neighbour = makeKey(ix + dx, iy + dy);
          if (grid.count(neighbour) == 0 || visited[neighbour]) {continue;}
          visited[neighbour] = true;
          stack.push_back(neighbour);
        }
      }
    }
    if (static_cast<int>(cluster.points.size()) < params.minClusterPoints) {continue;}
    finalizeCluster(cluster);
    if (params.maxClusterRadius > 0.0 && cluster.radius > params.maxClusterRadius) {
      splitIntoChunks(cluster, params, clusters);
    } else {
      clusters.push_back(std::move(cluster));
    }
  }
  return clusters;
}

}  // namespace tmp_lidar
