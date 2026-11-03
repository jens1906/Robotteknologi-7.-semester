#include "tmp_lidar/clustering.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
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

CellKey keyOf(const Point2 & p, double size)
{
  return makeKey(cellIndex(p.x, size), cellIndex(p.y, size));
}

struct UnionFind
{
  std::vector<std::size_t> parent;
  explicit UnionFind(std::size_t n)
  : parent(n) {std::iota(parent.begin(), parent.end(), std::size_t{0});}
  std::size_t find(std::size_t i)
  {
    while (parent[i] != i) {
      parent[i] = parent[parent[i]];
      i = parent[i];
    }
    return i;
  }
  void join(std::size_t a, std::size_t b) {parent[find(a)] = find(b);}
};

// Points (frame of the input) -> base_link cluster with centroid, measured centroid etc.
Cluster makeCluster(const std::vector<StampedPoint> & points, const std::vector<std::size_t> & members,
  const ClusterParams & params, const Pose2 & robot, double newest)
{
  Cluster c;
  c.points.reserve(members.size());
  Point2 fresh{0.0, 0.0};
  int freshCount = 0;
  for (auto i : members) {
    const Point2 p = robot.toBase(points[i].p);
    c.points.push_back(p);
    if (points[i].t >= newest - params.freshWindow - 1e-9) {
      fresh.x += p.x;
      fresh.y += p.y;
      ++freshCount;
    }
  }
  finalizeCluster(c);
  if (params.freshWindow <= 0.0) {   // off: every point counts as fresh
    c.freshPoints = static_cast<int>(c.points.size());
  } else if (freshCount >= params.minFreshPoints && freshCount > 0) {
    c.measuredCentroid = {fresh.x / freshCount, fresh.y / freshCount};
    c.freshPoints = freshCount;
  }
  return c;
}

// Large cluster -> pieces on a coarse grid of the input frame (every piece keeps at least 2 points).
void splitIntoChunks(const std::vector<StampedPoint> & points, const std::vector<std::size_t> & members,
  const ClusterParams & params, const Pose2 & robot, double newest, std::vector<Cluster> & out)
{
  std::unordered_map<CellKey, std::vector<std::size_t>> grid;
  for (auto i : members) {grid[keyOf(points[i].p, params.chunkSize)].push_back(i);}
  for (const auto & [key, cell] : grid) {
    if (cell.size() < 2) {continue;}
    Cluster piece = makeCluster(points, cell, params, robot, newest);
    piece.isChunk = true;
    piece.chunkKey = key;
    out.push_back(std::move(piece));
  }
}

int minimumPoints(double range, const ClusterParams & params)
{
  if (params.farRange <= 0.0) {return params.minClusterPoints;}
  const double share = std::clamp(range / params.farRange, 0.0, 1.0);
  return static_cast<int>(std::lround(params.minClusterPoints +
         share * (params.minClusterPointsFar - params.minClusterPoints)));
}

}  // namespace

std::vector<Cluster> clusterPoints(const std::vector<StampedPoint> & points, const ClusterParams & params,
  const Pose2 & robot)
{
  std::vector<Cluster> clusters;
  if (points.empty() || params.cellSize <= 0.0) {return clusters;}

  double newest = -std::numeric_limits<double>::infinity();
  for (const auto & s : points) {newest = std::max(newest, s.t);}

  std::unordered_map<CellKey, std::vector<std::size_t>> grid;
  grid.reserve(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {grid[keyOf(points[i].p, params.cellSize)].push_back(i);}

  // Link every point to its neighbours within its link distance.
  const Point2 origin{robot.x, robot.y};   // robot position in the frame of the points
  UnionFind sets(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    const Point2 & p = points[i].p;
    const double link = std::max(params.linkDistance, distance(p, origin) * params.linkAngle);
    const auto reach = static_cast<std::int64_t>(std::ceil(link / params.cellSize));
    const auto ix = cellIndex(p.x, params.cellSize), iy = cellIndex(p.y, params.cellSize);
    for (auto dx = -reach; dx <= reach; ++dx) {
      for (auto dy = -reach; dy <= reach; ++dy) {
        const auto it = grid.find(makeKey(ix + dx, iy + dy));
        if (it == grid.end()) {continue;}
        for (auto j : it->second) {
          if (j > i && distance(p, points[j].p) <= link) {sets.join(i, j);}
        }
      }
    }
  }

  std::unordered_map<std::size_t, std::vector<std::size_t>> groups;
  for (std::size_t i = 0; i < points.size(); ++i) {groups[sets.find(i)].push_back(i);}

  for (const auto & [root, members] : groups) {
    (void)root;
    Cluster cluster = makeCluster(points, members, params, robot, newest);
    if (static_cast<int>(cluster.points.size()) < minimumPoints(norm(cluster.closest), params)) {continue;}
    if (params.maxClusterRadius > 0.0 && cluster.radius > params.maxClusterRadius) {
      splitIntoChunks(points, members, params, robot, newest, clusters);
    } else {
      clusters.push_back(std::move(cluster));
    }
  }
  return clusters;
}

std::vector<Cluster> clusterPoints(const std::vector<Point2> & points, const ClusterParams & params,
  const Pose2 & robot)
{
  std::vector<StampedPoint> stamped;
  stamped.reserve(points.size());
  for (const auto & p : points) {stamped.push_back({p, 0.0});}
  return clusterPoints(stamped, params, robot);
}

}  // namespace tmp_lidar
