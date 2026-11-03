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
// With stamped points, measuredCentroid is the centroid of the points of the last freshWindow seconds (the newest
// clouds), when there are at least minFreshPoints of them: the short memory of the hit grid leaves a trail behind
// a moving object, which would drag its full centroid back. freshWindow <= 0 (the default) turns this off: on the
// bag go2_all_topics_6_10_11_28 the partial views of single clouds jumped more than the trail lagged (more ID
// jumps and flicker), so the full centroid is measured.
#pragma once

#include <vector>

#include "tmp_lidar/types.hpp"

namespace tmp_lidar
{

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
  double freshWindow = 0.0;        // s, <= 0: off
  int minFreshPoints = 3;
};

std::vector<Cluster> clusterPoints(const std::vector<StampedPoint> & points, const ClusterParams & params,
  const Pose2 & robot = Pose2{});

// Points without time: all count as fresh.
std::vector<Cluster> clusterPoints(const std::vector<Point2> & points, const ClusterParams & params,
  const Pose2 & robot = Pose2{});

}  // namespace tmp_lidar
