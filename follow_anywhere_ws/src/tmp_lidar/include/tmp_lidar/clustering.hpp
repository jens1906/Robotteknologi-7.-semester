// 2D obstacle points -> clusters. No ROS.
//
// Points go into a grid of cellSize; occupied cells touching (8 neighbours) form one cluster. Clusters
// with fewer than minClusterPoints points are dropped (single stray returns). A cluster larger than
// maxClusterRadius (a wall, a row of boxes) is cut into pieces on a grid of chunkSize, each reported as
// its own obstacle with isChunk set: navigation sees points and circles, and one circle around a whole
// wall would block the room.
// The grid is laid in the frame the points are given in. Give odom points and the robot pose: cells and wall
// pieces then stay in place on the floor while the robot moves (otherwise their borders slide over the points
// and clusters split and merge from cloud to cloud). The clusters come out in base_link of `robot`.
#pragma once

#include <vector>

#include "tmp_lidar/types.hpp"

namespace tmp_lidar
{

struct ClusterParams
{
  double cellSize = 0.1;
  int minClusterPoints = 3;
  double maxClusterRadius = 0.6;
  double chunkSize = 0.5;
};

std::vector<Cluster> clusterPoints(const std::vector<Point2> & points, const ClusterParams & params,
  const Pose2 & robot = Pose2{});

}  // namespace tmp_lidar
