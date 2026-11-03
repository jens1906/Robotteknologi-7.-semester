// Minimum cost one-to-one assignment (Hungarian method, O(n^3)). No ROS.
#pragma once

#include <vector>

namespace tmp_lidar
{

// cost[r][c]; entries >= forbidden are never assigned. Returns for each row the assigned column or -1.
std::vector<int> solveAssignment(const std::vector<std::vector<double>> & cost, double forbidden);

}  // namespace tmp_lidar
