// 2D obstacle points -> 360 deg laser scan ranges around base_link (as 2dSim's /perception/scan). No ROS.
// Ray i points at -pi + i * 2 pi / rays; each ray keeps the nearest point in its bin, +inf when empty.
#pragma once

#include <vector>

#include "tmp_lidar/types.hpp"

namespace tmp_lidar
{

std::vector<float> buildScan(const std::vector<Point2> & points, int rays, double rangeMin, double rangeMax);

// Same from obstacle clusters, each covering its whole bearing span: between neighbouring points (by bearing) of a
// cluster the range is interpolated. The sparse L1 hits a different scatter of rays on an object every cloud, so a
// scan of raw points switches most rays on and off from cloud to cloud (bag: 94 % of the rays within 4 m); the
// tracked clusters are steady and every ray across an object returns.
std::vector<float> buildScan(const std::vector<Cluster> & clusters, int rays, double rangeMin, double rangeMax);

}  // namespace tmp_lidar
