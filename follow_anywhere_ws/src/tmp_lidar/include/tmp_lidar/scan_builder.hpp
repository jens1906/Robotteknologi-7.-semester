// 2D obstacle points -> 360 deg laser scan ranges around base_link (as 2dSim's /perception/scan). No ROS.
// Ray i points at -pi + i * 2 pi / rays; each ray keeps the nearest point in its bin, +inf when empty.
#pragma once

#include <vector>

#include "tmp_lidar/types.hpp"

namespace tmp_lidar
{

std::vector<float> buildScan(const std::vector<Point2> & points, int rays, double rangeMin, double rangeMax);

}  // namespace tmp_lidar
