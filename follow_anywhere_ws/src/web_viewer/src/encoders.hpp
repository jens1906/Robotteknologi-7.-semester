// Converts camera and lidar messages into the compact payloads sent to the browser.
#pragma once

#include <string>
#include <vector>

#include <geometry_msgs/msg/transform.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace web_viewer
{

struct ImageOptions
{
  int max_width = 960;      // downscale wider images to this
  int jpeg_quality = 70;
  double depth_max_m = 6.0;  // depth images are colour mapped over [0, depth_max_m]
};

// Returns JPEG bytes, or an empty string for unsupported encodings.
std::string encode_image(const sensor_msgs::msg::Image & img, const ImageOptions & opt);

// Points as little-endian float32 x,y,z,intensity (intensity NaN when absent),
// thinned to at most max_points by a fixed stride.
std::vector<float> cloud_points(const sensor_msgs::msg::PointCloud2 & pc, size_t max_points);
std::vector<float> scan_points(const sensor_msgs::msg::LaserScan & scan);

// Moves x,y,z,intensity points in place by a rigid transform (rotation, then translation).
void transform_points(std::vector<float> & pts, const geometry_msgs::msg::Transform & t);

}  // namespace web_viewer
