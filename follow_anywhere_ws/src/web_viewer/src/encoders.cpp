#include "encoders.hpp"

#include <cmath>
#include <cstring>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace web_viewer
{

std::string encode_image(const sensor_msgs::msg::Image & img, const ImageOptions & opt)
{
  if (img.width == 0 || img.height == 0 || img.data.empty()) {return {};}
  const auto & e = img.encoding;
  int h = static_cast<int>(img.height), w = static_cast<int>(img.width);
  auto * data = const_cast<uint8_t *>(img.data.data());
  size_t step = img.step;

  auto wrap = [&](int type) {
      size_t need = step * img.height;
      if (img.data.size() < need) {return cv::Mat();}
      return cv::Mat(h, w, type, data, step);
    };

  cv::Mat bgr;
  if (e == "bgr8" || e == "8UC3") {
    bgr = wrap(CV_8UC3);
  } else if (e == "rgb8") {
    cv::Mat m = wrap(CV_8UC3);
    if (!m.empty()) {cv::cvtColor(m, bgr, cv::COLOR_RGB2BGR);}
  } else if (e == "bgra8") {
    cv::Mat m = wrap(CV_8UC4);
    if (!m.empty()) {cv::cvtColor(m, bgr, cv::COLOR_BGRA2BGR);}
  } else if (e == "rgba8") {
    cv::Mat m = wrap(CV_8UC4);
    if (!m.empty()) {cv::cvtColor(m, bgr, cv::COLOR_RGBA2BGR);}
  } else if (e == "mono8" || e == "8UC1") {
    bgr = wrap(CV_8UC1);
  } else if (e == "yuv422" || e == "uyvy" || e == "UYVY") {
    cv::Mat m = wrap(CV_8UC2);
    if (!m.empty()) {cv::cvtColor(m, bgr, cv::COLOR_YUV2BGR_UYVY);}
  } else if (e == "yuv422_yuy2" || e == "yuyv" || e == "YUYV") {
    cv::Mat m = wrap(CV_8UC2);
    if (!m.empty()) {cv::cvtColor(m, bgr, cv::COLOR_YUV2BGR_YUYV);}
  } else if (e == "16UC1" || e == "mono16" || e == "32FC1") {
    // Depth: millimetres for 16UC1, metres for 32FC1. Invalid (0/NaN) stays black.
    bool is_float = e == "32FC1";
    cv::Mat m = wrap(is_float ? CV_32FC1 : CV_16UC1);
    if (m.empty()) {return {};}
    if (img.is_bigendian) {return {};}
    double scale = e == "mono16" ? 255.0 / 65535.0 :
      255.0 / (opt.depth_max_m * (is_float ? 1.0 : 1000.0));
    cv::Mat m8, valid;
    m.convertTo(m8, CV_8UC1, scale);
    if (e == "mono16") {
      bgr = m8;
    } else {
      if (is_float) {
        valid = (m == m) & (m > 0);  // NaN != NaN
      } else {
        valid = m > 0;
      }
      cv::applyColorMap(m8, bgr, cv::COLORMAP_TURBO);
      bgr.setTo(cv::Scalar::all(0), ~valid);
    }
  } else {
    return {};
  }
  if (bgr.empty()) {return {};}

  if (opt.max_width > 0 && bgr.cols > opt.max_width) {
    cv::Mat small;
    double f = static_cast<double>(opt.max_width) / bgr.cols;
    cv::resize(bgr, small, cv::Size(), f, f, cv::INTER_AREA);
    bgr = small;
  }
  std::vector<uchar> buf;
  cv::imencode(".jpg", bgr, buf, {cv::IMWRITE_JPEG_QUALITY, opt.jpeg_quality});
  return std::string(buf.begin(), buf.end());
}

std::vector<float> cloud_points(const sensor_msgs::msg::PointCloud2 & pc, size_t max_points)
{
  using sensor_msgs::msg::PointField;
  int ox = -1, oy = -1, oz = -1, oi = -1;
  uint8_t ti = 0;
  for (const auto & f : pc.fields) {
    if (f.name == "x" && f.datatype == PointField::FLOAT32) {ox = f.offset;}
    if (f.name == "y" && f.datatype == PointField::FLOAT32) {oy = f.offset;}
    if (f.name == "z" && f.datatype == PointField::FLOAT32) {oz = f.offset;}
    if (f.name == "intensity" || f.name == "i") {
      oi = f.offset;
      ti = f.datatype;
    }
  }
  std::vector<float> out;
  if (ox < 0 || oy < 0 || oz < 0 || pc.is_bigendian) {return out;}

  size_t n = static_cast<size_t>(pc.width) * pc.height;
  if (pc.point_step == 0 || pc.data.size() < n * pc.point_step) {
    n = pc.point_step ? pc.data.size() / pc.point_step : 0;
  }
  size_t stride = max_points > 0 && n > max_points ? (n + max_points - 1) / max_points : 1;
  out.reserve((n / stride + 1) * 4);
  const float nan = std::numeric_limits<float>::quiet_NaN();

  for (size_t i = 0; i < n; i += stride) {
    // Rows may be padded: use row_step when the cloud is organised.
    size_t row = pc.width ? i / pc.width : 0, col = pc.width ? i % pc.width : i;
    const uint8_t * p = pc.data.data() + row * pc.row_step + col * pc.point_step;
    if (p + pc.point_step > pc.data.data() + pc.data.size()) {break;}
    float x, y, z, in = nan;
    std::memcpy(&x, p + ox, 4);
    std::memcpy(&y, p + oy, 4);
    std::memcpy(&z, p + oz, 4);
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {continue;}
    if (oi >= 0) {
      switch (ti) {
        case PointField::FLOAT32: std::memcpy(&in, p + oi, 4); break;
        case PointField::FLOAT64: {double d; std::memcpy(&d, p + oi, 8); in = static_cast<float>(d); break;}
        case PointField::UINT8: in = p[oi]; break;
        case PointField::UINT16: {uint16_t v; std::memcpy(&v, p + oi, 2); in = v; break;}
        default: break;
      }
    }
    out.insert(out.end(), {x, y, z, in});
  }
  return out;
}

std::vector<float> scan_points(const sensor_msgs::msg::LaserScan & scan)
{
  std::vector<float> out;
  out.reserve(scan.ranges.size() * 4);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (size_t i = 0; i < scan.ranges.size(); ++i) {
    float r = scan.ranges[i];
    if (!std::isfinite(r) || r < scan.range_min || r > scan.range_max) {continue;}
    float a = scan.angle_min + static_cast<float>(i) * scan.angle_increment;
    float in = i < scan.intensities.size() ? scan.intensities[i] : nan;
    out.insert(out.end(), {r * std::cos(a), r * std::sin(a), 0.f, in});
  }
  return out;
}

}  // namespace web_viewer
