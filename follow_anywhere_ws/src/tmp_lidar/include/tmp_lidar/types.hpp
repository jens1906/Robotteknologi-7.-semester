// Basic 2D types and geometry for tmp_lidar. No ROS.
// Frames: base_link (x forward, y left) unless a name says odom. Units: m, s, rad.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace tmp_lidar
{

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

struct Point2
{
  double x = 0.0;
  double y = 0.0;
};

struct Point3
{
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

inline Point2 nanPoint() {return {kNaN, kNaN};}
inline bool isFinite(const Point2 & p) {return std::isfinite(p.x) && std::isfinite(p.y);}
inline double norm(const Point2 & p) {return std::hypot(p.x, p.y);}
inline double distance(const Point2 & a, const Point2 & b) {return std::hypot(a.x - b.x, a.y - b.y);}

// Robot pose in odom: base_link -> odom.
struct Pose2
{
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;

  // base_link point -> odom point
  Point2 toOdom(const Point2 & p) const
  {
    const double c = std::cos(yaw), s = std::sin(yaw);
    return {x + c * p.x - s * p.y, y + s * p.x + c * p.y};
  }
  // odom point -> base_link point
  Point2 toBase(const Point2 & p) const
  {
    const double c = std::cos(yaw), s = std::sin(yaw);
    const double dx = p.x - x, dy = p.y - y;
    return {c * dx + s * dy, -s * dx + c * dy};
  }
  // base_link vector -> odom vector (rotation only)
  Point2 rotateToOdom(const Point2 & v) const
  {
    const double c = std::cos(yaw), s = std::sin(yaw);
    return {c * v.x - s * v.y, s * v.x + c * v.y};
  }
  Point2 rotateToBase(const Point2 & v) const
  {
    const double c = std::cos(yaw), s = std::sin(yaw);
    return {c * v.x + s * v.y, -s * v.x + c * v.y};
  }
};

// Point in the base_link of pose `from` -> the same floor point in the base_link of pose `to`.
inline Point2 shiftPoint(const Point2 & p, const Pose2 & from, const Pose2 & to)
{
  if (!isFinite(p)) {return p;}
  return to.toBase(from.toOdom(p));
}
inline Point2 shiftVector(const Point2 & v, const Pose2 & from, const Pose2 & to)
{
  return to.rotateToBase(from.rotateToOdom(v));
}

// Point of the segment a-b nearest p.
inline Point2 nearestOnSegment(const Point2 & a, const Point2 & b, const Point2 & p)
{
  const double dx = b.x - a.x, dy = b.y - a.y;
  const double length2 = dx * dx + dy * dy;
  double share = 0.0;
  if (length2 > 1e-12) {
    share = std::clamp(((p.x - a.x) * dx + (p.y - a.y) * dy) / length2, 0.0, 1.0);
  }
  return {a.x + share * dx, a.y + share * dy};
}

// One obstacle made of lidar points, in base_link of the cloud's time.
struct Cluster
{
  std::vector<Point2> points;
  Point2 centroid;
  Point2 closest;        // lidar point nearest the base_link origin
  double radius = 0.0;   // largest distance of a point from the centroid
  bool isChunk = false;  // a piece of a cluster that was too large (a wall): static by assumption

  // filled by ClusterTracker
  int trackId = -1;
  Point2 velocity;                 // over the ground, base_link axes; already 0 when standing still
  double velocitySigma = kNaN;     // m/s, 1 sigma
  bool isCoasting = false;         // not seen in this cloud: last points, moved by the velocity
  double extraSigma = 0.0;         // m, added to the position sigma (grows while coasting)
};

// Fills centroid, closest and radius from points.
inline void finalizeCluster(Cluster & c)
{
  c.centroid = {0.0, 0.0};
  double best = std::numeric_limits<double>::infinity();
  for (const auto & p : c.points) {
    c.centroid.x += p.x;
    c.centroid.y += p.y;
    const double r = norm(p);
    if (r < best) {
      best = r;
      c.closest = p;
    }
  }
  if (!c.points.empty()) {
    c.centroid.x /= static_cast<double>(c.points.size());
    c.centroid.y /= static_cast<double>(c.points.size());
  }
  c.radius = 0.0;
  for (const auto & p : c.points) {c.radius = std::max(c.radius, distance(p, c.centroid));}
}

// Distance from p to the nearest point of the cluster.
inline double distanceToCluster(const Cluster & c, const Point2 & p)
{
  double best = std::numeric_limits<double>::infinity();
  for (const auto & q : c.points) {best = std::min(best, distance(p, q));}
  return best;
}

// Cluster point nearest the segment a-b (the line of sight point, measured instead of modelled).
inline Point2 clusterPointNearestSegment(const Cluster & c, const Point2 & a, const Point2 & b)
{
  double best = std::numeric_limits<double>::infinity();
  Point2 out = nanPoint();
  for (const auto & q : c.points) {
    const double d = distance(q, nearestOnSegment(a, b, q));
    if (d < best) {
      best = d;
      out = q;
    }
  }
  return out;
}

}  // namespace tmp_lidar
