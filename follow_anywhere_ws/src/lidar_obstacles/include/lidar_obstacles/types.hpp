// Basic 2D types and geometry for lidar_obstacles. No ROS.
// Frames: base_link (x forward, y left) unless a name says odom. Units: m, s, rad.
//
// This file is shared by every other file in the package. It holds:
// - the point types (2D, 2D with a time stamp, 3D),
// - small geometry helpers (length, distance, "not a number" points),
// - Pose2: the robot's position and heading in odom, with functions that move points between the robot's own
//   frame (base_link) and the fixed world frame (odom),
// - Cluster: one obstacle made of lidar points, and helpers that compute its centre, nearest point and size.
// Everything here is "inline" so it can live in a header and be used without a separate .cpp file.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace lidar_obstacles
{

// Numbers
// NaN marks "no value": a point or sigma that is unknown. Any calculation with NaN gives NaN, and
// std::isfinite(NaN) is false, which is how the code checks whether a value is present.
constexpr double NOT_A_NUMBER = std::numeric_limits<double>::quiet_NaN();

// A point on the floor plane (the height is dropped once a lidar point has been accepted as an obstacle).
struct Point2
{
  double x = 0.0;
  double y = 0.0;
};

// A point with the time of the cloud it came from.
// Used by the hit grid to forget points that are older than its memory.
struct StampedPoint
{
  Point2 point;
  double seconds = 0.0;
};

// A raw lidar point with its height, as it comes out of the PointCloud2 message.
struct Point3
{
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

// A point where both coordinates are NaN: "this point is not known" (for example no line of sight point).
inline Point2 MakeNotANumberPoint() {return {NOT_A_NUMBER, NOT_A_NUMBER};}

// True when both coordinates are real numbers (not NaN and not infinite).
inline bool IsFinite(const Point2 & point) {return std::isfinite(point.x) && std::isfinite(point.y);}

// Distance of a point from the origin. In base_link this is the distance from the robot;
// for a velocity vector it is the speed.
inline double CalculateLength(const Point2 & point) {return std::hypot(point.x, point.y);}

// Straight-line distance between two points on the floor.
inline double CalculateDistance(const Point2 & firstPoint, const Point2 & secondPoint)
{
  return std::hypot(firstPoint.x - secondPoint.x, firstPoint.y - secondPoint.y);
}

// Robot pose in odom: base_link -> odom.
// x, y is where the robot stands in the odom (world) frame, yaw is the direction it faces (rad, 0 = odom x axis).
// The functions below are 2D rotations + translations: they let the code keep things that do not move
// (walls, the hit grid, tracks) in odom, and convert them back to base_link for publishing.
struct Pose2
{
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;

  // base_link point -> odom point
  // Rotate the point by the robot's heading, then add the robot's position.
  Point2 ConvertToOdom(const Point2 & point) const
  {
    const double cosine = std::cos(yaw);
    const double sine = std::sin(yaw);
    return {x + cosine * point.x - sine * point.y, y + sine * point.x + cosine * point.y};
  }

  // odom point -> base_link point
  // The inverse of ConvertToOdom: subtract the robot's position, then rotate by minus the heading.
  Point2 ConvertToBase(const Point2 & point) const
  {
    const double cosine = std::cos(yaw);
    const double sine = std::sin(yaw);
    const double offsetX = point.x - x;
    const double offsetY = point.y - y;
    return {cosine * offsetX + sine * offsetY, -sine * offsetX + cosine * offsetY};
  }

  // base_link vector -> odom vector (rotation only)
  // Vectors (velocities) have a direction but no position, so they are only rotated, never shifted.
  Point2 RotateToOdom(const Point2 & vector) const
  {
    const double cosine = std::cos(yaw);
    const double sine = std::sin(yaw);
    return {cosine * vector.x - sine * vector.y, sine * vector.x + cosine * vector.y};
  }

  // odom vector -> base_link vector (rotation only), the inverse of RotateToOdom.
  Point2 RotateToBase(const Point2 & vector) const
  {
    const double cosine = std::cos(yaw);
    const double sine = std::sin(yaw);
    return {cosine * vector.x + sine * vector.y, -sine * vector.x + cosine * vector.y};
  }
};

// Point in the base_link of pose `poseThen` -> the same floor point in the base_link of pose `poseNow`.
// Used when a camera message was made at another moment than the lidar cloud: the robot has moved in between,
// so a point seen "1 m ahead" then may be "0.9 m ahead" now. Going through odom removes the robot's motion.
// NaN points (unknown) are passed through unchanged.
inline Point2 ShiftPointBetweenPoses(const Point2 & point, const Pose2 & poseThen, const Pose2 & poseNow)
{
  if (!IsFinite(point)) {
    return point;
  }
  return poseNow.ConvertToBase(poseThen.ConvertToOdom(point));
}

// Same as ShiftPointBetweenPoses for a velocity: only the robot's turning matters, not its movement.
inline Point2 RotateVectorBetweenPoses(const Point2 & vector, const Pose2 & poseThen, const Pose2 & poseNow)
{
  return poseNow.RotateToBase(poseThen.RotateToOdom(vector));
}

// Point of the segment segmentStart-segmentEnd nearest point.
// Projects the point onto the line through the segment, then clamps the result to the segment's ends
// (fraction 0 = segmentStart, 1 = segmentEnd). A segment of zero length returns segmentStart.
inline Point2 FindNearestPointOnSegment(const Point2 & segmentStart, const Point2 & segmentEnd, const Point2 & point)
{
  const double segmentX = segmentEnd.x - segmentStart.x;
  const double segmentY = segmentEnd.y - segmentStart.y;
  const double squaredLength = segmentX * segmentX + segmentY * segmentY;
  double fraction = 0.0;
  if (squaredLength > 1e-12) {
    fraction = std::clamp(
      ((point.x - segmentStart.x) * segmentX + (point.y - segmentStart.y) * segmentY) / squaredLength, 0.0, 1.0);
  }
  return {segmentStart.x + fraction * segmentX, segmentStart.y + fraction * segmentY};
}

// One obstacle made of lidar points, in base_link of the cloud's time.
// Made by the clustering (points, centroid, closest, radius, chunk fields), then filled in further by the
// tracker (track id, velocity and the other fields below "filled by ClusterTracker").
struct Cluster
{
  std::vector<Point2> points;
  Point2 centroid;
  Point2 closest;        // lidar point nearest the base_link origin
  double radius = 0.0;   // largest distance of a point from the centroid
  bool isChunk = false;  // a piece of a cluster that was too large (a wall): static by assumption
  std::int64_t chunkKey = 0;   // isChunk: the odom cell of the piece, the same from cloud to cloud

  // filled by ClusterTracker
  int trackId = -1;
  Point2 velocity;                 // over the ground, base_link axes; already 0 when standing still
  double velocitySigma = NOT_A_NUMBER;     // m/s, 1 sigma
  bool isCoasting = false;         // not seen in this cloud: last points, moved by the velocity
  double extraSigma = 0.0;         // m, added to the position sigma (grows while coasting)
  double existence = NOT_A_NUMBER;         // probability that the object is real (debug)
  int motionMode = -1;             // most likely motion model: 0 static, 1 constant velocity, 2 turning (debug)
};

// Fills centroid, closest and radius from points.
// Called every time a cluster's points change (new cluster, merged cluster, coasting cluster).
inline void FinalizeCluster(Cluster & cluster)
{
  // One pass: sum the points for the centroid and remember the point nearest the robot.
  cluster.centroid = {0.0, 0.0};
  double closestDistance = std::numeric_limits<double>::infinity();
  for (const auto & point : cluster.points) {
    cluster.centroid.x += point.x;
    cluster.centroid.y += point.y;
    const double pointDistance = CalculateLength(point);
    if (pointDistance < closestDistance) {
      closestDistance = pointDistance;
      cluster.closest = point;
    }
  }
  // Average of the points = centroid (skipped for an empty cluster to avoid dividing by zero).
  if (!cluster.points.empty()) {
    cluster.centroid.x /= static_cast<double>(cluster.points.size());
    cluster.centroid.y /= static_cast<double>(cluster.points.size());
  }
  // Second pass: the radius is the distance from the centroid to the farthest point.
  cluster.radius = 0.0;
  for (const auto & point : cluster.points) {
    cluster.radius = std::max(cluster.radius, CalculateDistance(point, cluster.centroid));
  }
}

// Distance from point to the nearest point of the cluster.
// The fusion uses this to decide whether a camera person and a lidar cluster are the same object.
inline double CalculateDistanceToCluster(const Cluster & cluster, const Point2 & point)
{
  double nearestDistance = std::numeric_limits<double>::infinity();
  for (const auto & clusterPoint : cluster.points) {
    nearestDistance = std::min(nearestDistance, CalculateDistance(point, clusterPoint));
  }
  return nearestDistance;
}

// Cluster point nearest the segment segmentStart-segmentEnd (the line of sight point, measured instead of modelled).
// The segment is the line from the camera to the leader. The returned point is the part of this obstacle that
// comes closest to blocking the robot's view of the leader. NaN when the cluster has no points.
inline Point2 FindClusterPointNearestSegment(const Cluster & cluster, const Point2 & segmentStart,
  const Point2 & segmentEnd)
{
  double nearestDistance = std::numeric_limits<double>::infinity();
  Point2 nearestPoint = MakeNotANumberPoint();
  for (const auto & clusterPoint : cluster.points) {
    const double segmentDistance =
      CalculateDistance(clusterPoint, FindNearestPointOnSegment(segmentStart, segmentEnd, clusterPoint));
    if (segmentDistance < nearestDistance) {
      nearestDistance = segmentDistance;
      nearestPoint = clusterPoint;
    }
  }
  return nearestPoint;
}

}  // namespace lidar_obstacles
