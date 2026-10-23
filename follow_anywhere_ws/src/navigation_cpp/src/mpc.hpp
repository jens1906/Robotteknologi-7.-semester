#pragma once

#include <array>
#include <optional>

#include <Eigen/Dense>

namespace navigation_cpp
{

// The fields of perception_msgs/Detection the MPC uses for the leader.
struct Leader
{
  double range = 0.0;
  double bearing = 0.0;
  double vx = 0.0;            // velocity.x
  double vy = 0.0;            // velocity.y
  double velocity_std = 0.0;
};

// python: (i.nearest_range, i.nearest_bearing, speed_triplet)
struct Obstacle
{
  Eigen::Vector2d polar = Eigen::Vector2d::Zero();  // (range, bearing)
  Eigen::Vector2d cart = Eigen::Vector2d::Zero();   // (r cos b, r sin b)
  Eigen::Vector2d vel = Eigen::Vector2d::Zero();    // (vx, vy)
  double std = 0.0;
  double spd = 0.0;                                 // vel.norm()
};

Obstacle make_obstacle(
  double range, double bearing,
  double vx = 0.0, double vy = 0.0, double std = 0.0);

constexpr int MAX_OBSTACLES = 10;

// Preallocated list of the MAX_OBSTACLES nearest obstacles, sorted by range.
struct ObstacleList
{
  std::array<Obstacle, MAX_OBSTACLES> items{};
  int size = 0;

  void clear() {size = 0;}
  // keeps items sorted by polar[0]; when full, drops the farthest
  void push_nearest(const Obstacle & o);
};

// python: FollowLeader.A() / FollowLeader.B()
Eigen::Matrix2d A(const Leader & leader, const Eigen::Vector3d & speed);
Eigen::Matrix<double, 2, 3> B(const Leader & leader);

// The MPC part of python's FollowLeader.compute() (from `dt = self.dt` on).
// Returns u.value (3 x N) if the problem was solved, otherwise nullopt.
std::optional<Eigen::MatrixXd> compute_mpc(
  const Leader & leader,
  const ObstacleList & obstacles,
  const std::optional<Eigen::MatrixXd> & last_path,
  double dt);

}  // namespace navigation_cpp
