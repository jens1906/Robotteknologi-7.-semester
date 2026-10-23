#include <chrono>
#include <cmath>
#include <initializer_list>

#include <gtest/gtest.h>

#include "mpc.hpp"

using navigation_cpp::compute_mpc;
using navigation_cpp::Leader;
using navigation_cpp::make_obstacle;
using navigation_cpp::MAX_OBSTACLES;
using navigation_cpp::Obstacle;
using navigation_cpp::ObstacleList;

static constexpr double DT = 0.1;

static ObstacleList list(std::initializer_list<Obstacle> obstacles)
{
  ObstacleList l;
  for (const Obstacle & o : obstacles) {
    l.push_nearest(o);
  }
  return l;
}

TEST(Mpc, DrivesTowardsFarLeaderAndTurns)
{
  const auto u = compute_mpc(Leader{3.0, 0.3}, {}, std::nullopt, DT);
  ASSERT_TRUE(u);
  EXPECT_GT((*u)(0, 0), 0.0);
  EXPECT_GT((*u)(2, 0), 0.0);
}

TEST(Mpc, StandsStillAtReference)
{
  const auto u = compute_mpc(Leader{2.0, 0.0}, {}, std::nullopt, DT);
  ASSERT_TRUE(u);
  EXPECT_NEAR((*u)(0, 0), 0.0, 1e-3);
  EXPECT_NEAR((*u)(1, 0), 0.0, 1e-3);
  EXPECT_NEAR((*u)(2, 0), 0.0, 1e-3);
}

TEST(Mpc, DoesNotDriveIntoObstacleAhead)
{
  const auto free = compute_mpc(Leader{4.0, 0.0}, {}, std::nullopt, DT);
  ASSERT_TRUE(free);
  const auto u = compute_mpc(Leader{4.0, 0.0}, list({make_obstacle(1.2, 0.0)}), std::nullopt, DT);
  if (u) {
    EXPECT_LT((*u)(0, 0), (*free)(0, 0));
  }
}

TEST(Mpc, WarmStartWithLastPath)
{
  const auto first = compute_mpc(Leader{3.0, 0.2}, list({make_obstacle(2.0, 1.0)}), std::nullopt, DT);
  ASSERT_TRUE(first);
  const auto second = compute_mpc(Leader{3.0, 0.2}, list({make_obstacle(2.0, 1.0)}), first, DT);
  ASSERT_TRUE(second);
  EXPECT_TRUE(std::isfinite((*second)(0, 0)));
}

TEST(Mpc, SolvesFast)
{
  const ObstacleList obstacles = list({
    make_obstacle(2.0, 0.5), make_obstacle(3.0, -0.4),
    make_obstacle(1.5, 1.2, 0.3, 0.0, 0.05)});
  const auto t0 = std::chrono::steady_clock::now();
  const int runs = 100;
  for (int i = 0; i < runs; ++i) {
    compute_mpc(Leader{3.0, 0.1}, obstacles, std::nullopt, DT);
  }
  const double ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - t0).count() / runs;
  RecordProperty("mean_solve_ms", std::to_string(ms));
  std::printf("mean solve: %.3f ms\n", ms);
  EXPECT_LT(ms, 5.0);
}

TEST(ObstacleList, KeepsNearest)
{
  ObstacleList l;
  for (int r = 15; r >= 1; --r) {
    l.push_nearest(make_obstacle(r, 0.0));
  }
  ASSERT_EQ(l.size, MAX_OBSTACLES);
  for (int i = 0; i < l.size; ++i) {
    EXPECT_DOUBLE_EQ(l.items[i].polar[0], i + 1.0);
  }

  // a farther one is dropped, a nearer one pushes out the farthest
  l.push_nearest(make_obstacle(20.0, 0.0));
  EXPECT_DOUBLE_EQ(l.items[MAX_OBSTACLES - 1].polar[0], 10.0);
  l.push_nearest(make_obstacle(0.5, 0.0));
  EXPECT_DOUBLE_EQ(l.items[0].polar[0], 0.5);
  EXPECT_DOUBLE_EQ(l.items[MAX_OBSTACLES - 1].polar[0], 9.0);
}
