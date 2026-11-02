#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "tmp_lidar/cloud_filter.hpp"
#include "tmp_lidar/cluster_tracker.hpp"
#include "tmp_lidar/clustering.hpp"
#include "tmp_lidar/fusion.hpp"
#include "tmp_lidar/hit_grid.hpp"
#include "tmp_lidar/scan_builder.hpp"

using namespace tmp_lidar;

namespace
{

// Points on the near half of a circle (what a lidar sees of a person or pole).
std::vector<Point2> blob(double cx, double cy, double radius = 0.2, int n = 13)
{
  std::vector<Point2> points;
  const double toward = std::atan2(-cy, -cx);
  for (int i = 0; i < n; ++i) {
    const double a = toward - M_PI / 2 + M_PI * i / (n - 1);
    points.push_back({cx + radius * std::cos(a), cy + radius * std::sin(a)});
  }
  return points;
}

std::vector<Cluster> clustersOf(std::vector<Point2> points)
{
  auto clusters = clusterPoints(points, ClusterParams{});
  ClusterTracker tracker;
  tracker.update(clusters, Pose2{}, true, 0.0);
  return clusters;
}

}  // namespace

TEST(CloudFilter, RemovesFloorBodyAndFarPoints)
{
  FilterParams p;
  GroundPlane flat;
  flat.c = -0.38;
  std::vector<Point3> cloud = {
    {2.0, 0.0, -0.38},     // floor
    {2.0, 0.0, -0.29},     // floor (within groundHeight)
    {0.2, 0.0, 0.0},       // body
    {9.0, 0.0, 0.0},       // too far
    {2.0, 0.0, 2.0},       // above maxHeight
    {2.0, 0.5, 0.0},       // kept
    {NAN, 0.0, 0.0},       // invalid
  };
  const auto out = filterCloud(cloud, p, flat);
  ASSERT_EQ(out.points.size(), 1u);
  EXPECT_DOUBLE_EQ(out.points[0].y, 0.5);
}

// A floor tilted 3 deg (body pitched) with 2 cm noise out to 6 m, and a 0.5 m box standing on it at 4 m.
TEST(CloudFilter, TiltedFloorIsRemovedBoxStays)
{
  std::mt19937 rng(1);
  std::normal_distribution<double> noise(0.0, 0.02);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  const double slope = std::tan(3.0 * M_PI / 180.0);
  auto floorZ = [&](double x) {return -0.38 + slope * x;};
  std::vector<Point3> cloud;
  for (int i = 0; i < 1500; ++i) {
    const double x = 6.0 * uni(rng), y = 6.0 * uni(rng);
    if (std::hypot(x, y) > 6.0) {continue;}
    cloud.push_back({x, y, floorZ(x) + noise(rng)});
  }
  int boxPoints = 0;
  for (int i = 0; i < 40; ++i) {
    const double y = 0.25 * uni(rng);
    cloud.push_back({4.0, y, floorZ(4.0) + 0.15 + 0.35 * (uni(rng) + 1.0) / 2.0});
    ++boxPoints;
  }

  FilterParams p;
  GroundEstimator ground;
  const GroundPlane plane = ground.update(cloud, p);
  EXPECT_TRUE(plane.fitted);
  EXPECT_NEAR(plane.tiltDeg(), 3.0, 0.5);
  EXPECT_NEAR(plane.c, -0.38, 0.02);
  const auto out = filterCloud(cloud, p, plane);
  int boxKept = 0, floorKept = 0;
  for (const auto & q : out.points) {(std::abs(q.x - 4.0) < 0.01 ? boxKept : floorKept)++;}
  EXPECT_EQ(boxKept, boxPoints);
  EXPECT_LE(floorKept, 2);   // > 5 sigma of noise only

  // The same cloud with a flat floor at -0.38 would keep the tilted floor far ahead.
  GroundPlane flat;
  flat.c = -0.38;
  EXPECT_GT(static_cast<int>(filterCloud(cloud, p, flat).points.size()) - boxPoints, 50);
}

TEST(CloudFilter, KeepsLastPlaneWhenFitFails)
{
  FilterParams p;
  GroundEstimator ground;
  std::vector<Point3> floor;
  for (int i = 0; i < 200; ++i) {floor.push_back({1.0 + 0.01 * i, 0.3 * std::sin(i), -0.36});}
  EXPECT_NEAR(ground.update(floor, p).c, -0.36, 1e-6);
  const std::vector<Point3> nothing = {{1.0, 0.0, 0.5}};
  const auto kept = ground.update(nothing, p);
  EXPECT_NEAR(kept.c, -0.36, 1e-6);
}

TEST(HitGrid, SingleHitIsNotOccupiedTwoAre)
{
  HitGrid grid;
  grid.add({{1.05, 1.05}}, 0.0);
  EXPECT_TRUE(grid.occupiedPoints(0.0).empty());
  grid.add({{1.07, 1.02}, {3.0, 3.0}}, 0.07);
  const auto points = grid.occupiedPoints(0.07);
  EXPECT_EQ(points.size(), 2u);   // both points of the twice-hit cell, not the speckle at (3, 3)
  // Two points of the same cloud in one cell are one hit.
  HitGrid other;
  other.add({{1.05, 1.05}, {1.06, 1.06}}, 0.0);
  EXPECT_TRUE(other.occupiedPoints(0.0).empty());
  // Forgotten after memorySeconds.
  grid.add({}, 1.0);
  EXPECT_TRUE(grid.occupiedPoints(1.0).empty());
}

TEST(Clustering, SeparatesBlobsAndDropsStrays)
{
  auto points = blob(2.0, 0.0);
  const auto second = blob(0.0, 3.0);
  points.insert(points.end(), second.begin(), second.end());
  points.push_back({-4.0, -4.0});   // stray
  const auto clusters = clusterPoints(points, ClusterParams{});
  ASSERT_EQ(clusters.size(), 2u);
  for (const auto & c : clusters) {EXPECT_FALSE(c.isChunk);}
}

TEST(Clustering, SplitsWalls)
{
  std::vector<Point2> wall;
  for (int i = 0; i < 100; ++i) {wall.push_back({3.0, -2.5 + 0.05 * i});}
  const auto clusters = clusterPoints(wall, ClusterParams{});
  ASSERT_GT(clusters.size(), 5u);
  for (const auto & c : clusters) {
    EXPECT_TRUE(c.isChunk);
    EXPECT_LE(c.radius, 0.5);
  }
}

TEST(Tracker, EstimatesVelocityOverTheGround)
{
  ClusterTracker tracker;
  Cluster c;
  for (int k = 0; k < 20; ++k) {
    const double t = 0.1 * k;
    std::vector<Cluster> clusters = clusterPoints(blob(2.0 + 1.0 * t, 0.0), ClusterParams{});
    tracker.update(clusters, Pose2{}, true, t);
    ASSERT_EQ(clusters.size(), 1u);
    c = clusters[0];
  }
  EXPECT_NEAR(c.velocity.x, 1.0, 0.05);
  EXPECT_NEAR(c.velocity.y, 0.0, 0.05);
}

TEST(Tracker, StandingObjectSeenFromMovingRobotIsStill)
{
  ClusterTracker tracker;
  Cluster c;
  for (int k = 0; k < 20; ++k) {
    const double t = 0.1 * k;
    const Pose2 robot{0.5 * t, 0.0, 0.0};
    std::vector<Cluster> clusters = clusterPoints(blob(3.0 - robot.x, 0.0), ClusterParams{});
    tracker.update(clusters, robot, true, t);
    c = clusters[0];
  }
  EXPECT_DOUBLE_EQ(c.velocity.x, 0.0);
}

TEST(Fusion, EnrichesMatchedPersonAndDeduplicates)
{
  auto points = blob(2.0, 0.0);
  const auto wall = blob(0.0, 3.0);
  points.insert(points.end(), wall.begin(), wall.end());
  const auto clusters = clustersOf(points);
  ASSERT_EQ(clusters.size(), 2u);

  FusionInput in;
  in.haveCamera = true;
  in.leader.status = kNoLeader;
  ObstacleData person;
  person.closest = {1.9, 0.1};   // camera: a bit off
  person.positionSigma = 0.3;
  person.velocitySigma = 1.5;
  in.obstacles.push_back(person);

  Fusion fusion;
  const auto out = fusion.fuse(in, clusters, Pose2{}, 0.0);
  ASSERT_EQ(out.obstacles.size(), 2u);   // the person once + the other cluster, not 3
  const auto & p = out.obstacles[0];
  EXPECT_EQ(p.source, Source::CameraLidar);
  EXPECT_NEAR(p.data.closest.x, 1.8, 1e-6);   // nearest lidar point of the blob
  EXPECT_NEAR(p.data.closest.y, 0.0, 1e-6);
  EXPECT_LT(p.data.positionSigma, 0.1);
  EXPECT_EQ(out.obstacles[1].source, Source::Lidar);
  EXPECT_NEAR(out.obstacles[1].data.closest.y, 2.8, 1e-6);
}

TEST(Fusion, UnmatchedCameraPersonPassesThrough)
{
  Fusion fusion;
  FusionInput in;
  in.haveCamera = true;
  ObstacleData person;
  person.closest = {5.0, 0.0};
  person.positionSigma = 0.1;
  in.obstacles.push_back(person);
  const auto out = fusion.fuse(in, clustersOf(blob(0.0, 3.0)), Pose2{}, 0.0);
  ASSERT_EQ(out.obstacles.size(), 2u);
  EXPECT_EQ(out.obstacles[0].source, Source::Camera);
  EXPECT_DOUBLE_EQ(out.obstacles[0].data.closest.x, 5.0);
}

TEST(Fusion, LeaderEnrichedAndLineOfSightFromLidar)
{
  auto points = blob(3.0, 0.0);
  const auto blocker = blob(1.5, 0.3);   // partly in the line camera -> leader
  points.insert(points.end(), blocker.begin(), blocker.end());
  const auto clusters = clustersOf(points);

  FusionInput in;
  in.haveCamera = true;
  in.leader.status = kVisible;
  in.leader.closest = {2.85, 0.05};
  in.leader.positionSigma = 0.2;
  in.leader.velocitySigma = 0.3;
  in.leader.lostDuration = 0.0;
  Fusion fusion;
  const auto out = fusion.fuse(in, clusters, Pose2{}, 0.0);
  EXPECT_TRUE(out.leaderEnriched);
  EXPECT_NEAR(out.leader.closest.x, 2.8, 1e-6);
  EXPECT_NEAR(out.leader.lastKnown.x, 2.8, 1e-6);
  ASSERT_EQ(out.obstacles.size(), 1u);   // only the blocker: the leader is not an obstacle
  const Point2 los = out.obstacles[0].data.lineOfSight;
  ASSERT_TRUE(isFinite(los));
  EXPECT_LT(los.y, 0.3);   // the blocker's point nearest the line y = 0
}

TEST(Fusion, LostLeaderHoldsItsClusterAndKeepsPointInPlace)
{
  Fusion fusion;
  FusionInput in;
  in.haveCamera = true;
  in.leader.status = kVisible;
  in.leader.closest = {2.8, 0.0};
  in.leader.positionSigma = 0.2;
  auto clusters = clustersOf(blob(3.0, 0.0));
  fusion.fuse(in, clusters, Pose2{}, 0.0);

  in.leader.status = kLost;
  in.leader.closest = nanPoint();
  in.leader.lostDuration = 0.5;
  const auto held = fusion.fuse(in, clusters, Pose2{}, 0.5);
  EXPECT_EQ(held.leaderCluster, 0);
  EXPECT_TRUE(held.obstacles.empty());

  // After leaderHoldTime the cluster is an ordinary obstacle; the point stays on the floor while driving.
  const Pose2 moved{1.0, 0.0, 0.0};
  const auto later = fusion.fuse(in, {}, moved, 2.0);
  EXPECT_NEAR(later.leader.lastKnown.x, 1.8, 1e-6);
}

TEST(Scan, NearestPointPerRay)
{
  const auto ranges = buildScan({{2.0, 0.0}, {3.0, 0.0}, {0.0, -1.0}}, 360, 0.3, 10.0);
  ASSERT_EQ(ranges.size(), 360u);
  EXPECT_FLOAT_EQ(ranges[180], 2.0f);   // bearing 0
  EXPECT_FLOAT_EQ(ranges[90], 1.0f);    // bearing -pi/2
  EXPECT_TRUE(std::isinf(ranges[0]));
}

TEST(Tracker, UnconfirmedNotPublishedConfirmedCoastsThenDrops)
{
  ClusterTracker tracker;   // confirmHits 2, coastTime 0.3
  auto first = clusterPoints(blob(2.0, 0.0), ClusterParams{});
  tracker.update(first, Pose2{}, true, 0.0);
  EXPECT_TRUE(tracker.publishable(first, Pose2{}, 0.0).empty());   // seen once

  auto second = clusterPoints(blob(2.0, 0.0), ClusterParams{});
  tracker.update(second, Pose2{}, true, 0.07);
  auto out = tracker.publishable(second, Pose2{}, 0.07);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_FALSE(out[0].isCoasting);

  std::vector<Cluster> none;
  tracker.update(none, Pose2{}, true, 0.14);   // missed by the lidar
  out = tracker.publishable(none, Pose2{}, 0.14);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_TRUE(out[0].isCoasting);
  EXPECT_NEAR(out[0].closest.x, 1.8, 1e-6);
  EXPECT_GT(out[0].extraSigma, 0.0);

  // Coasting points stay on the floor while the robot drives.
  const Pose2 moved{0.5, 0.0, 0.0};
  out = tracker.publishable(none, moved, 0.2);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_NEAR(out[0].closest.x, 1.3, 1e-6);

  // At 1.8 m the limit is coastTime + 1.8 / coastFarRange * (coastTimeFar - coastTime) = 0.48 s.
  EXPECT_NEAR(tracker.coastLimit(1.8), 0.48, 1e-9);
  tracker.update(none, Pose2{}, true, 0.6);
  EXPECT_TRUE(tracker.publishable(none, Pose2{}, 0.6).empty());   // missed for 0.53 s
}

TEST(Tracker, FarObjectsCoastLonger)
{
  ClusterTracker tracker;
  for (double t : {0.0, 0.07}) {
    auto c = clusterPoints(blob(5.2, 0.0), ClusterParams{});
    tracker.update(c, Pose2{}, true, t);
  }
  std::vector<Cluster> none;
  tracker.update(none, Pose2{}, true, 0.6);
  EXPECT_EQ(tracker.publishable(none, Pose2{}, 0.6).size(), 1u);   // 0.53 s missed, limit 0.8 s at 5 m
  tracker.update(none, Pose2{}, true, 0.9);
  EXPECT_TRUE(tracker.publishable(none, Pose2{}, 0.9).empty());
}

TEST(Clustering, OdomGridGivesBaseLinkClusters)
{
  const Pose2 robot{10.0, 5.0, M_PI / 2};
  std::vector<Point2> odom;
  for (const auto & p : blob(2.0, 0.0)) {odom.push_back(robot.toOdom(p));}
  const auto clusters = clusterPoints(odom, ClusterParams{}, robot);
  ASSERT_EQ(clusters.size(), 1u);
  EXPECT_NEAR(clusters[0].closest.x, 1.8, 1e-6);
  EXPECT_NEAR(clusters[0].closest.y, 0.0, 1e-6);
}
