#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "lidar_obstacles/cloud_filter.hpp"
#include "lidar_obstacles/cluster_tracker.hpp"
#include "lidar_obstacles/clustering.hpp"
#include "lidar_obstacles/fusion.hpp"

using namespace lidar_obstacles;

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
  auto clusters = ClusterPoints(points, ClusterParams{});
  ClusterTracker tracker;
  tracker.Update(clusters, Pose2{}, true, 0.0);
  return clusters;
}

}  // namespace

// Checks FilterCloud on a flat floor: floor points, points inside the robot's body box, points too far away and
// points above maxHeight are all removed; only the one real obstacle point is kept.
TEST(CloudFilter, RemovesFloorBodyAndFarPoints)
{
  FilterParams p;
  GroundPlane flat;
  flat.heightAtOrigin = -0.38;
  std::vector<Point3> cloud = {
    {2.0, 0.0, -0.38},     // floor
    {2.0, 0.0, -0.29},     // floor (within groundHeight)
    {0.2, 0.0, 0.0},       // body
    {9.0, 0.0, 0.0},       // too far
    {2.0, 0.0, 2.0},       // above maxHeight
    {2.0, 0.5, 0.0},       // kept
    {NAN, 0.0, 0.0},       // invalid
  };
  const auto out = FilterCloud(cloud, p, flat);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_DOUBLE_EQ(out[0].y, 0.5);
}

TEST(CloudFilter, LowPointsNextToTheBodyAreFeet)
{
  FilterParams p;
  GroundPlane flat;
  flat.heightAtOrigin = -0.38;
  std::vector<Point3> cloud = {
    {0.3, -0.3, -0.23},
    {0.3, -0.3, 0.02},
    {1.0, 0.0, -0.23},
  };
  const auto out = FilterCloud(cloud, p, flat);
  ASSERT_EQ(out.size(), 2u);
  EXPECT_DOUBLE_EQ(out[0].x, 0.3);
  EXPECT_DOUBLE_EQ(out[1].x, 1.0);
}

// A floor tilted 3 deg (body pitched) with 2 cm noise out to 6 m, and a 0.5 m box standing on it at 4 m.
// Checks the floor plane fit (GroundEstimator): the tilt and height are found, the tilted floor is removed
// everywhere and the box points are kept.
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
  const GroundPlane plane = ground.Update(cloud, p);
  EXPECT_TRUE(plane.fitted);
  EXPECT_NEAR(plane.CalculateTiltDegrees(), 3.0, 0.5);
  EXPECT_NEAR(plane.heightAtOrigin, -0.38, 0.02);
  const auto out = FilterCloud(cloud, p, plane);
  int boxKept = 0, floorKept = 0;
  for (const auto & q : out) {(std::abs(q.x - 4.0) < 0.01 ? boxKept : floorKept)++;}
  EXPECT_EQ(boxKept, boxPoints);
  EXPECT_LE(floorKept, 2);   // > 5 sigma of noise only

  // The same cloud with a flat floor at -0.38 would keep the tilted floor far ahead.
  GroundPlane flat;
  flat.heightAtOrigin = -0.38;
  EXPECT_GT(static_cast<int>(FilterCloud(cloud, p, flat).size()) - boxPoints, 50);
}

// Checks that a cloud without enough floor points does not reset the floor: the last good plane is kept.
TEST(CloudFilter, KeepsLastPlaneWhenFitFails)
{
  FilterParams p;
  GroundEstimator ground;
  std::vector<Point3> floor;
  for (int i = 0; i < 200; ++i) {floor.push_back({1.0 + 0.01 * i, 0.3 * std::sin(i), -0.36});}
  EXPECT_NEAR(ground.Update(floor, p).heightAtOrigin, -0.36, 1e-6);
  const std::vector<Point3> nothing = {{1.0, 0.0, 0.5}};
  const auto kept = ground.Update(nothing, p);
  EXPECT_NEAR(kept.heightAtOrigin, -0.36, 1e-6);
}

// Checks the hit grid: a cell hit by one cloud is not occupied, a cell hit by two different clouds is, and
// everything is forgotten after memorySeconds.
TEST(HitGrid, SingleHitIsNotOccupiedTwoAre)
{
  HitGrid grid;
  grid.Add({{1.05, 1.05}}, 0.0);
  EXPECT_TRUE(grid.FindOccupiedPoints(0.0).empty());
  grid.Add({{1.07, 1.02}, {3.0, 3.0}}, 0.07);
  const auto points = grid.FindOccupiedPoints(0.07);
  EXPECT_EQ(points.size(), 2u);   // both points of the twice-hit cell, not the speckle at (3, 3)
  // Two points of the same cloud in one cell are one hit.
  HitGrid other;
  other.Add({{1.05, 1.05}, {1.06, 1.06}}, 0.0);
  EXPECT_TRUE(other.FindOccupiedPoints(0.0).empty());
  // Forgotten after memorySeconds.
  grid.Add({}, 1.0);
  EXPECT_TRUE(grid.FindOccupiedPoints(1.0).empty());
}

// Checks the clustering: two separate objects become two clusters, and single stray points are dropped.
TEST(Clustering, SeparatesBlobsAndDropsStrays)
{
  auto points = blob(2.0, 0.0);
  const auto second = blob(0.0, 3.0);
  points.insert(points.end(), second.begin(), second.end());
  points.push_back({-4.0, -4.0});   // stray
  const auto clusters = ClusterPoints(points, ClusterParams{});
  ASSERT_EQ(clusters.size(), 2u);
  for (const auto & c : clusters) {EXPECT_FALSE(c.isChunk);}
}

// Checks that a cluster larger than maxClusterRadius (a wall) is cut into chunk pieces.
TEST(Clustering, SplitsWalls)
{
  std::vector<Point2> wall;
  for (int i = 0; i < 100; ++i) {wall.push_back({3.0, -2.5 + 0.05 * i});}
  const auto clusters = ClusterPoints(wall, ClusterParams{});
  ASSERT_GT(clusters.size(), 5u);
  for (const auto & c : clusters) {
    EXPECT_TRUE(c.isChunk);
    EXPECT_LE(c.radius, 0.5);
  }
}

// Checks that the tracker finds the right velocity of an object moving at a constant speed.
TEST(Tracker, EstimatesVelocityOverTheGround)
{
  ClusterTracker tracker;
  Cluster c;
  for (int k = 0; k < 20; ++k) {
    const double t = 0.1 * k;
    std::vector<Cluster> clusters = ClusterPoints(blob(2.0 + 1.0 * t, 0.0), ClusterParams{});
    tracker.Update(clusters, Pose2{}, true, t);
    ASSERT_EQ(clusters.size(), 1u);
    c = clusters[0];
  }
  EXPECT_NEAR(c.velocity.x, 1.0, 0.05);
  EXPECT_NEAR(c.velocity.y, 0.0, 0.05);
}

// Checks that a standing object gets velocity 0 even while the robot itself drives past it (the robot's own
// motion is removed through odom).
TEST(Tracker, StandingObjectSeenFromMovingRobotIsStill)
{
  ClusterTracker tracker;
  Cluster c;
  for (int k = 0; k < 20; ++k) {
    const double t = 0.1 * k;
    const Pose2 robot{0.5 * t, 0.0, 0.0};
    std::vector<Cluster> clusters = ClusterPoints(blob(3.0 - robot.x, 0.0), ClusterParams{});
    tracker.Update(clusters, robot, true, t);
    c = clusters[0];
  }
  EXPECT_DOUBLE_EQ(c.velocity.x, 0.0);
}

// Checks that a camera person next to a lidar cluster is matched to it: the person gets the lidar's nearest
// point and accuracy, and the cluster is not reported a second time.
TEST(Fusion, EnrichesMatchedPersonAndDeduplicates)
{
  auto points = blob(2.0, 0.0);
  const auto wall = blob(0.0, 3.0);
  points.insert(points.end(), wall.begin(), wall.end());
  const auto clusters = clustersOf(points);
  ASSERT_EQ(clusters.size(), 2u);

  FusionInput in;
  in.hasCamera = true;
  in.leader.status = NO_LEADER;
  ObstacleData person;
  person.closest = {1.9, 0.1};   // camera: a bit off
  person.positionSigma = 0.3;
  person.velocitySigma = 1.5;
  in.obstacles.push_back(person);

  Fusion fusion;
  const auto out = fusion.Fuse(in, clusters, Pose2{}, 0.0);
  ASSERT_EQ(out.obstacles.size(), 2u);   // the person once + the other cluster, not 3
  const auto & p = out.obstacles[0];
  EXPECT_FALSE(p.isLidarOnly);
  EXPECT_NEAR(p.data.closest.x, 1.8, 1e-6);   // nearest lidar point of the blob
  EXPECT_NEAR(p.data.closest.y, 0.0, 1e-6);
  EXPECT_LT(p.data.positionSigma, 0.1);
  EXPECT_TRUE(out.obstacles[1].isLidarOnly);
  EXPECT_NEAR(out.obstacles[1].data.closest.y, 2.8, 1e-6);
}

// Checks that a camera person with no lidar cluster nearby is passed through unchanged.
TEST(Fusion, UnmatchedCameraPersonPassesThrough)
{
  Fusion fusion;
  FusionInput in;
  in.hasCamera = true;
  ObstacleData person;
  person.closest = {5.0, 0.0};
  person.positionSigma = 0.1;
  in.obstacles.push_back(person);
  const auto out = fusion.Fuse(in, clustersOf(blob(0.0, 3.0)), Pose2{}, 0.0);
  ASSERT_EQ(out.obstacles.size(), 2u);
  EXPECT_FALSE(out.obstacles[0].isLidarOnly);
  EXPECT_DOUBLE_EQ(out.obstacles[0].data.closest.x, 5.0);
}

// Checks that the leader gets the lidar's position, is not reported as an obstacle, and that the line of
// sight point of a blocking obstacle comes from its lidar points.
TEST(Fusion, LeaderEnrichedAndLineOfSightFromLidar)
{
  auto points = blob(3.0, 0.0);
  const auto blocker = blob(1.5, 0.3);   // partly in the line camera -> leader
  points.insert(points.end(), blocker.begin(), blocker.end());
  const auto clusters = clustersOf(points);

  FusionInput in;
  in.hasCamera = true;
  in.leader.status = LEADER_VISIBLE;
  in.leader.closest = {2.85, 0.05};
  in.leader.positionSigma = 0.2;
  in.leader.velocitySigma = 0.3;
  in.leader.lostDuration = 0.0;
  Fusion fusion;
  const auto out = fusion.Fuse(in, clusters, Pose2{}, 0.0);
  EXPECT_EQ(std::count(out.clusterOwner.begin(), out.clusterOwner.end(), OWNER_LEADER), 1);
  EXPECT_NEAR(out.leader.closest.x, 2.8, 1e-6);
  EXPECT_NEAR(out.leader.lastKnown.x, 2.8, 1e-6);
  ASSERT_EQ(out.obstacles.size(), 1u);   // only the blocker: the leader is not an obstacle
  const Point2 los = out.obstacles[0].data.lineOfSight;
  ASSERT_TRUE(IsFinite(los));
  EXPECT_LT(los.y, 0.3);   // the blocker's point nearest the line y = 0
}

// Checks the leader hold: when the camera loses the leader, its cluster stays the leader's for leaderHoldTime,
// and the last known point stays on the same floor spot while the robot moves.
TEST(Fusion, LostLeaderHoldsItsClusterAndKeepsPointInPlace)
{
  Fusion fusion;
  FusionInput in;
  in.hasCamera = true;
  in.leader.status = LEADER_VISIBLE;
  in.leader.closest = {2.8, 0.0};
  in.leader.positionSigma = 0.2;
  auto clusters = clustersOf(blob(3.0, 0.0));
  fusion.Fuse(in, clusters, Pose2{}, 0.0);

  in.leader.status = LEADER_LOST;
  in.leader.closest = MakeNotANumberPoint();
  in.leader.lostDuration = 0.5;
  const auto held = fusion.Fuse(in, clusters, Pose2{}, 0.5);
  EXPECT_EQ(held.clusterOwner[0], OWNER_LEADER);
  EXPECT_TRUE(held.obstacles.empty());

  // After leaderHoldTime the cluster is an ordinary obstacle; the point stays on the floor while driving.
  const Pose2 moved{1.0, 0.0, 0.0};
  const auto later = fusion.Fuse(in, {}, moved, 2.0);
  EXPECT_NEAR(later.leader.lastKnown.x, 1.8, 1e-6);
}

TEST(Fusion, VisibleLeaderWithoutDepthTakesItsCluster)
{
  Fusion fusion;
  FusionInput in;
  in.hasCamera = true;
  in.leader.status = LEADER_VISIBLE;
  in.leader.closest = {2.8, 0.0};
  in.leader.positionSigma = 0.2;
  auto clusters = clustersOf(blob(3.0, 0.0));
  fusion.Fuse(in, clusters, Pose2{}, 0.0);

  in.leader.closest = MakeNotANumberPoint();
  const auto noDepth = fusion.Fuse(in, clusters, Pose2{}, 0.1);
  EXPECT_NEAR(noDepth.leader.closest.x, 2.8, 1e-6);
  EXPECT_EQ(noDepth.clusterOwner[0], OWNER_LEADER);
  EXPECT_TRUE(noDepth.obstacles.empty());

  const auto afterHold = fusion.Fuse(in, clusters, Pose2{}, 2.0);
  EXPECT_FALSE(IsFinite(afterHold.leader.closest));
}

// Checks that clustering odom points with a robot pose returns the clusters in base_link.
TEST(Clustering, OdomGridGivesBaseLinkClusters)
{
  const Pose2 robot{10.0, 5.0, M_PI / 2};
  std::vector<Point2> odom;
  for (const auto & p : blob(2.0, 0.0)) {odom.push_back(robot.ConvertToOdom(p));}
  const auto clusters = ClusterPoints(odom, ClusterParams{}, robot);
  ASSERT_EQ(clusters.size(), 1u);
  EXPECT_NEAR(clusters[0].closest.x, 1.8, 1e-6);
  EXPECT_NEAR(clusters[0].closest.y, 0.0, 1e-6);
}

// --- persistence -----------------------------------------------------------------------------------------

namespace
{

// Feeds a blob at (x, y) (or nothing when !seen) to the tracker at time t; returns what is published.
std::vector<Cluster> step(ClusterTracker & tracker, double x, double y, bool seen, double t,
  const Pose2 & robot = Pose2{})
{
  std::vector<Cluster> clusters;
  if (seen) {clusters = ClusterPoints(blob(x, y), ClusterParams{});}
  tracker.Update(clusters, robot, true, t);
  return tracker.FindPublishable(clusters, robot, t);
}

constexpr double kCloud = 1.0 / 15.0;

}  // namespace

// Checks that a cluster seen only once (noise) never becomes a published obstacle.
TEST(Persistence, SingleSpuriousClusterIsNeverPublished)
{
  ClusterTracker tracker;
  EXPECT_TRUE(step(tracker, 2.0, 0.0, true, 0.0).empty());
  for (int k = 1; k < 10; ++k) {EXPECT_TRUE(step(tracker, 2.0, 0.0, false, k * kCloud).empty());}
}

// Checks the track life cycle: confirmed after being seen again, still published (coasting) for a while
// after it disappears, then removed.
TEST(Persistence, ConfirmedOnSecondSightingCoastsThenDrops)
{
  ClusterTracker tracker;
  step(tracker, 2.0, 0.0, true, 0.0);
  auto out = step(tracker, 2.0, 0.0, true, kCloud);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_FALSE(out[0].isCoasting);
  for (int k = 2; k < 15; ++k) {step(tracker, 2.0, 0.0, true, k * kCloud);}

  out = step(tracker, 2.0, 0.0, false, 15 * kCloud);   // missed
  ASSERT_EQ(out.size(), 1u);
  EXPECT_TRUE(out[0].isCoasting);
  EXPECT_NEAR(out[0].closest.x, 1.8, 1e-6);   // standing: stays where it was seen

  // Coasting points stay on the floor while the robot drives.
  out = tracker.FindPublishable({}, Pose2{0.5, 0.0, 0.0}, 15 * kCloud);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_NEAR(out[0].closest.x, 1.3, 1e-6);

  // Gone for good: dropped within trackTimeout.
  int published = 1;
  for (int k = 16; k < 31; ++k) {published = static_cast<int>(step(tracker, 2.0, 0.0, false, k * kCloud).size());}
  EXPECT_EQ(published, 0);
}

// Checks that an object the lidar sees only in some clouds stays published without blinking.
TEST(Persistence, FlickeringObjectStaysPublished)
{
  // Near: missed every fifth cloud. Far (4.5 m): missed 3 clouds in a row, twice.
  ClusterTracker near, far;
  const std::vector<bool> farSeen = {1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0};
  for (int k = 0; k < 40; ++k) {
    const auto a = step(near, 1.5, 0.0, k % 5 != 4, k * kCloud);
    const auto b = step(far, 4.5, 0.0, farSeen[static_cast<std::size_t>(k) % farSeen.size()], k * kCloud);
    if (k >= 2) {
      EXPECT_EQ(a.size(), 1u) << "near, cloud " << k;
      EXPECT_EQ(b.size(), 1u) << "far, cloud " << k;
    }
  }
}

// Checks that an object hidden behind a nearer one is kept longer than one that just disappeared.
TEST(Persistence, OccludedObjectSurvivesLonger)
{
  ClusterTracker open, hidden;
  for (int k = 0; k < 15; ++k) {
    step(open, 3.0, 0.0, true, k * kCloud);
    step(hidden, 3.0, 0.0, true, k * kCloud);
  }
  int openAlive = 0, hiddenAlive = 0;
  for (int k = 15; k < 45; ++k) {
    std::vector<Cluster> none;
    open.Update(none, Pose2{}, true, k * kCloud);
    openAlive += open.FindPublishable(none, Pose2{}, k * kCloud).empty() ? 0 : 1;

    auto blocker = ClusterPoints(blob(1.5, 0.0, 0.3, 25), ClusterParams{});   // something walked in between
    hidden.Update(blocker, Pose2{}, true, k * kCloud);
    for (const auto & c : hidden.FindPublishable(blocker, Pose2{}, k * kCloud)) {
      if (c.isCoasting) {++hiddenAlive;}
    }
  }
  EXPECT_LE(openAlive, 15);     // gone within trackTimeout
  EXPECT_EQ(hiddenAlive, 30);   // 2 s occluded: still there
}

// Checks the detection probability: it falls with distance and is 0 outside the lidar's range.
TEST(Persistence, DetectionFallsWithRangeAndIsZeroOutOfView)
{
  ClusterTracker tracker;
  EXPECT_GT(tracker.CalculateDetectionProbability(0.8), tracker.CalculateDetectionProbability(4.0));
  EXPECT_EQ(tracker.CalculateDetectionProbability(0.1), 0.0);
  EXPECT_EQ(tracker.CalculateDetectionProbability(10.0), 0.0);
}

// --- motion model --------------------------------------------------------------------------------------

// Checks the motion model: when a walking object stops, its published velocity goes to 0 within a few clouds.
TEST(Imm, StopIsPickedUpQuickly)
{
  ClusterTracker tracker;
  double x = 2.0;
  Cluster c;
  int k = 0;
  for (; k < 30; ++k) {   // walk at 1 m/s for 2 s
    x += 1.0 * kCloud;
    auto clusters = ClusterPoints(blob(x, 0.0), ClusterParams{});
    tracker.Update(clusters, Pose2{}, true, k * kCloud);
    c = clusters[0];
  }
  EXPECT_NEAR(c.velocity.x, 1.0, 0.1);
  EXPECT_NE(c.motionMode, ImmFilter::STATIC_MODEL);   // CV or CT: on a straight line they are the same
  int stoppedAfter = -1;
  for (int j = 0; j < 30; ++j, ++k) {   // stand
    auto clusters = ClusterPoints(blob(x, 0.0), ClusterParams{});
    tracker.Update(clusters, Pose2{}, true, k * kCloud);
    if (stoppedAfter < 0 && clusters[0].velocity.x == 0.0) {stoppedAfter = j;}
  }
  ASSERT_GE(stoppedAfter, 0);
  EXPECT_LE(stoppedAfter * kCloud, 0.5);
}

// Checks the motion model on a person walking a circle: the IMM (with its turning model) estimates the velocity
// better than a filter with only the constant velocity model.
TEST(Imm, TurningModelFollowsACircleBetterThanConstantVelocity)
{
  ImmParams cvOnly;
  cvOnly.initialModeProbability = {1e-9, 1.0, 1e-9};
  cvOnly.modeSwitchRate = 0.0;
  ImmFilter imm, cv;
  const double radius = 2.0, speed = 1.0, R = 0.05 * 0.05;   // a person walking a curve
  const double w = speed / radius;
  std::mt19937 rng(3);
  std::normal_distribution<double> noise(0.0, 0.05);
  auto truth = [&](double t) {return Point2{radius * std::cos(w * t), radius * std::sin(w * t)};};
  auto truthVelocity = [&](double t) {return Point2{-speed * std::sin(w * t), speed * std::cos(w * t)};};
  imm.Initialize(truth(0.0), R, 1.5, ImmParams{});
  cv.Initialize(truth(0.0), R, 1.5, cvOnly);
  double errorImm = 0.0, errorCv = 0.0;
  int n = 0;
  for (int k = 1; k < 300; ++k) {
    const double t = k * kCloud;
    imm.Predict(kCloud);
    cv.Predict(kCloud);
    const Point2 z{truth(t).x + noise(rng), truth(t).y + noise(rng)};
    imm.Update(z, R);
    cv.Update(z, R);
    if (k > 30) {   // after settling
      errorImm += std::pow(CalculateDistance(imm.CalculateMovingVelocity().velocity, truthVelocity(t)), 2);
      errorCv += std::pow(CalculateDistance(cv.CalculateMovingVelocity().velocity, truthVelocity(t)), 2);
      ++n;
    }
  }
  EXPECT_LT(std::sqrt(errorImm / n), 0.85 * std::sqrt(errorCv / n));
  EXPECT_GT(imm.GetModeProbability(ImmFilter::TURNING_MODEL), 0.5);
  EXPECT_NEAR(CalculateLength(imm.CalculateMovingVelocity().velocity), speed, 0.1);
}

// --- association ---------------------------------------------------------------------------------------

// Checks that SolveAssignment finds the cheapest total pairing, not just the cheapest single pair first.
TEST(Assignment, MinimumTotalCostNotGreedy)
{
  // Greedy would take (0,0) at cost 1 and leave row 1 with 10; the optimum is 2 + 2.
  const auto a = SolveAssignment({{1.0, 2.0}, {2.0, 10.0}}, 1e9);
  EXPECT_EQ(a[0], 1);
  EXPECT_EQ(a[1], 0);
  const auto b = SolveAssignment({{1e9, 1e9}, {3.0, 1e9}, {1.0, 1e9}}, 1e9);
  EXPECT_EQ(b[0], -1);
  EXPECT_EQ(b[1], -1);
  EXPECT_EQ(b[2], 0);
}

// Checks that two people walking past each other keep their own track ids.
TEST(Tracker, CrossingPeopleKeepTheirIds)
{
  ClusterTracker tracker;
  int idA = -1, idB = -1;
  for (int k = 0; k < 60; ++k) {
    const double t = k * kCloud;
    // A walks +x, B walks -x, 0.6 m apart in y, passing each other at t = 2 s.
    const double ax = 1.0 + 1.0 * t, bx = 5.0 - 1.0 * t;
    auto points = blob(ax, 0.3, 0.15, 9);
    const auto second = blob(bx, -0.3, 0.15, 9);
    points.insert(points.end(), second.begin(), second.end());
    auto clusters = ClusterPoints(points, ClusterParams{});
    tracker.Update(clusters, Pose2{}, true, t);
    for (const auto & c : clusters) {
      const bool isA = c.centroid.y > 0.0;
      int & id = isA ? idA : idB;
      if (k == 5) {id = c.trackId;}
      if (k > 5) {EXPECT_EQ(c.trackId, id) << (isA ? "A" : "B") << " at cloud " << k;}
    }
  }
  EXPECT_NE(idA, idB);
}

// Checks that two clusters belonging to one tracked person (e.g. two legs) are merged into one.
TEST(Tracker, PiecesOfOneTrackedPersonAreJoined)
{
  ClusterTracker tracker;
  for (int k = 0; k < 10; ++k) {
    auto c = ClusterPoints(blob(2.0, 0.0, 0.2, 13), ClusterParams{});
    tracker.Update(c, Pose2{}, true, k * kCloud);
  }
  // Now seen as two legs 0.3 m apart.
  auto points = blob(2.0, 0.15, 0.05, 5);
  const auto other = blob(2.0, -0.15, 0.05, 5);
  points.insert(points.end(), other.begin(), other.end());
  auto clusters = ClusterPoints(points, ClusterParams{});
  ASSERT_EQ(clusters.size(), 2u);
  tracker.Update(clusters, Pose2{}, true, 10 * kCloud);
  ASSERT_EQ(clusters.size(), 1u);
  EXPECT_EQ(clusters[0].points.size(), 10u);
  EXPECT_EQ(tracker.FindPublishable(clusters, Pose2{}, 10 * kCloud).size(), 1u);
}

// --- clustering ------------------------------------------------------------------------------------------

// Checks that a far object with sparse points stays one cluster (the link distance grows with range).
TEST(Clustering, SparseFarObjectIsOneCluster)
{
  std::vector<Point2> sparse;
  for (int i = 0; i < 5; ++i) {sparse.push_back({5.0, -0.3 + 0.18 * i});}   // 0.18 m apart at 5 m
  EXPECT_EQ(ClusterPoints(sparse, ClusterParams{}).size(), 1u);
  std::vector<Point2> near;
  for (int i = 0; i < 5; ++i) {near.push_back({1.0, -0.3 + 0.18 * i});}   // same spacing near: separate points
  EXPECT_TRUE(ClusterPoints(near, ClusterParams{}).empty());
}

// Checks that wall pieces keep the same chunk key while the robot moves (the chunk grid is in odom).
TEST(Clustering, WallPieceKeysStayWhileRobotMoves)
{
  std::vector<Point2> wall;
  for (int i = 0; i < 100; ++i) {wall.push_back({3.0, -2.5 + 0.05 * i});}   // odom
  std::vector<std::int64_t> before, after;
  for (const auto & c : ClusterPoints(wall, ClusterParams{}, Pose2{})) {before.push_back(c.chunkKey);}
  for (const auto & c : ClusterPoints(wall, ClusterParams{}, Pose2{0.7, 0.3, 0.4})) {after.push_back(c.chunkKey);}
  std::sort(before.begin(), before.end());
  std::sort(after.begin(), after.end());
  EXPECT_EQ(before, after);
}

// --- empty clouds, miss window, scan from obstacles -----------------------------------------------------

// Checks the cloud gate: nearly empty clouds are skipped, but not more than maxSkipped in a row.
TEST(CloudGate, SkipsEmptyCloudsButNotForever)
{
  CloudGate gate;   // minFraction 0.3, maxSkipped 4
  for (int k = 0; k < 10; ++k) {EXPECT_TRUE(gate.Accept(1400));}
  EXPECT_FALSE(gate.Accept(0));
  EXPECT_FALSE(gate.Accept(200));    // < 30 % of 1400
  EXPECT_TRUE(gate.Accept(1100));    // normal again
  for (int k = 0; k < 4; ++k) {EXPECT_FALSE(gate.Accept(0));}
  EXPECT_TRUE(gate.Accept(0));       // 5th in a row: the lidar really went quiet
  EXPECT_NEAR(gate.CalculateTypicalCount(), 1400.0, 1e-9);   // quiet clouds do not lower the typical count
}

// Checks the miss window: an object missing from a single cloud does not lose any existence.
TEST(Persistence, MissingFromOneCloudCostsNothing)
{
  // Alternately seen and missed at 4.5 m, forever: with missWindow 2 no miss ever counts.
  TrackerParams p;
  ClusterTracker windowed(p);
  p.missWindow = 1;
  ClusterTracker every(p);
  double rWindowed = 0.0, rEvery = 0.0;
  for (int k = 0; k < 60; ++k) {
    std::vector<Cluster> a, b;
    if (k % 2 == 0) {
      a = ClusterPoints(blob(4.5, 0.0), ClusterParams{});
      b = ClusterPoints(blob(4.5, 0.0), ClusterParams{});
    }
    windowed.Update(a, Pose2{}, true, k * kCloud);
    every.Update(b, Pose2{}, true, k * kCloud);
    for (auto & c : windowed.FindPublishable(a, Pose2{}, k * kCloud)) {rWindowed = c.existence;}
    for (auto & c : every.FindPublishable(b, Pose2{}, k * kCloud)) {rEvery = c.existence;}
  }
  EXPECT_GT(rWindowed, rEvery);
  EXPECT_GT(rWindowed, 0.99);
}

// Checks BuildScan: every ray across an obstacle gets a range (not just the rays its points fall on), and a
// cluster right behind the robot does not wrap all the way round.
TEST(Scan, ObstacleCoversItsWholeBearingSpan)
{
  // Three points of one object 0.2 m apart at 2 m: the scan covers all rays between them.
  Cluster c;
  c.points = {{2.0, -0.2}, {2.0, 0.0}, {2.0, 0.2}};
  FinalizeCluster(c);
  const auto filled = BuildScan(std::vector<Cluster>{c}, 720, 0.35, 6.0);
  auto returns = [](const std::vector<float> & r) {
      return std::count_if(r.begin(), r.end(), [](float v) {return std::isfinite(v);});
    };
  EXPECT_GE(returns(filled), 22);   // ~11.4 deg at 0.5 deg per ray
  // In between, the interpolated range: about 2 m, never farther than the points.
  EXPECT_NEAR(filled[360], 2.0, 0.01);
  // A cluster behind the robot across +-pi stays in one piece (no ray all the way around).
  Cluster behind;
  behind.points = {{-2.0, -0.1}, {-2.0, 0.1}};
  FinalizeCluster(behind);
  EXPECT_LE(returns(BuildScan(std::vector<Cluster>{behind}, 720, 0.35, 6.0)), 15);   // 5.7 deg, not 354
}
