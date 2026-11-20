// Outputs from the tracked clusters. No ROS. See fusion.hpp.
#include "lidar_obstacles/fusion.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

namespace lidar_obstacles
{

// ===== Fusion ======================================================================================================
// Camera + lidar fusion. See fusion.hpp for the steps.

// Helpers only used in this file.
namespace
{

// Matching values
constexpr int LEADER_PERSON = -1;   // person index of the leader in the matching

// One camera person to be matched: the leader or one camera obstacle, with its closest point and how
// uncertain that point is.
struct Person
{
  int index;            // LEADER_PERSON or camera obstacle index
  Point2 point;
  double sigma;
};

// A possible match between a person and a cluster, with its cost (distance, lower = better).
struct PersonClusterPair
{
  double cost;
  std::size_t personIndex;
  std::size_t clusterIndex;
};

// A sigma is usable when it is a real, positive number (NaN or 0 = unknown).
bool IsSigmaFinite(double sigma) {return std::isfinite(sigma) && sigma > 0.0;}

}  // namespace

// Combines two velocity estimates, giving each a weight of 1 / sigma^2: the more certain one counts more,
// and the result is more certain than either. When one of them is unknown, the other is used as it is.
void BlendVelocity(const Point2 & firstVelocity, double firstSigma, const Point2 & secondVelocity,
  double secondSigma, Point2 & blendedVelocity, double & blendedSigma)
{
  if (!IsSigmaFinite(secondSigma)) {
    blendedVelocity = firstVelocity;
    blendedSigma = firstSigma;
    return;
  }
  if (!IsSigmaFinite(firstSigma)) {
    blendedVelocity = secondVelocity;
    blendedSigma = secondSigma;
    return;
  }
  const double firstWeight = 1.0 / (firstSigma * firstSigma);
  const double secondWeight = 1.0 / (secondSigma * secondSigma);
  const double totalWeight = firstWeight + secondWeight;
  blendedVelocity = {(firstVelocity.x * firstWeight + secondVelocity.x * secondWeight) / totalWeight,
    (firstVelocity.y * firstWeight + secondVelocity.y * secondWeight) / totalWeight};
  blendedSigma = 1.0 / std::sqrt(totalWeight);
}

// Position uncertainty of a cluster: that of a lidar point at the distance of its nearest point
// (a + b * range, never below minimumPositionUncertainty), plus the cluster's extraSigma (grows while coasting).
double Fusion::CalculateClusterSigma(const Cluster & cluster) const
{
  const double range = CalculateLength(cluster.closest);
  return std::max(params.minimumPositionUncertainty, params.lidarRangeStd[0] + params.lidarRangeStd[1] * range) +
         cluster.extraSigma;
}

// Forget the leader's cluster and last known point (no camera, or no leader any more).
void Fusion::Reset()
{
  leaderTrackId = -1;
  hasLeaderPoint = false;
  leaderPointSigma = NOT_A_NUMBER;
}

FusionOutput Fusion::Fuse(const FusionInput & input, const std::vector<Cluster> & clusters, const Pose2 & robot,
  double cloudSeconds)
{
  // Start: no cluster owned by anyone, the leader as the camera reported it.
  // Without camera data or without a leader, everything about the leader is forgotten.
  FusionOutput output;
  output.clusterOwner.assign(clusters.size(), OWNER_NONE);
  output.leader = input.leader;
  if (!input.hasCamera || input.leader.status == NO_LEADER) {
    Reset();
    output.leader = LeaderData{};
  }

  // --- 1. match camera people to clusters
  // The list of people: the leader (only while VISIBLE and with a known point), then every camera obstacle
  // with a known point.
  std::vector<Person> people;
  if (input.hasCamera) {
    if (output.leader.status == LEADER_VISIBLE && IsFinite(output.leader.closest)) {
      people.push_back({LEADER_PERSON, output.leader.closest, output.leader.positionSigma});
    }
    for (std::size_t obstacleIndex = 0; obstacleIndex < input.obstacles.size(); ++obstacleIndex) {
      const ObstacleData & obstacle = input.obstacles[obstacleIndex];
      if (IsFinite(obstacle.closest)) {
        people.push_back({static_cast<int>(obstacleIndex), obstacle.closest, obstacle.positionSigma});
      }
    }
  }

  // Every person/cluster pair within the gate. The gate is matchGate, or larger for an uncertain person
  // (matchSigmas * its sigma). The leader's cluster from last time gets stickyBonus off, so the leader does
  // not jump to a neighbouring cluster when two are about equally close.
  std::vector<PersonClusterPair> pairs;
  for (std::size_t personIndex = 0; personIndex < people.size(); ++personIndex) {
    const Person & person = people[personIndex];
    const double gate = std::max(params.matchGate,
      std::isfinite(person.sigma) ? params.matchSigmas * person.sigma : 0.0);
    for (std::size_t clusterIndex = 0; clusterIndex < clusters.size(); ++clusterIndex) {
      const double clusterDistance = CalculateDistanceToCluster(clusters[clusterIndex], person.point);
      if (clusterDistance > gate) {
        continue;
      }
      double cost = clusterDistance;
      if (person.index == LEADER_PERSON && leaderTrackId >= 0 && clusters[clusterIndex].trackId == leaderTrackId) {
        cost -= params.stickyBonus;
      }
      pairs.push_back({cost, personIndex, clusterIndex});
    }
  }
  // Greedy matching: go through the pairs from cheapest to most expensive and take each pair whose person and
  // cluster are both still free. (Ties are broken by index so the result is always the same.)
  std::sort(pairs.begin(), pairs.end(), [](const PersonClusterPair & first, const PersonClusterPair & second) {
      return std::tie(first.cost, first.personIndex, first.clusterIndex) <
             std::tie(second.cost, second.personIndex, second.clusterIndex);
    });
  std::vector<int> clusterOfPerson(people.size(), -1);
  for (const auto & pair : pairs) {
    if (clusterOfPerson[pair.personIndex] >= 0 || output.clusterOwner[pair.clusterIndex] != OWNER_NONE) {
      continue;
    }
    clusterOfPerson[pair.personIndex] = static_cast<int>(pair.clusterIndex);
    output.clusterOwner[pair.clusterIndex] =
      people[pair.personIndex].index == LEADER_PERSON ? OWNER_LEADER : people[pair.personIndex].index;
  }

  // --- 2a. leader
  // If the leader got a cluster: use the lidar's nearest point and accuracy, blend the velocities, and
  // remember the cluster's track id and the time (for the hold below).
  LeaderData & leader = output.leader;
  for (std::size_t personIndex = 0; personIndex < people.size(); ++personIndex) {
    if (people[personIndex].index != LEADER_PERSON || clusterOfPerson[personIndex] < 0) {
      continue;
    }
    const auto clusterIndex = static_cast<std::size_t>(clusterOfPerson[personIndex]);
    const Cluster & cluster = clusters[clusterIndex];
    leader.closest = cluster.closest;
    leader.positionSigma = CalculateClusterSigma(cluster);
    BlendVelocity(input.leader.velocity, input.leader.velocitySigma, cluster.velocity, cluster.velocitySigma,
      leader.velocity, leader.velocitySigma);
    leaderTrackId = cluster.trackId;
    leaderMatchSeconds = cloudSeconds;
  }

  if (leader.status == LEADER_VISIBLE) {
    // last_known_point = this frame's point (the camera's own when there was no depth this frame).
    // It is stored in odom so it stays on the same floor spot while the robot moves.
    const Point2 leaderPoint = IsFinite(leader.closest) ? leader.closest : leader.lastKnown;
    if (IsFinite(leaderPoint)) {
      leaderPointOdom = robot.ConvertToOdom(leaderPoint);
      leaderPointSigma = leader.positionSigma;
      hasLeaderPoint = true;
    }
  } else if (leader.status == LEADER_LOST) {
    // Hold: the leader's last cluster, while still tracked and not taken by a camera person.
    // The camera lost the leader, but the lidar may still track the same object: for up to leaderHoldTime the
    // leader's last point follows that cluster, and the cluster is not reported as an obstacle.
    if (leaderTrackId >= 0 && cloudSeconds - leaderMatchSeconds <= params.leaderHoldTime) {
      for (std::size_t clusterIndex = 0; clusterIndex < clusters.size(); ++clusterIndex) {
        const Cluster & cluster = clusters[clusterIndex];
        if (cluster.trackId != leaderTrackId || output.clusterOwner[clusterIndex] != OWNER_NONE) {
          continue;
        }
        output.clusterOwner[clusterIndex] = OWNER_LEADER;
        leaderPointOdom = robot.ConvertToOdom(cluster.closest);
        leaderPointSigma = CalculateClusterSigma(cluster);
        hasLeaderPoint = true;
        break;
      }
    }
    // While lost, the uncertainty grows with how long the leader has been lost (velocity sigma * time).
    if (hasLeaderPoint) {
      const double lostSeconds = std::isfinite(leader.lostDuration) ? leader.lostDuration : 0.0;
      const double velocitySigma = std::isfinite(leader.velocitySigma) ? leader.velocitySigma : 0.0;
      leader.positionSigma = leaderPointSigma + velocitySigma * lostSeconds;
    }
  }
  // Publish the stored last known point in the current base_link.
  if (hasLeaderPoint && leader.status != NO_LEADER) {
    leader.lastKnown = robot.ConvertToBase(leaderPointOdom);
  }

  // --- 2b + 3. obstacles
  // Line of sight: the line from the camera to the leader. For each obstacle, its point nearest that line
  // (NaN when there is no leader point).
  const Point2 sightEnd = leader.lastKnown;
  const bool hasSightLine = IsFinite(sightEnd);
  auto FindLineOfSightPoint = [&](const Cluster & cluster) {
      return hasSightLine ?
             FindClusterPointNearestSegment(cluster, params.cameraPosition, sightEnd) : MakeNotANumberPoint();
    };

  // Camera people first. A person with a cluster gets the lidar's position, accuracy and line of sight point
  // and a blended velocity; a person without one is passed through as the camera saw it.
  if (input.hasCamera) {
    // clusterOfObstacle[camera obstacle index] = its matched cluster (people[] skips obstacles without a
    // point, so the indices are translated here).
    std::vector<int> clusterOfObstacle(input.obstacles.size(), -1);
    for (std::size_t personIndex = 0; personIndex < people.size(); ++personIndex) {
      if (people[personIndex].index >= 0) {
        clusterOfObstacle[static_cast<std::size_t>(people[personIndex].index)] = clusterOfPerson[personIndex];
      }
    }
    for (std::size_t obstacleIndex = 0; obstacleIndex < input.obstacles.size(); ++obstacleIndex) {
      const ObstacleData & cameraObstacle = input.obstacles[obstacleIndex];
      FusedObstacle fusedObstacle;
      fusedObstacle.data = cameraObstacle;
      const int clusterIndex = clusterOfObstacle[obstacleIndex];
      if (clusterIndex >= 0) {
        const Cluster & cluster = clusters[static_cast<std::size_t>(clusterIndex)];
        fusedObstacle.data.closest = cluster.closest;
        fusedObstacle.data.positionSigma = CalculateClusterSigma(cluster);
        fusedObstacle.data.lineOfSight = FindLineOfSightPoint(cluster);
        BlendVelocity(cameraObstacle.velocity, cameraObstacle.velocitySigma, cluster.velocity, cluster.velocitySigma,
          fusedObstacle.data.velocity, fusedObstacle.data.velocitySigma);
      }
      output.obstacles.push_back(fusedObstacle);
    }
  }

  // Then every cluster nobody owns: a lidar-only obstacle (wall, box, a person the camera does not see).
  for (std::size_t clusterIndex = 0; clusterIndex < clusters.size(); ++clusterIndex) {
    if (output.clusterOwner[clusterIndex] != OWNER_NONE) {
      continue;
    }
    const Cluster & cluster = clusters[clusterIndex];
    FusedObstacle lidarObstacle;
    lidarObstacle.isLidarOnly = true;
    lidarObstacle.data.closest = cluster.closest;
    lidarObstacle.data.lineOfSight = FindLineOfSightPoint(cluster);
    lidarObstacle.data.velocity = cluster.velocity;
    lidarObstacle.data.positionSigma = CalculateClusterSigma(cluster);
    lidarObstacle.data.velocitySigma = cluster.velocitySigma;
    output.obstacles.push_back(lidarObstacle);
  }
  return output;
}

// ===== Scan builder ================================================================================================
// Laser scan from obstacle clusters. See the scan builder section of fusion.hpp.

// Helpers only used in this file.
namespace
{

// A cluster point in scan coordinates: rayPosition is the (fractional) ray index its bearing falls on,
// range its distance from the robot.
struct PolarPoint
{
  double rayPosition = 0.0;
  double range = 0.0;
};

}  // namespace

std::vector<float> BuildScan(const std::vector<Cluster> & clusters, int rayCount, double rangeMin, double rangeMax)
{
  // Start with every ray empty (+inf = nothing seen). rayStep is the angle between two rays.
  std::vector<float> ranges(static_cast<std::size_t>(std::max(rayCount, 1)), std::numeric_limits<float>::infinity());
  const auto rayTotal = static_cast<long>(ranges.size());
  const double rayStep = 2.0 * M_PI / static_cast<double>(rayTotal);
  // Writes a range into a ray, keeping the nearest one when several obstacles fall on the same ray.
  // The index is wrapped around (ray -1 = last ray, ray rayTotal = ray 0) so a cluster across +-pi works.
  auto PutRange = [&](long rayIndex, double range) {
      if (range < rangeMin || range > rangeMax) {
        return;
      }
      float & rayRange = ranges[static_cast<std::size_t>(((rayIndex % rayTotal) + rayTotal) % rayTotal)];
      rayRange = std::min(rayRange, static_cast<float>(range));
    };
  for (const auto & cluster : clusters) {
    if (cluster.points.empty()) {
      continue;
    }
    // Bearings unwrapped around the centroid's, so a cluster behind the robot (across +-pi) stays in one piece.
    // Without this, a point at +179 deg and one at -179 deg would look 358 deg apart and the interpolation
    // below would fill a ray all the way round the robot.
    const double centroidBearing = std::atan2(cluster.centroid.y, cluster.centroid.x);
    std::vector<PolarPoint> polarPoints;
    polarPoints.reserve(cluster.points.size());
    for (const auto & point : cluster.points) {
      const double bearing =
        centroidBearing + std::remainder(std::atan2(point.y, point.x) - centroidBearing, 2.0 * M_PI);
      polarPoints.push_back({(bearing + M_PI) / rayStep, CalculateLength(point)});
    }
    // Sort by bearing (then by range) so neighbouring points in the list are neighbours around the robot.
    std::sort(polarPoints.begin(), polarPoints.end(), [](const PolarPoint & first, const PolarPoint & second) {
        return std::tie(first.rayPosition, first.range) < std::tie(second.rayPosition, second.range);
      });
    for (std::size_t pointIndex = 0; pointIndex < polarPoints.size(); ++pointIndex) {
      // The point itself, on its nearest ray.
      PutRange(std::lround(polarPoints[pointIndex].rayPosition), polarPoints[pointIndex].range);
      if (pointIndex + 1 == polarPoints.size()) {
        continue;
      }
      // Every ray between this point and the next one gets a range interpolated in a straight line between
      // the two, so the object shows up as a solid surface instead of separate dots.
      const PolarPoint & startPoint = polarPoints[pointIndex];
      const PolarPoint & endPoint = polarPoints[pointIndex + 1];
      for (auto rayIndex = static_cast<long>(std::ceil(startPoint.rayPosition));
        rayIndex <= static_cast<long>(std::floor(endPoint.rayPosition)); ++rayIndex)
      {
        const double fraction = endPoint.rayPosition > startPoint.rayPosition ?
          (static_cast<double>(rayIndex) - startPoint.rayPosition) / (endPoint.rayPosition - startPoint.rayPosition) :
          0.0;
        PutRange(rayIndex, startPoint.range + fraction * (endPoint.range - startPoint.range));
      }
    }
  }
  return ranges;
}

}  // namespace lidar_obstacles
