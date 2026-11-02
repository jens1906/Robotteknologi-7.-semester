#include "tmp_lidar/fusion.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>

namespace tmp_lidar
{

namespace
{

constexpr int kLeaderPerson = -1;   // person index of the leader in the matching

bool finiteSigma(double s) {return std::isfinite(s) && s > 0.0;}

}  // namespace

void blendVelocity(const Point2 & a, double sigmaA, const Point2 & b, double sigmaB, Point2 & out,
  double & sigmaOut)
{
  if (!finiteSigma(sigmaB)) {
    out = a;
    sigmaOut = sigmaA;
    return;
  }
  if (!finiteSigma(sigmaA)) {
    out = b;
    sigmaOut = sigmaB;
    return;
  }
  const double wa = 1.0 / (sigmaA * sigmaA), wb = 1.0 / (sigmaB * sigmaB);
  out = {(a.x * wa + b.x * wb) / (wa + wb), (a.y * wa + b.y * wb) / (wa + wb)};
  sigmaOut = 1.0 / std::sqrt(wa + wb);
}

double Fusion::lidarSigma(double range) const
{
  return std::max(params_.minimumPositionUncertainty,
           params_.lidarRangeStd[0] + params_.lidarRangeStd[1] * range);
}

void Fusion::reset()
{
  leaderTrack_ = -1;
  haveLeaderPoint_ = false;
  leaderPointSigma_ = kNaN;
}

FusionOutput Fusion::fuse(const FusionInput & input, const std::vector<Cluster> & clusters, const Pose2 & robot,
  double t)
{
  FusionOutput out;
  out.clusterOwner.assign(clusters.size(), kOwnerNone);
  out.leader = input.leader;
  if (!input.haveCamera || input.leader.status == kNoLeader) {
    reset();
    out.leader = LeaderData{};
  }

  // --- 1. match camera people to clusters
  struct Person
  {
    int index;            // kLeaderPerson or camera obstacle index
    Point2 point;
    double sigma;
  };
  std::vector<Person> people;
  if (input.haveCamera) {
    if (out.leader.status == kVisible && isFinite(out.leader.closest)) {
      people.push_back({kLeaderPerson, out.leader.closest, out.leader.positionSigma});
    }
    for (std::size_t i = 0; i < input.obstacles.size(); ++i) {
      if (isFinite(input.obstacles[i].closest)) {
        people.push_back({static_cast<int>(i), input.obstacles[i].closest, input.obstacles[i].positionSigma});
      }
    }
  }

  std::vector<std::tuple<double, std::size_t, std::size_t>> pairs;   // cost, person, cluster
  for (std::size_t pi = 0; pi < people.size(); ++pi) {
    const auto & person = people[pi];
    const double gate = std::max(params_.matchGate,
        std::isfinite(person.sigma) ? params_.matchSigmas * person.sigma : 0.0);
    for (std::size_t ci = 0; ci < clusters.size(); ++ci) {
      const double d = distanceToCluster(clusters[ci], person.point);
      if (d > gate) {continue;}
      double cost = d;
      if (person.index == kLeaderPerson && leaderTrack_ >= 0 && clusters[ci].trackId == leaderTrack_) {
        cost -= params_.stickyBonus;
      }
      pairs.emplace_back(cost, pi, ci);
    }
  }
  std::sort(pairs.begin(), pairs.end());
  std::vector<int> clusterOfPerson(people.size(), -1);
  for (const auto & [cost, pi, ci] : pairs) {
    (void)cost;
    if (clusterOfPerson[pi] >= 0 || out.clusterOwner[ci] != kOwnerNone) {continue;}
    clusterOfPerson[pi] = static_cast<int>(ci);
    out.clusterOwner[ci] = people[pi].index == kLeaderPerson ? kOwnerLeader : people[pi].index;
  }

  // --- 2a. leader
  LeaderData & leader = out.leader;
  for (std::size_t pi = 0; pi < people.size(); ++pi) {
    if (people[pi].index != kLeaderPerson || clusterOfPerson[pi] < 0) {continue;}
    const auto ci = static_cast<std::size_t>(clusterOfPerson[pi]);
    const Cluster & c = clusters[ci];
    leader.closest = c.closest;
    leader.positionSigma = lidarSigma(norm(c.closest)) + c.extraSigma;
    blendVelocity(input.leader.velocity, input.leader.velocitySigma, c.velocity, c.velocitySigma,
      leader.velocity, leader.velocitySigma);
    out.leaderEnriched = true;
    out.leaderCluster = static_cast<int>(ci);
    leaderTrack_ = c.trackId;
    leaderMatchTime_ = t;
  }

  if (leader.status == kVisible) {
    // last_known_point = this frame's point (the camera's own when there was no depth this frame).
    const Point2 point = isFinite(leader.closest) ? leader.closest : leader.lastKnown;
    if (isFinite(point)) {
      leaderPointOdom_ = robot.toOdom(point);
      leaderPointSigma_ = leader.positionSigma;
      haveLeaderPoint_ = true;
    }
  } else if (leader.status == kLost) {
    // Hold: the leader's last cluster, while still tracked and not taken by a camera person.
    if (leaderTrack_ >= 0 && t - leaderMatchTime_ <= params_.leaderHoldTime) {
      for (std::size_t ci = 0; ci < clusters.size(); ++ci) {
        if (clusters[ci].trackId != leaderTrack_ || out.clusterOwner[ci] != kOwnerNone) {continue;}
        out.clusterOwner[ci] = kOwnerLeader;
        out.leaderCluster = static_cast<int>(ci);
        leaderPointOdom_ = robot.toOdom(clusters[ci].closest);
        leaderPointSigma_ = lidarSigma(norm(clusters[ci].closest)) + clusters[ci].extraSigma;
        haveLeaderPoint_ = true;
        break;
      }
    }
    if (haveLeaderPoint_) {
      const double lost = std::isfinite(leader.lostDuration) ? leader.lostDuration : 0.0;
      const double velocitySigma = std::isfinite(leader.velocitySigma) ? leader.velocitySigma : 0.0;
      leader.positionSigma = leaderPointSigma_ + velocitySigma * lost;
    }
  }
  if (haveLeaderPoint_ && leader.status != kNoLeader) {
    leader.lastKnown = robot.toBase(leaderPointOdom_);
  }

  // --- 2b + 3. obstacles
  const Point2 sightEnd = leader.lastKnown;
  const bool haveSight = isFinite(sightEnd);
  auto lineOfSight = [&](const Cluster & c) {
      return haveSight ? clusterPointNearestSegment(c, params_.cameraPosition, sightEnd) : nanPoint();
    };

  if (input.haveCamera) {
    std::vector<int> clusterOfObstacle(input.obstacles.size(), -1);
    for (std::size_t pi = 0; pi < people.size(); ++pi) {
      if (people[pi].index >= 0) {
        clusterOfObstacle[static_cast<std::size_t>(people[pi].index)] = clusterOfPerson[pi];
      }
    }
    for (std::size_t i = 0; i < input.obstacles.size(); ++i) {
      FusedObstacle fused;
      fused.data = input.obstacles[i];
      const int ci = clusterOfObstacle[i];
      if (ci >= 0) {
        const Cluster & c = clusters[static_cast<std::size_t>(ci)];
        fused.source = Source::CameraLidar;
        fused.cluster = ci;
        fused.data.closest = c.closest;
        fused.data.positionSigma = lidarSigma(norm(c.closest)) + c.extraSigma;
        fused.data.lineOfSight = lineOfSight(c);
        blendVelocity(input.obstacles[i].velocity, input.obstacles[i].velocitySigma, c.velocity, c.velocitySigma,
          fused.data.velocity, fused.data.velocitySigma);
      }
      out.obstacles.push_back(fused);
    }
  }

  for (std::size_t ci = 0; ci < clusters.size(); ++ci) {
    if (out.clusterOwner[ci] != kOwnerNone) {continue;}
    const Cluster & c = clusters[ci];
    FusedObstacle fused;
    fused.source = Source::Lidar;
    fused.cluster = static_cast<int>(ci);
    fused.data.closest = c.closest;
    fused.data.lineOfSight = lineOfSight(c);
    fused.data.velocity = c.velocity;
    fused.data.positionSigma = lidarSigma(norm(c.closest)) + c.extraSigma;
    fused.data.velocitySigma = c.velocitySigma;
    out.obstacles.push_back(fused);
  }
  return out;
}

}  // namespace tmp_lidar
