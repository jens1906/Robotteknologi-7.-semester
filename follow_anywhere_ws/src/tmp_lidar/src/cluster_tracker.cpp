#include "tmp_lidar/cluster_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

namespace tmp_lidar
{

void ClusterTracker::update(std::vector<Cluster> & clusters, const Pose2 & robot, bool haveOdometry, double t)
{
  haveOdometry_ = haveOdometry;
  // Forget old tracks (and everything if time jumped back, e.g. a looped rosbag).
  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [&](const Track & track) {
      return t - track.lastSeen > params_.trackTimeout || t < track.lastSeen;
    }), tracks_.end());

  std::vector<Point2> positions;
  positions.reserve(clusters.size());
  for (const auto & c : clusters) {positions.push_back(robot.toOdom(c.centroid));}

  // Greedy nearest matching inside the gate. A coasting track is predicted forward by its motion.
  std::vector<std::tuple<double, std::size_t, std::size_t>> pairs;
  for (std::size_t ci = 0; ci < clusters.size(); ++ci) {
    for (std::size_t ti = 0; ti < tracks_.size(); ++ti) {
      const Track & track = tracks_[ti];
      const double dt = t - track.lastSeen;
      const Point2 predicted{track.position.x + track.motion.x * dt, track.position.y + track.motion.y * dt};
      const double d = distance(positions[ci], predicted);
      if (d <= params_.trackGate) {pairs.emplace_back(d, ci, ti);}
    }
  }
  std::sort(pairs.begin(), pairs.end());
  std::vector<int> trackOfCluster(clusters.size(), -1);
  std::vector<bool> trackUsed(tracks_.size(), false);
  for (const auto & [d, ci, ti] : pairs) {
    (void)d;
    if (trackOfCluster[ci] >= 0 || trackUsed[ti]) {continue;}
    trackOfCluster[ci] = static_cast<int>(ti);
    trackUsed[ti] = true;
  }

  const double unknownVariance = params_.unknownVelocitySigma * params_.unknownVelocitySigma;
  for (std::size_t ci = 0; ci < clusters.size(); ++ci) {
    Track * track = nullptr;
    if (trackOfCluster[ci] >= 0) {
      track = &tracks_[static_cast<std::size_t>(trackOfCluster[ci])];
      const double dt = t - track->lastSeen;
      if (dt > 1e-3) {
        const Point2 raw{(positions[ci].x - track->position.x) / dt, (positions[ci].y - track->position.y) / dt};
        if (track->sightings < 2) {
          track->velocity = raw;
        } else {
          const double ex = raw.x - track->velocity.x, ey = raw.y - track->velocity.y;
          track->velocity.x += params_.velocityAlpha * ex;
          track->velocity.y += params_.velocityAlpha * ey;
          const double residual = 0.5 * (ex * ex + ey * ey);
          track->variance += params_.velocityAlpha * (residual - track->variance);
        }
        ++track->sightings;
      }
    } else {
      Track fresh;
      fresh.id = nextId_++;
      fresh.position = positions[ci];
      fresh.variance = unknownVariance;
      fresh.lastSeen = t;
      fresh.sightings = 1;
      tracks_.push_back(fresh);
      track = &tracks_.back();
    }
    track->position = positions[ci];
    track->lastSeen = t;

    Cluster & c = clusters[ci];
    track->isChunk = c.isChunk;
    track->points.clear();
    for (const auto & p : c.points) {track->points.push_back(robot.toOdom(p));}

    c.trackId = track->id;
    if (c.isChunk) {
      track->motion = {0.0, 0.0};
      c.velocitySigma = params_.staticVelocitySigma;
    } else if (!haveOdometry || track->sightings < 2) {
      track->motion = {0.0, 0.0};
      c.velocitySigma = params_.unknownVelocitySigma;
    } else {
      // variance is the spread of the raw differences; a first-order filter with weight a passes
      // a / (2 - a) of white noise variance.
      const double a = params_.velocityAlpha;
      const double sigma = std::max(params_.minimumVelocitySigma, std::sqrt(track->variance * a / (2.0 - a)));
      const bool standing = norm(track->velocity) < std::max(params_.minimumSpeed, 2.0 * sigma);
      track->motion = standing ? Point2{0.0, 0.0} : track->velocity;
      c.velocitySigma = sigma;
    }
    c.velocity = robot.rotateToBase(track->motion);
    track->publishedSigma = c.velocitySigma;
  }
}

double ClusterTracker::coastLimit(double range) const
{
  const double share = params_.coastFarRange > 0.0 ? std::clamp(range / params_.coastFarRange, 0.0, 1.0) : 1.0;
  return params_.coastTime + share * std::max(0.0, params_.coastTimeFar - params_.coastTime);
}

namespace
{

template<typename Track>
double rangeOf(const Track & track, const Pose2 & robot)
{
  double best = std::numeric_limits<double>::infinity();
  for (const auto & p : track.points) {best = std::min(best, norm(robot.toBase(p)));}
  return best;
}

}  // namespace

std::vector<Cluster> ClusterTracker::publishable(const std::vector<Cluster> & clusters, const Pose2 & robot,
  double t) const
{
  std::vector<Cluster> out;
  out.reserve(clusters.size());
  for (const auto & track : tracks_) {
    if (track.sightings < params_.confirmHits) {continue;}
    const double dt = t - track.lastSeen;
    if (dt <= 1e-9) {
      for (const auto & c : clusters) {
        if (c.trackId == track.id) {
          out.push_back(c);
          break;
        }
      }
      continue;
    }
    if (dt > coastLimit(rangeOf(track, robot)) + 1e-9) {continue;}
    Cluster c;
    c.isChunk = track.isChunk;
    c.isCoasting = true;
    c.trackId = track.id;
    for (const auto & p : track.points) {
      c.points.push_back(robot.toBase({p.x + track.motion.x * dt, p.y + track.motion.y * dt}));
    }
    finalizeCluster(c);
    c.velocity = robot.rotateToBase(track.motion);
    c.velocitySigma = track.publishedSigma;
    c.extraSigma = params_.coastSigmaRate * dt;
    out.push_back(std::move(c));
  }
  return out;
}

}  // namespace tmp_lidar
