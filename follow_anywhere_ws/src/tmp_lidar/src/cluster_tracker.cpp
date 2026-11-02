#include "tmp_lidar/cluster_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>

namespace tmp_lidar
{

void ClusterTracker::update(std::vector<Cluster> & clusters, const Pose2 & robot, bool haveOdometry, double t)
{
  // Forget old tracks (and everything if time jumped back, e.g. a looped rosbag).
  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [&](const Track & track) {
      return t - track.lastSeen > params_.trackTimeout || t < track.lastSeen;
    }), tracks_.end());

  std::vector<Point2> positions;
  positions.reserve(clusters.size());
  for (const auto & c : clusters) {positions.push_back(robot.toOdom(c.centroid));}

  // Greedy nearest matching inside the gate.
  std::vector<std::tuple<double, std::size_t, std::size_t>> pairs;
  for (std::size_t ci = 0; ci < clusters.size(); ++ci) {
    for (std::size_t ti = 0; ti < tracks_.size(); ++ti) {
      const double d = distance(positions[ci], tracks_[ti].position);
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
      tracks_.push_back(Track{nextId_++, positions[ci], {0.0, 0.0}, unknownVariance, t, 1});
      track = &tracks_.back();
    }
    track->position = positions[ci];
    track->lastSeen = t;

    Cluster & c = clusters[ci];
    c.trackId = track->id;
    if (c.isChunk) {
      c.velocity = {0.0, 0.0};
      c.velocitySigma = params_.staticVelocitySigma;
    } else if (!haveOdometry || track->sightings < 2) {
      c.velocity = {0.0, 0.0};
      c.velocitySigma = params_.unknownVelocitySigma;
    } else {
      // variance is the spread of the raw differences; a first-order filter with weight a passes
      // a / (2 - a) of white noise variance.
      const double a = params_.velocityAlpha;
      const double sigma = std::max(params_.minimumVelocitySigma, std::sqrt(track->variance * a / (2.0 - a)));
      const Point2 v = robot.rotateToBase(track->velocity);
      const bool standing = norm(v) < std::max(params_.minimumSpeed, 2.0 * sigma);
      c.velocity = standing ? Point2{0.0, 0.0} : v;
      c.velocitySigma = sigma;
    }
  }
}

}  // namespace tmp_lidar
