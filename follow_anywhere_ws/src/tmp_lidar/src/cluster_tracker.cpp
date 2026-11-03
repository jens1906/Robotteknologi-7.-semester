#include "tmp_lidar/cluster_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

#include "tmp_lidar/assignment.hpp"

namespace tmp_lidar
{

namespace
{

constexpr int kBearingBins = 360;
constexpr double kForbidden = 1e9;

int bearingBin(const Point2 & p)
{
  const double a = std::atan2(p.y, p.x);   // -pi..pi
  const int bin = static_cast<int>(std::floor((a + M_PI) / (2.0 * M_PI) * kBearingBins));
  return std::clamp(bin, 0, kBearingBins - 1);
}

double bayesSeen(double r, double pd, double pfa)
{
  return r * pd / (r * pd + (1.0 - r) * pfa);
}

double bayesMissed(double r, double pd, double pfa)
{
  const double num = r * (1.0 - pd);
  const double den = num + (1.0 - r) * (1.0 - pfa);
  return den > 0.0 ? num / den : r;
}

double nearestRange(const std::vector<Point2> & basePoints)
{
  double best = std::numeric_limits<double>::infinity();
  for (const auto & p : basePoints) {best = std::min(best, norm(p));}
  return best;
}

}  // namespace

double ClusterTracker::detectionProbability(double range) const
{
  if (!(range >= params_.rangeMin && range <= params_.rangeMax)) {return 0.0;}
  const double span = params_.detectionFarRange - params_.detectionNearRange;
  const double share = span > 0.0 ? std::clamp((range - params_.detectionNearRange) / span, 0.0, 1.0) :
    (range > params_.detectionNearRange ? 1.0 : 0.0);
  return params_.detectionProbabilityNear +
         share * (params_.detectionProbabilityFar - params_.detectionProbabilityNear);
}

double ClusterTracker::measurementVariance(const Cluster & c) const
{
  double sigma = params_.lidarRangeStd[0] + params_.lidarRangeStd[1] * norm(c.closest) +
    params_.shapeSigmaScale * c.radius;
  if (c.freshPoints == 0) {sigma += params_.staleCentroidSigma;}
  return sigma * sigma;
}

double ClusterTracker::falseAlarm(const Cluster & c) const
{
  const double n = std::max<double>(1.0, static_cast<double>(c.points.size()));
  const double scale = std::max(1.0, params_.falseAlarmPoints / n);
  return std::clamp(params_.falseAlarmProbability * scale, 1e-6, 0.5);
}

Point2 ClusterTracker::coastShift(const Track & track) const
{
  // Move the last points as the published velocity says: a standing object stays where it was seen.
  if (track.isChunk || (track.motion.x == 0.0 && track.motion.y == 0.0)) {return {0.0, 0.0};}
  const Point2 now = track.filter.position();
  return {now.x - track.positionAtSighting.x, now.y - track.positionAtSighting.y};
}

void ClusterTracker::mergeByTracks(std::vector<Cluster> & clusters, const Pose2 & robot) const
{
  if (params_.mergeRadius <= 0.0) {return;}
  // Each cluster -> the confirmed track it fits best (inside the gate).
  std::map<std::size_t, std::vector<std::size_t>> byTrack;
  for (std::size_t ci = 0; ci < clusters.size(); ++ci) {
    const Cluster & c = clusters[ci];
    if (c.isChunk) {continue;}
    const Point2 z = robot.toOdom(c.measuredCentroid);
    const double variance = measurementVariance(c);
    double best = std::numeric_limits<double>::infinity();
    std::size_t bestTrack = tracks_.size();
    for (std::size_t ti = 0; ti < tracks_.size(); ++ti) {
      const Track & track = tracks_[ti];
      if (!track.confirmed || track.isChunk) {continue;}
      const double euclid = distance(z, track.filter.position());
      if (euclid > params_.maximumGate) {continue;}
      const double d2 = track.filter.mahalanobis2(z, variance * track.noiseScale);
      if (d2 > params_.gateChi2 && euclid > params_.minimumGate) {continue;}
      if (d2 < best) {
        best = d2;
        bestTrack = ti;
      }
    }
    if (bestTrack < tracks_.size()) {byTrack[bestTrack].push_back(ci);}
  }

  std::vector<bool> drop(clusters.size(), false);
  for (const auto & [ti, members] : byTrack) {
    (void)ti;
    if (members.size() < 2) {continue;}
    Cluster merged;
    Point2 fresh{0.0, 0.0};
    int freshCount = 0;
    bool allFresh = true;
    for (auto ci : members) {
      const Cluster & c = clusters[ci];
      merged.points.insert(merged.points.end(), c.points.begin(), c.points.end());
      if (c.freshPoints > 0) {
        fresh.x += c.measuredCentroid.x * c.freshPoints;
        fresh.y += c.measuredCentroid.y * c.freshPoints;
        freshCount += c.freshPoints;
      } else {
        allFresh = false;
      }
    }
    finalizeCluster(merged);
    if (merged.radius > params_.mergeRadius) {continue;}
    if (allFresh && freshCount > 0) {
      merged.measuredCentroid = {fresh.x / freshCount, fresh.y / freshCount};
      merged.freshPoints = freshCount;
    }
    clusters[members.front()] = std::move(merged);
    for (std::size_t k = 1; k < members.size(); ++k) {drop[members[k]] = true;}
  }
  std::size_t keep = 0;
  for (std::size_t ci = 0; ci < clusters.size(); ++ci) {
    if (!drop[ci]) {
      if (keep != ci) {clusters[keep] = std::move(clusters[ci]);}
      ++keep;
    }
  }
  clusters.resize(keep);
}

bool ClusterTracker::isOccluded(const Track & track, const std::vector<double> & nearest, const Pose2 & robot) const
{
  if (track.points.empty()) {return false;}
  const Point2 shift = coastShift(track);
  int covered = 0;
  for (const auto & p : track.points) {
    const Point2 b = robot.toBase({p.x + shift.x, p.y + shift.y});
    if (nearest[static_cast<std::size_t>(bearingBin(b))] < norm(b) - params_.occlusionMargin) {++covered;}
  }
  return 2 * covered >= static_cast<int>(track.points.size());
}

void ClusterTracker::update(std::vector<Cluster> & clusters, const Pose2 & robot, bool haveOdometry, double t)
{
  if (t < lastTime_) {tracks_.clear();}   // time jumped back (e.g. a looped rosbag)
  lastTime_ = t;

  // --- predict every track to t
  for (auto & track : tracks_) {
    track.filter.predict(t - track.lastPredicted);
    track.lastPredicted = t;
  }

  // --- join pieces of one object
  mergeByTracks(clusters, robot);

  std::vector<Point2> measured;
  std::vector<double> variances;
  measured.reserve(clusters.size());
  for (const auto & c : clusters) {
    measured.push_back(robot.toOdom(c.measuredCentroid));
    variances.push_back(measurementVariance(c));
  }

  // --- associate
  std::vector<int> trackOfCluster(clusters.size(), -1);
  std::vector<bool> trackSeen(tracks_.size(), false);
  // wall pieces first: the track of the same odom cell
  std::map<std::int64_t, std::size_t> chunkTracks;
  for (std::size_t ti = 0; ti < tracks_.size(); ++ti) {
    if (tracks_[ti].isChunk) {chunkTracks[tracks_[ti].chunkKey] = ti;}
  }
  for (std::size_t ci = 0; ci < clusters.size(); ++ci) {
    if (!clusters[ci].isChunk) {continue;}
    const auto it = chunkTracks.find(clusters[ci].chunkKey);
    if (it != chunkTracks.end() && !trackSeen[it->second]) {
      trackOfCluster[ci] = static_cast<int>(it->second);
      trackSeen[it->second] = true;
    }
  }
  // everything else, wall pieces or not (a wall piece whose cell changed, a cluster that grew into a wall or
  // shrank out of one): minimum total cost inside the gates
  std::vector<std::size_t> freeClusters, freeTracks;
  for (std::size_t ci = 0; ci < clusters.size(); ++ci) {
    if (trackOfCluster[ci] < 0) {freeClusters.push_back(ci);}
  }
  for (std::size_t ti = 0; ti < tracks_.size(); ++ti) {
    if (!trackSeen[ti]) {freeTracks.push_back(ti);}
  }
  if (!freeClusters.empty() && !freeTracks.empty()) {
    std::vector<std::vector<double>> cost(freeClusters.size(), std::vector<double>(freeTracks.size(), kForbidden));
    for (std::size_t a = 0; a < freeClusters.size(); ++a) {
      const std::size_t ci = freeClusters[a];
      for (std::size_t b = 0; b < freeTracks.size(); ++b) {
        const Track & track = tracks_[freeTracks[b]];
        const double euclid = distance(measured[ci], track.filter.position());
        if (euclid > params_.maximumGate) {continue;}
        double logDet = 0.0;
        const double d2 = track.filter.mahalanobis2(measured[ci], variances[ci] * track.noiseScale, &logDet);
        if (d2 > params_.gateChi2 && euclid > params_.minimumGate) {continue;}
        cost[a][b] = std::min(d2, params_.gateChi2) + logDet;
      }
    }
    // logDet may be negative: shift all allowed costs to be positive.
    double lowest = 0.0;
    for (const auto & row : cost) {
      for (double v : row) {
        if (v < kForbidden) {lowest = std::min(lowest, v);}
      }
    }
    for (auto & row : cost) {
      for (double & v : row) {
        if (v < kForbidden) {v -= lowest;}
      }
    }
    const auto assigned = solveAssignment(cost, kForbidden);
    for (std::size_t a = 0; a < freeClusters.size(); ++a) {
      if (assigned[a] < 0) {continue;}
      const std::size_t ti = freeTracks[static_cast<std::size_t>(assigned[a])];
      trackOfCluster[freeClusters[a]] = static_cast<int>(ti);
      trackSeen[ti] = true;
    }
  }

  // --- missed tracks: existence down, by how likely the lidar was to see them
  std::vector<double> nearest(kBearingBins, std::numeric_limits<double>::infinity());
  for (const auto & c : clusters) {
    for (const auto & p : c.points) {
      auto & bin = nearest[static_cast<std::size_t>(bearingBin(p))];
      bin = std::min(bin, norm(p));
    }
  }
  for (std::size_t ti = 0; ti < tracks_.size(); ++ti) {
    if (trackSeen[ti]) {continue;}
    Track & track = tracks_[ti];
    const Point2 shift = coastShift(track);
    std::vector<Point2> now;
    now.reserve(track.points.size());
    for (const auto & p : track.points) {now.push_back(robot.toBase({p.x + shift.x, p.y + shift.y}));}
    double pd = detectionProbability(nearestRange(now));
    track.occluded = isOccluded(track, nearest, robot);
    if (track.occluded) {pd *= params_.occludedDetectionFactor;}
    // A miss only counts once the track was missed in every cloud of a window of missWindow clouds.
    if (++track.missStreak % std::max(1, params_.missWindow) == 0) {
      track.existence = bayesMissed(track.existence, pd, params_.falseAlarmProbability);
    }
  }

  // --- seen tracks and new tracks
  for (std::size_t ci = 0; ci < clusters.size(); ++ci) {
    Cluster & c = clusters[ci];
    const double pfa = falseAlarm(c);
    Track * track = nullptr;
    if (trackOfCluster[ci] >= 0) {
      track = &tracks_[static_cast<std::size_t>(trackOfCluster[ci])];
      const double variance = variances[ci] * track->noiseScale;
      const double nis = track->filter.mahalanobis2(measured[ci], variance);
      track->filter.update(measured[ci], variance);
      if (track->sightings >= 2) {   // the first innovation includes the unknown velocity
        const double a = params_.noiseAdaptRate;
        track->noiseScale = std::clamp(track->noiseScale * (1.0 - a + a * std::min(nis, 50.0) / 2.0), 1.0,
            params_.maxNoiseScale);
      }
      const double pd = std::max(1e-3, detectionProbability(norm(c.closest)));
      track->existence = bayesSeen(track->existence, pd, pfa);
      ++track->sightings;
      track->missStreak = 0;
    } else {
      Track fresh;
      fresh.id = nextId_++;
      fresh.isChunk = c.isChunk;
      fresh.chunkKey = c.chunkKey;
      fresh.filter.init(measured[ci], variances[ci], params_.unknownVelocitySigma, params_.imm);
      fresh.existence = params_.birthExistence;
      fresh.sightings = 1;
      fresh.lastPredicted = t;
      tracks_.push_back(std::move(fresh));
      trackSeen.push_back(true);
      track = &tracks_.back();
    }
    track->existence = std::min(track->existence, params_.maxExistence);
    track->isChunk = c.isChunk;
    track->chunkKey = c.chunkKey;
    track->lastSeen = t;
    track->occluded = false;
    track->points.clear();
    for (const auto & p : c.points) {track->points.push_back(robot.toOdom(p));}
    track->positionAtSighting = track->filter.position();
    track->sigmaAtSighting = track->isChunk ? 0.0 : track->filter.positionSigma();

    c.trackId = track->id;
    if (c.isChunk) {
      track->motion = {0.0, 0.0};
      track->motionMode = ImmFilter::kStatic;
      c.velocitySigma = params_.staticVelocitySigma;
    } else {
      track->motionMode = track->filter.likeliestMode();
      if (!haveOdometry || track->sightings < std::max(2, params_.velocitySightings)) {
        track->motion = {0.0, 0.0};
        c.velocitySigma = params_.unknownVelocitySigma;
      } else {
        const double sigma = std::max(params_.minimumVelocitySigma, track->filter.movingVelocitySigma());
        const Point2 v = track->filter.movingVelocity();
        const bool standing =
          track->filter.modeProbability(ImmFilter::kStatic) > params_.staticPublishProbability ||
          norm(v) < std::max(params_.minimumSpeed, 2.0 * sigma);
        track->motion = standing ? Point2{0.0, 0.0} : v;
        c.velocitySigma = sigma;
      }
    }
    c.velocity = robot.rotateToBase(track->motion);
    track->publishedSigma = c.velocitySigma;
  }

  // --- confirm with hysteresis, drop the dead
  for (auto & track : tracks_) {
    if (track.existence >= params_.confirmExistence) {track.confirmed = true;}
    if (track.existence < params_.keepExistence) {track.confirmed = false;}
  }
  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [&](const Track & track) {
      const double timeout = track.occluded ? params_.occludedTimeout : params_.trackTimeout;
      return track.existence < params_.deleteExistence || t - track.lastSeen > timeout ||
      track.filter.positionSigma() > params_.maxPositionSigma;
    }), tracks_.end());

  for (auto & c : clusters) {
    for (const auto & track : tracks_) {
      if (track.id == c.trackId) {
        c.existence = track.existence;
        c.motionMode = track.motionMode;
        break;
      }
    }
  }
}

std::vector<Cluster> ClusterTracker::publishable(const std::vector<Cluster> & clusters, const Pose2 & robot,
  double t) const
{
  std::vector<Cluster> out;
  out.reserve(clusters.size());
  for (const auto & track : tracks_) {
    if (!track.confirmed) {continue;}
    if (t - track.lastSeen <= 1e-9) {
      for (const auto & c : clusters) {
        if (c.trackId == track.id) {
          out.push_back(c);
          break;
        }
      }
      continue;
    }
    // Missed in this cloud: last points, moved by the prediction. The filter was predicted to the time of the
    // last update(); a later t (publishing between clouds) is not predicted further.
    Cluster c;
    c.isChunk = track.isChunk;
    c.chunkKey = track.chunkKey;
    c.isCoasting = true;
    c.trackId = track.id;
    const Point2 shift = coastShift(track);
    for (const auto & p : track.points) {c.points.push_back(robot.toBase({p.x + shift.x, p.y + shift.y}));}
    finalizeCluster(c);
    c.velocity = robot.rotateToBase(track.motion);
    c.velocitySigma = track.publishedSigma;
    c.extraSigma = track.isChunk ? 0.0 : std::max(0.0, track.filter.positionSigma() - track.sigmaAtSighting);
    c.existence = track.existence;
    c.motionMode = track.motionMode;
    out.push_back(std::move(c));
  }
  return out;
}

}  // namespace tmp_lidar
