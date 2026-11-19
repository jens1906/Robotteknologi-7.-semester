// Micro-benchmark of SearchFilter::step() at the spec defaults: p50 / p99 / max over many ticks.
// Scene: corridor with a side room, 720-beam scan, an unmatched camera frame every tick, the robot
// driving (window shifts) and turning. Run: ros2 run leader_search bench_step [ticks]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "leader_search/search_filter.hpp"

using namespace leader_search;

namespace {
struct Seg {
  float x0, y0, x1, y1;
};
void make_scan(const Pose2& robot, const std::vector<Seg>& walls, std::vector<float>* ranges) {
  const int n = static_cast<int>(ranges->size());
  for (int i = 0; i < n; ++i) {
    const float a = robot.yaw - 3.14159265f + i * 6.2831853f / n;
    const float dx = std::cos(a), dy = std::sin(a);
    float best = 12.0f;
    for (const Seg& s : walls) {
      const float ex = s.x1 - s.x0, ey = s.y1 - s.y0, den = dx * ey - dy * ex;
      if (std::fabs(den) < 1e-9f) continue;
      const float wx = s.x0 - robot.x, wy = s.y0 - robot.y;
      const float t = (wx * ey - wy * ex) / den, u = (wx * dy - wy * dx) / den;
      if (t > 0 && u >= 0 && u <= 1 && t < best) best = t;
    }
    (*ranges)[i] = best;
  }
}
}  // namespace

int main(int argc, char** argv) {
  const int ticks = argc > 1 ? std::atoi(argv[1]) : 5000;
  Params p;
  p.t_max = 1e9f;  // stay in SEARCH
  p.p_min = 0.0f;
  p.p_out_max = 1.1f;
  SearchFilter f;
  if (!f.init(p)) return 1;
  f.enable_stage_timing(true);
  // Per stage, per time-since-loss bucket: sum and samples (for mean) and max.
  double st_sum[3][kNumStages] = {}, st_max[3][kNumStages] = {};
  long st_n[3] = {};
  const std::vector<Seg> walls = {{-20, -1.5f, 40, -1.5f}, {-20, 1.5f, 3, 1.5f}, {4, 1.5f, 40, 1.5f},
                                  {3, 1.5f, 3, 5}, {4, 1.5f, 4, 5}};
  std::vector<float> ranges(720);
  ScanView scan{ranges.data(), 720, -3.14159265f, 6.2831853f / 720, 0.05f, 12.0f, Pose2{}};
  Observation miss;
  std::vector<double> us, bucket[3];  // all, and by time since loss: 0-5 s, 5-10 s, 10-20 s
  us.reserve(ticks);
  for (auto& b : bucket) b.reserve(ticks);
  Pose2 robot;
  TrackInit t;
  t.p = {2.0f, 0.0f};
  t.v = {1.2f, 0.0f};
  for (int i = 0; i < ticks; ++i) {
    if (i % 200 == 0) {  // re-seed every 20 s so the belief stays busy
      robot = Pose2{};
      make_scan(robot, walls, &ranges);
      f.on_loss(t, robot, &scan);
    }
    robot.x += 0.04f;
    robot.yaw = 0.3f * std::sin(0.05f * i);
    miss.robot = robot;
    make_scan(robot, walls, &ranges);
    const auto t0 = std::chrono::steady_clock::now();
    const SearchOutput out = f.step(robot, &scan, &miss, 1, 0.1f);
    const auto t1 = std::chrono::steady_clock::now();
    const double dt_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    us.push_back(dt_us);
    const int bi = out.time_since_loss < 5.0f ? 0 : out.time_since_loss < 10.0f ? 1 : 2;
    bucket[bi].push_back(dt_us);
    ++st_n[bi];
    for (int s = 0; s < kNumStages; ++s) {
      st_sum[bi][s] += f.stage_us()[s];
      st_max[bi][s] = std::max(st_max[bi][s], f.stage_us()[s]);
    }
  }
  std::sort(us.begin(), us.end());
  auto q = [&](double f) { return us[static_cast<size_t>(f * (us.size() - 1))]; };
  const char* names[3] = {" 0-5 s", " 5-10 s", "10-20 s"};
  for (int i = 0; i < 3; ++i) {
    auto& b = bucket[i];
    if (b.empty()) continue;
    std::sort(b.begin(), b.end());
    std::printf("  since loss %s: p50 %7.1f us  p99 %7.1f us  (%zu ticks)\n", names[i],
                b[b.size() / 2], b[static_cast<size_t>(0.99 * (b.size() - 1))], b.size());
  }
  std::printf("\nmean (max) us per stage    %16s %16s %16s\n", "0-5 s", "5-10 s", "10-20 s");
  for (int s = 0; s < kNumStages; ++s) {
    std::printf("  %-22s", SearchFilter::stage_name(s));
    for (int i = 0; i < 3; ++i)
      std::printf("  %6.1f (%6.1f)", st_n[i] ? st_sum[i][s] / st_n[i] : 0.0, st_max[i][s]);
    std::printf("\n");
  }
  std::printf("step() over %d ticks (W=%d, r=%.2f, 720 beams): p50 %.1f us  p99 %.1f us  max %.1f us\n",
              ticks, f.window_cells(), f.resolution(), q(0.5), q(0.99), us.back());
  std::printf("budget: p99 <= 1000 us on one Jetson Nano core (spec), %s here\n",
              q(0.99) <= 1000.0 ? "met" : "NOT met");
  return 0;
}
