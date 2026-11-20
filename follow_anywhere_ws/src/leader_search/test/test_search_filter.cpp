// Unit tests for the search core: one TEST per bullet of the spec's "Unit tests" list, plus params.
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <numeric>
#include <random>
#include <vector>

#include "leader_search/search_filter.hpp"

using namespace leader_search;

// ------------------------------------------------------------------------------------------------
// Allocation counter: replaced global operator new/delete and malloc family (glibc).

namespace {
std::atomic<bool> g_counting{false};
std::atomic<long> g_allocs{0};
inline void note_alloc() {
  if (g_counting.load(std::memory_order_relaxed)) g_allocs.fetch_add(1, std::memory_order_relaxed);
}
}  // namespace

extern "C" {
void* __libc_malloc(size_t);
void* __libc_calloc(size_t, size_t);
void* __libc_realloc(void*, size_t);
void* __libc_memalign(size_t, size_t);
void __libc_free(void*);
void* malloc(size_t n) {
  note_alloc();
  return __libc_malloc(n);
}
void* calloc(size_t a, size_t b) {
  note_alloc();
  return __libc_calloc(a, b);
}
void* realloc(void* p, size_t n) {
  note_alloc();
  return __libc_realloc(p, n);
}
void free(void* p) { __libc_free(p); }
}

void* operator new(size_t n) {
  note_alloc();
  if (void* p = __libc_malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void* operator new[](size_t n) { return operator new(n); }
void* operator new(size_t n, const std::nothrow_t&) noexcept {
  note_alloc();
  return __libc_malloc(n ? n : 1);
}
void* operator new[](size_t n, const std::nothrow_t&) noexcept {
  return operator new(n, std::nothrow);
}
void* operator new(size_t n, std::align_val_t a) {
  note_alloc();
  if (void* p = __libc_memalign(static_cast<size_t>(a), n ? n : 1)) return p;
  throw std::bad_alloc();
}
void* operator new[](size_t n, std::align_val_t a) { return operator new(n, a); }
void operator delete(void* p) noexcept { __libc_free(p); }
void operator delete[](void* p) noexcept { __libc_free(p); }
void operator delete(void* p, size_t) noexcept { __libc_free(p); }
void operator delete[](void* p, size_t) noexcept { __libc_free(p); }
void operator delete(void* p, std::align_val_t) noexcept { __libc_free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { __libc_free(p); }
void operator delete(void* p, size_t, std::align_val_t) noexcept { __libc_free(p); }
void operator delete[](void* p, size_t, std::align_val_t) noexcept { __libc_free(p); }

// ------------------------------------------------------------------------------------------------
// Helpers

namespace {

constexpr float kPi = 3.14159265358979f;
float wrap_angle_for_test(float a) { return std::remainder(a, 2.0f * kPi); }

struct Seg {
  float x0, y0, x1, y1;
};

// Synthetic 360 degree LaserScan in base_link from wall segments (odom frame).
struct FakeScan {
  std::vector<float> ranges;
  ScanView view;
  void make(const Pose2& robot, const std::vector<Seg>& walls, int beams = 720, float rmax = 12) {
    ranges.assign(beams, std::numeric_limits<float>::infinity());
    const float inc = 2.0f * kPi / beams;
    for (int i = 0; i < beams; ++i) {
      const float a = robot.yaw - kPi + i * inc;
      const float dx = std::cos(a), dy = std::sin(a);
      for (const Seg& s : walls) {
        const float ex = s.x1 - s.x0, ey = s.y1 - s.y0;
        const float den = dx * ey - dy * ex;
        if (std::fabs(den) < 1e-9f) continue;
        const float wx = s.x0 - robot.x, wy = s.y0 - robot.y;
        const float t = (wx * ey - wy * ex) / den;
        const float u = (wx * dy - wy * dx) / den;
        if (t > 0 && u >= 0 && u <= 1 && t < ranges[i] && t < rmax) ranges[i] = t;
      }
    }
    view = ScanView{ranges.data(), beams, -kPi, inc, 0.05f, rmax, Pose2{}};
  }
};

// The tests use the spec's 20 x 20 square window (cell maths below relies on it).
Params test_params() {
  Params p;
  p.window_radius = 0.0f;
  p.tail_cut = 0.0f;  // exact model: conservation and drop accounting are checked to 1e-5
  return p;
}

Params quiet_params() {
  Params p = test_params();
  p.stopped_enabled = false;
  return p;
}

TrackInit track_at(float x, float y, float vx, float vy) {
  TrackInit t;
  t.p = {x, y};
  t.v = {vx, vy};
  return t;
}

double total_mass(const SearchFilter& f) {
  double s = f.out_of_window();
  const int N = f.num_cells();
  for (int j = 0; j < N * kHeadings; ++j) s += f.belief()[j];
  for (int c = 0; c < N; ++c) s += f.stopped()[c];
  return s;
}

// Mean position (odom) of the in-window belief.
Vec2 mean_pos(const SearchFilter& f) {
  std::vector<float> marg(f.num_cells());
  f.belief_marginal(marg.data());
  double sx = 0, sy = 0, sw = 0;
  for (int c = 0; c < f.num_cells(); ++c) {
    const Vec2 q = f.cell_centre(c);
    sx += marg[c] * q.x;
    sy += marg[c] * q.y;
    sw += marg[c];
  }
  return {float(sx / sw), float(sy / sw)};
}

// Occupancy window with the given window cells occupied, rest free.
std::vector<uint8_t> occ_with(const SearchFilter& f, const std::vector<std::pair<int, int>>& cells) {
  std::vector<uint8_t> occ(f.num_cells(), kFree);
  for (auto [x, y] : cells) occ[y * f.window_cells() + x] = kOccupied;
  return occ;
}

}  // namespace

// ------------------------------------------------------------------------------------------------
// Step 1: params, init, allocation budget

TEST(Params, YamlMatchesDefaults) {
  Params p;
  std::string err;
  ASSERT_TRUE(load_params(LEADER_SEARCH_CONFIG, &p, &err)) << err;
  const Params d;
  EXPECT_EQ(p.window_cells, d.window_cells);
  EXPECT_FLOAT_EQ(p.window_radius, 15.0f);
  EXPECT_FLOAT_EQ(p.resolution, d.resolution);
  EXPECT_FLOAT_EQ(p.sigma_h, d.sigma_h);
  EXPECT_FLOAT_EQ(p.cam_fov, d.cam_fov);
  EXPECT_EQ(p.blocked_mode, BlockedMode::DROP);
  EXPECT_EQ(p.K_f, d.K_f);
  for (int k = 0; k < kPdBins; ++k) EXPECT_FLOAT_EQ(p.p_d[k], d.p_d[k]);
}

TEST(Params, RejectsInvalid) {
  Params p;
  p.window_cells = 2;
  std::string err;
  EXPECT_FALSE(validate(p, &err));
  p = Params{};
  p.p_d[3] = 1.0f;
  EXPECT_FALSE(validate(p, &err));
  SearchFilter f;
  EXPECT_FALSE(f.init(p, &err));
}

TEST(Init, ArenaNearSpecBudget) {
  SearchFilter f;
  ASSERT_TRUE(f.init(test_params()));
  // Spec: about 125 KB at the defaults (+3.2 KB stopped max-product, + planner scratch).
  EXPECT_LT(f.arena_bytes(), 160u * 1024u);
  EXPECT_EQ(f.mode(), Mode::TRACKING);
}

TEST(Init, RoundWindowAtDefaults) {
  SearchFilter f;
  ASSERT_TRUE(f.init(Params{}));  // 15 m radius at 0.5 m: 60 x 60 cells
  EXPECT_EQ(f.window_cells(), 60);
  EXPECT_LT(f.arena_bytes(), 1500u * 1024u);
  EXPECT_GE(f.cell_of({14.0f, 0.25f}), 0);
  EXPECT_EQ(f.cell_of({11.0f, 11.0f}), -1);  // in the square, outside the circle
  EXPECT_GE(f.cell_of({10.0f, 10.0f}), 0);
  // Mass never enters the corners: it leaves to out_of_window at the circle.
  Params p;
  p.stopped_enabled = false;
  p.tail_cut = 0.0f;  // exact conservation below
  SearchFilter g;
  ASSERT_TRUE(g.init(p));
  std::vector<uint8_t> occ(g.num_cells(), kFree);
  g.debug_set_occ(occ.data());
  g.on_loss(track_at(9.0f, 9.0f, 1.5f, 1.5f), Pose2{}, nullptr);
  const int W = g.window_cells();
  for (int i = 0; i < 80; ++i) {
    g.debug_filter_tick(Pose2{}, nullptr, nullptr, 0.1f);
    ASSERT_NEAR(g.last_total(), 1.0, 1e-5);
  }
  for (int c = 0; c < g.num_cells(); ++c) {
    const float dx = (c % W - W / 2) * 0.5f, dy = (c / W - W / 2) * 0.5f;
    if (std::hypot(dx, dy) > 15.0f) {
      ASSERT_EQ(g.belief()[c * kHeadings + 1], 0.0f);
    }
  }
  EXPECT_GT(g.out_of_window(), 0.3f);
  // A loss point beyond the circle is clamped onto it.
  g.on_loss(track_at(20.0f, 0.0f, 1.0f, 0.0f), Pose2{}, nullptr);
  SearchOutput out = g.step(Pose2{}, nullptr, nullptr, 0, 0.1f);
  EXPECT_TRUE(out.mode == Mode::SEARCH || out.mode == Mode::GIVE_UP);
}

TEST(Alloc, ZeroAllocationsAcross1000Steps) {
  SearchFilter f;
  ASSERT_TRUE(f.init(test_params()));
  std::vector<Seg> walls = {{-3, -2, 6, -2}, {-3, 2, 6, 2}, {2, 0.5f, 2, 2}};
  FakeScan scan;
  scan.make(Pose2{}, walls);
  Observation match;
  match.matched = match.has_track = true;
  match.track = track_at(1.5f, 0, 1.0f, 0);
  Observation miss;
  f.step(Pose2{}, &scan.view, &match, 1, 0.1f);  // TRACKING with a track
  g_allocs = 0;
  g_counting = true;
  Pose2 robot;
  for (int i = 0; i < 1000; ++i) {
    robot.x += 0.03f;  // drives window shifts
    robot.yaw = 0.2f * std::sin(i * 0.05f);
    miss.robot = robot;
    const bool reacquire = i % 400 == 399;
    f.step(robot, &scan.view, reacquire ? &match : &miss, 1, 0.1f);
  }
  g_counting = false;
  EXPECT_EQ(g_allocs.load(), 0);
}

// ------------------------------------------------------------------------------------------------
// Step 2: prediction, DROP, out_of_window, window shift

TEST(Predict, ConservesMassWithoutWallsOrObservations) {
  SearchFilter f;
  ASSERT_TRUE(f.init(quiet_params()));
  f.on_loss(track_at(1.0f, 0.5f, 1.2f, 0.4f), Pose2{}, nullptr);
  auto occ = occ_with(f, {});
  f.debug_set_occ(occ.data());
  for (int i = 0; i < 200; ++i) {
    f.debug_filter_tick(Pose2{}, nullptr, nullptr, 0.1f);
    ASSERT_NEAR(f.last_total(), 1.0, 1e-5) << "tick " << i;
    ASSERT_EQ(f.dropped_mass(), 0.0f);
  }
  EXPECT_GT(f.out_of_window(), 0.5f);  // after 20 s most of it has left the 10 m window
  EXPECT_NEAR(total_mass(f), 1.0, 1e-5);
}

TEST(Predict, MeanMovesAtSpeedAlongHeading) {
  for (float theta : {0.0f, kPi / 2, kPi / 4, -3 * kPi / 4}) {
    Params p = quiet_params();
    p.sigma_h = 0.0f;
    SearchFilter f;
    ASSERT_TRUE(f.init(p));
    const float s = 1.0f;
    f.on_loss(track_at(0.25f, 0.25f, s * std::cos(theta), s * std::sin(theta)), Pose2{}, nullptr);
    auto occ = occ_with(f, {});
    f.debug_set_occ(occ.data());
    const Vec2 m0 = mean_pos(f);
    for (int i = 0; i < 20; ++i) f.debug_filter_tick(Pose2{}, nullptr, nullptr, 0.1f);
    const Vec2 m1 = mean_pos(f);
    const float ex = m0.x + 2.0f * s * std::cos(theta), ey = m0.y + 2.0f * s * std::sin(theta);
    EXPECT_LT(std::hypot(m1.x - ex, m1.y - ey), p.resolution) << "theta " << theta;
  }
}

TEST(Predict, HeadingVarianceGrowsLinearly) {
  Params p = quiet_params();
  p.sigma_theta0 = 0.05f;  // all mass on h = 0
  SearchFilter f;
  ASSERT_TRUE(f.init(p));
  f.on_loss(track_at(0.25f, 0.25f, 0.3f, 0), Pose2{}, nullptr);
  auto occ = occ_with(f, {});
  f.debug_set_occ(occ.data());
  const float dt = 0.1f, dtheta = kPi / 4;
  const float q = p.sigma_h * p.sigma_h * dt / (2 * dtheta * dtheta);
  for (int k = 1; k <= 15; ++k) {
    f.debug_filter_tick(Pose2{}, nullptr, nullptr, dt);
    double w[kHeadings] = {0}, sw = 0;
    for (int c = 0; c < f.num_cells(); ++c)
      for (int h = 0; h < kHeadings; ++h) w[h] += f.belief()[c * kHeadings + h];
    double var = 0;
    for (int h = 0; h < kHeadings; ++h) {
      const int off = h < 4 ? h : h - 8;  // wrapped heading offset from h = 0
      var += w[h] * off * off;
      sw += w[h];
    }
    var /= sw;
    // Discrete random walk: var = 2 q k (heading-index units), i.e. sigma_h^2 t / dtheta^2.
    EXPECT_NEAR(var, 2 * q * k, 0.05 * 2 * q * k + 1e-3) << "tick " << k;
  }
}

TEST(Predict, WindowShiftMovesBeliefAndCountsLostMass) {
  SearchFilter f;
  ASSERT_TRUE(f.init(quiet_params()));
  f.on_loss(track_at(-4.0f, 0.25f, 0.5f, 0), Pose2{}, nullptr);
  std::vector<float> before(f.num_cells());
  f.belief_marginal(before.data());
  const Vec2 o0 = f.window_origin();
  Params p = quiet_params();
  // Robot moves 2 cells (+1 m) in x with no time passing: shift only.
  SearchFilter g;
  ASSERT_TRUE(g.init(p));
  g.on_loss(track_at(-4.0f, 0.25f, 0.5f, 0), Pose2{}, nullptr);
  g.debug_filter_tick(Pose2{1.0f, 0, 0}, nullptr, nullptr, 0.0f);
  EXPECT_FLOAT_EQ(g.window_origin().x, o0.x + 1.0f);
  std::vector<float> after(g.num_cells());
  g.belief_marginal(after.data());
  const int W = g.window_cells();
  double lost = 0;
  for (int y = 0; y < W; ++y) {
    for (int x = 0; x < 2; ++x) lost += before[y * W + x];
    for (int x = 0; x + 2 < W; ++x) EXPECT_NEAR(after[y * W + x], before[y * W + x + 2], 1e-6);
  }
  EXPECT_NEAR(g.out_of_window(), lost, 1e-5);
  EXPECT_GT(lost, 0.05);
}

TEST(Walls, DropAtWall) {
  Params p = quiet_params();
  p.sigma_h = 0.0f;
  SearchFilter f;
  ASSERT_TRUE(f.init(p));
  f.on_loss(track_at(1.25f, 0.25f, 1.5f, 0.0f), Pose2{}, nullptr);
  const int W = f.window_cells(), wall_x = 14;  // window x = 14 is odom x in [2, 2.5)
  std::vector<std::pair<int, int>> wall;
  for (int y = 0; y < W; ++y) wall.push_back({wall_x, y});
  auto occ = occ_with(f, wall);
  f.debug_set_occ(occ.data());
  const float pm_o = 1.5f * 0.1f / p.resolution, pm_d = pm_o / std::sqrt(2.0f);
  auto behind = [&] {  // mass on the far side of the wall (the on_loss Gaussian has a tail there)
    double sum = 0;
    for (int y = 0; y < W; ++y)
      for (int x = wall_x + 1; x < W; ++x)
        for (int h = 0; h < kHeadings; ++h) sum += f.belief()[(y * W + x) * kHeadings + h];
    return sum;
  };
  float total_dropped = 0;
  for (int i = 0; i < 40; ++i) {
    // Expected drop: p_m(h) * b(c, h) over closed edges (no diffusion, no stopping).
    double expect = 0;
    for (int c = 0; c < f.num_cells(); ++c)
      for (int h = 0; h < kHeadings; ++h)
        if (f.edge_open()[c * kHeadings + h] == kEdgeBlocked)
          expect += (h % 2 ? pm_d : pm_o) * f.belief()[c * kHeadings + h];
    const double before = behind();
    f.debug_filter_tick(Pose2{}, nullptr, nullptr, 0.1f);
    EXPECT_NEAR(f.dropped_mass(), expect, 1e-6) << "tick " << i;
    total_dropped += f.dropped_mass();
    // Undo the normalisation: nothing crosses the wall, so the far side can only lose mass.
    EXPECT_LE(behind() * f.last_total(), before * (1 + 1e-6) + 1e-12) << "tick " << i;
  }
  EXPECT_GT(total_dropped, 0.3f);
}

// ------------------------------------------------------------------------------------------------
// Step 3: edge mask, REDIRECT

TEST(Walls, EdgeMaskNoCornerCutting) {
  SearchFilter f;
  ASSERT_TRUE(f.init(quiet_params()));
  f.on_loss(track_at(0, 0, 1, 0), Pose2{}, nullptr);
  const int W = f.window_cells();
  auto occ = occ_with(f, {{11, 11}});
  f.debug_set_occ(occ.data());
  auto e = [&](int x, int y, int h) { return f.edge_open()[(y * W + x) * kHeadings + h]; };
  EXPECT_EQ(e(10, 11, 0), kEdgeBlocked);  // into the obstacle
  EXPECT_EQ(e(10, 11, 1), kEdgeBlocked);  // diagonal past it
  EXPECT_EQ(e(10, 11, 7), kEdgeBlocked);
  EXPECT_EQ(e(10, 10, 1), kEdgeBlocked);  // diagonal into it
  EXPECT_EQ(e(10, 10, 0), kEdgeOpen);
  EXPECT_EQ(e(11, 11, 4), kEdgeBlocked);  // out of the obstacle
  EXPECT_EQ(e(W - 1, 5, 0), kEdgeLeaves);
  EXPECT_EQ(e(0, 0, 5), kEdgeLeaves);
}

TEST(Walls, RedirectAtWall) {
  // Obstacle at window cell (12, 11); mass heads diagonally (h = 1) at it from around (11, 10).
  // Every closed heading-1 edge here has an open h+-1 neighbour, so REDIRECT loses nothing.
  auto run = [](BlockedMode mode, SearchFilter& f) {
    Params p = quiet_params();
    p.sigma_h = 0.0f;
    p.sigma_theta0 = 0.05f;
    p.blocked_mode = mode;
    ASSERT_TRUE(f.init(p));
    // Window origin is (-5, -5): cell (11, 10) centre = (0.75, 0.25).
    f.on_loss(track_at(0.75f, 0.25f, 1.0f, 1.0f), Pose2{}, nullptr);
    auto occ = occ_with(f, {{12, 11}});
    f.debug_set_occ(occ.data());
    f.on_loss(track_at(0.75f, 0.25f, 1.0f, 1.0f), Pose2{}, nullptr);  // no mass in the obstacle
    f.debug_filter_tick(Pose2{}, nullptr, nullptr, 0.1f);
  };
  SearchFilter drop, redir;
  run(BlockedMode::DROP, drop);
  run(BlockedMode::REDIRECT, redir);
  EXPECT_GT(drop.dropped_mass(), 0.01f);
  EXPECT_EQ(redir.dropped_mass(), 0.0f);
  EXPECT_NEAR(redir.last_total(), 1.0, 1e-6);  // nothing lost

  // Pre-normalisation difference REDIRECT - DROP: only on open h+-1 of cells with a closed edge.
  const int W = redir.window_cells(), N = redir.num_cells();
  double diff_sum = 0;
  for (int c = 0; c < N; ++c) {
    for (int h = 0; h < kHeadings; ++h) {
      const double a = redir.belief()[c * kHeadings + h] * redir.last_total();
      const double b = drop.belief()[c * kHeadings + h] * drop.last_total();
      const double d = a - b;
      diff_sum += d;
      if (std::fabs(d) < 1e-7) continue;  // float rounding of undoing the normalisation
      const int g1 = (h + 1) % 8, g2 = (h + 7) % 8;
      const bool from_closed = redir.edge_open()[c * kHeadings + g1] == kEdgeBlocked ||
                               redir.edge_open()[c * kHeadings + g2] == kEdgeBlocked;
      EXPECT_TRUE(from_closed) << "cell " << c % W << "," << c / W << " h " << h;
      EXPECT_EQ(redir.edge_open()[c * kHeadings + h], kEdgeOpen);
      EXPECT_GT(d, 0.0);
    }
  }
  EXPECT_NEAR(diff_sum, drop.dropped_mass(), 1e-6);
}

TEST(Walls, InitialBeliefDoesNotSpillThroughWalls) {
  // Corridor wall at window row 7 (odom y in [-1.5, -1)); leader 1 m from it.
  SearchFilter f;
  ASSERT_TRUE(f.init(test_params()));
  std::vector<std::pair<int, int>> wall;
  for (int x = 0; x < f.window_cells(); ++x) wall.push_back({x, 7});
  auto occ = occ_with(f, wall);
  f.debug_set_occ(occ.data());
  TrackInit t = track_at(1.25f, 0.0f, 1.0f, 0.0f);
  t.cov[0] = t.cov[3] = 0.5f;  // wide enough to reach across the wall
  f.on_loss(t, Pose2{}, nullptr);
  std::vector<float> marg(f.num_cells());
  f.belief_marginal(marg.data());
  const int W = f.window_cells();
  double behind = 0, front = 0;
  for (int y = 0; y < W; ++y)
    for (int x = 0; x < W; ++x) (y <= 7 ? behind : front) += marg[y * W + x];
  EXPECT_EQ(behind, 0.0);
  EXPECT_NEAR(front, 1.0, 1e-5);
}

TEST(Walls, ScanRasterizesFreeAndOccupied) {
  SearchFilter f;
  ASSERT_TRUE(f.init(quiet_params()));
  FakeScan scan;
  scan.make(Pose2{}, {{2.1f, -3, 2.1f, 3}});
  f.on_loss(track_at(1, 0, 1, 0), Pose2{}, &scan.view);
  EXPECT_EQ(f.occ()[f.cell_of({1.0f, 0.1f})], kFree);
  EXPECT_EQ(f.occ()[f.cell_of({2.2f, 0.1f})], kOccupied);
  EXPECT_EQ(f.occ()[f.cell_of({3.5f, 0.1f})], kUnknown);  // behind the wall
  // Next scan misses the wall (blind sector): with occ_memory the wall stays.
  FakeScan empty;
  empty.make(Pose2{}, {});
  for (float& r : empty.ranges) r = std::numeric_limits<float>::quiet_NaN();
  f.step(Pose2{}, &empty.view, nullptr, 0, 0.1f);
  EXPECT_EQ(f.occ()[f.cell_of({2.2f, 0.1f})], kOccupied);
}

// ------------------------------------------------------------------------------------------------
// Step 4: observation and normalisation

TEST(Observe, SeenScaledOccludedUntouched) {
  Params p = test_params();  // stopped on, so stopped is checked too
  SearchFilter f;
  ASSERT_TRUE(f.init(p));
  TrackInit t = track_at(2.0f, 0.0f, 0.2f, 0.0f);
  t.cov[0] = t.cov[3] = 4.0f;  // wide: mass everywhere
  f.on_loss(t, Pose2{}, nullptr);
  // A short wall at window x = 14 (odom x 2..2.5) across y in [-1, 1).
  std::vector<std::pair<int, int>> wall;
  for (int y = 8; y < 12; ++y) wall.push_back({14, y});
  auto occ = occ_with(f, wall);
  f.debug_set_occ(occ.data());
  const int N = f.num_cells();
  std::vector<float> b(f.belief(), f.belief() + N * kHeadings), st(f.stopped(), f.stopped() + N);
  std::vector<float> m(f.maxprod(), f.maxprod() + N * kHeadings);
  const Pose2 robot{0, 0, 0};
  f.debug_observe(robot);
  int n_seen = 0;
  for (int c = 0; c < N; ++c) {
    const Vec2 q = f.cell_centre(c);
    const float d = std::hypot(q.x, q.y);
    if (f.seen()[c]) {
      ++n_seen;
      const float fac = 1.0f - p.p_d[std::min(9, int(d))];
      for (int h = 0; h < kHeadings; ++h) {
        EXPECT_EQ(f.belief()[c * kHeadings + h], b[c * kHeadings + h] * fac);
        EXPECT_EQ(f.maxprod()[c * kHeadings + h], m[c * kHeadings + h] + std::log(fac));
      }
      EXPECT_EQ(f.stopped()[c], st[c] * fac);
      EXPECT_LE(d, p.cam_range);
      EXPECT_LE(std::fabs(std::atan2(q.y, q.x)), p.cam_fov / 2 + 0.3f);
    } else {
      EXPECT_EQ(std::memcmp(&f.belief()[c * kHeadings], &b[c * kHeadings], 8 * sizeof(float)), 0);
      EXPECT_EQ(std::memcmp(&f.stopped()[c], &st[c], sizeof(float)), 0);
    }
  }
  EXPECT_GT(n_seen, 40);
  // Behind the wall, in the FOV and in range: occluded.
  EXPECT_FALSE(f.seen()[f.cell_of({3.25f, 0.25f})]);
  EXPECT_FALSE(f.seen()[f.cell_of({2.25f, 0.25f})]);  // the wall cell itself
  EXPECT_TRUE(f.seen()[f.cell_of({1.25f, 0.25f})]);
  EXPECT_TRUE(f.seen()[f.cell_of({3.25f, 2.75f})]);  // in the FOV, beside the wall
  EXPECT_FALSE(f.seen()[f.cell_of({-1.25f, 0.25f})]);  // behind the robot
}

TEST(Observe, NormalisesToOne) {
  SearchFilter f;
  ASSERT_TRUE(f.init(test_params()));
  f.on_loss(track_at(2.0f, 0.0f, 0.5f, 0.0f), Pose2{}, nullptr);
  Observation miss;
  for (int i = 0; i < 10; ++i) f.debug_filter_tick(Pose2{}, nullptr, &miss, 0.1f);
  EXPECT_LT(f.last_total(), 0.999f);  // the observations removed mass before normalising
  EXPECT_NEAR(total_mass(f), 1.0, 1e-5);
  float mmax = -INFINITY;
  for (int j = 0; j < f.num_cells() * kHeadings; ++j) mmax = std::max(mmax, f.maxprod()[j]);
  for (int c = 0; c < f.num_cells(); ++c) mmax = std::max(mmax, f.maxprod_stopped()[c]);
  EXPECT_EQ(mmax, 0.0f);
}

// ------------------------------------------------------------------------------------------------
// Update order: shuffled cell order gives bit-identical buffers.

TEST(Order, ShuffledCellOrderIsBitIdentical) {
  for (BlockedMode mode : {BlockedMode::DROP, BlockedMode::REDIRECT}) {
    Params p = test_params();
    p.blocked_mode = mode;
    SearchFilter a, b;
    ASSERT_TRUE(a.init(p));
    ASSERT_TRUE(b.init(p));
    std::vector<int32_t> order(a.num_cells());
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), std::mt19937(42));
    b.debug_set_cell_order(order.data());
    std::vector<Seg> walls = {{-3, -1.5f, 8, -1.5f}, {-3, 1.5f, 4, 1.5f}, {3, 0.2f, 3.2f, 1.5f}};
    TrackInit t = track_at(1.0f, 0.0f, 1.2f, 0.3f);
    FakeScan scan;
    scan.make(Pose2{}, walls);
    a.on_loss(t, Pose2{}, &scan.view);
    b.on_loss(t, Pose2{}, &scan.view);
    Pose2 robot;
    Observation miss;
    for (int i = 0; i < 60; ++i) {
      robot.x += 0.04f;
      robot.yaw = 0.3f * std::sin(0.1f * i);
      miss.robot = robot;
      scan.make(robot, walls);
      a.step(robot, &scan.view, &miss, i % 3 ? 1 : 0, 0.1f);
      b.step(robot, &scan.view, &miss, i % 3 ? 1 : 0, 0.1f);
      const size_t NS = a.num_cells() * kHeadings * sizeof(float), N = a.num_cells();
      ASSERT_EQ(std::memcmp(a.belief(), b.belief(), NS), 0) << "tick " << i;
      ASSERT_EQ(std::memcmp(a.maxprod(), b.maxprod(), NS), 0) << "tick " << i;
      ASSERT_EQ(std::memcmp(a.stopped(), b.stopped(), N * sizeof(float)), 0) << "tick " << i;
      ASSERT_EQ(std::memcmp(a.maxprod_stopped(), b.maxprod_stopped(), N * sizeof(float)), 0);
      ASSERT_EQ(std::memcmp(a.edge_open(), b.edge_open(), N * kHeadings), 0);
    }
  }
}

// ------------------------------------------------------------------------------------------------
// Step 5: max-product, target with hysteresis, intercept

namespace {
// Corridor along x between window rows 8 and 12 (odom y = -1 and y = 1), robot at the origin.
std::vector<uint8_t> corridor(const SearchFilter& f) {
  std::vector<std::pair<int, int>> walls;
  for (int x = 0; x < f.window_cells(); ++x) walls.push_back({x, 8}), walls.push_back({x, 12});
  return occ_with(f, walls);
}
int argmax_m(const SearchFilter& f) {
  int best = 0;
  for (int j = 0; j < f.num_cells() * kHeadings; ++j)
    if (f.maxprod()[j] > f.maxprod()[best]) best = j;
  return best;
}
}  // namespace

TEST(MaxProduct, CorridorArgmaxOnConstantVelocityLine) {
  for (float s : {0.6f, 1.0f, 1.5f}) {
    SearchFilter f;
    ASSERT_TRUE(f.init(test_params()));  // defaults: heading diffusion and stopped state on
    auto occ = corridor(f);
    f.debug_set_occ(occ.data());
    f.on_loss(track_at(-3.75f, 0.25f, s, 0), Pose2{}, nullptr);
    for (int k = 1; k <= 30; ++k) {
      f.debug_filter_tick(Pose2{}, nullptr, nullptr, 0.1f);
      const int j = argmax_m(f);
      const Vec2 q = f.cell_centre(j / kHeadings);
      const float cv_x = -3.75f + s * 0.1f * k;
      if (cv_x > 4.5f) break;  // the line leaves the window
      EXPECT_EQ(j % kHeadings, 0) << "s " << s << " tick " << k;
      EXPECT_FLOAT_EQ(q.y, 0.25f) << "s " << s << " tick " << k;
      EXPECT_LE(std::fabs(q.x - cv_x), 0.5f * f.resolution() + 1e-4f) << "s " << s << " tick " << k;
    }
  }
}

TEST(MaxProduct, ObservationMovesTargetOffTheLine) {
  // Open area; the robot keeps seeing the straight-ahead line, so the most likely path turns.
  SearchFilter f;
  ASSERT_TRUE(f.init(test_params()));
  auto occ = occ_with(f, {});
  f.debug_set_occ(occ.data());
  f.on_loss(track_at(0.75f, 0.25f, 1.0f, 0), Pose2{}, nullptr);
  Observation miss;
  miss.robot = Pose2{0, 0, 0};
  for (int k = 0; k < 20; ++k) f.debug_filter_tick(Pose2{}, nullptr, &miss, 0.1f);
  const Vec2 q = f.cell_centre(argmax_m(f) / kHeadings);
  EXPECT_FALSE(f.seen()[argmax_m(f) / kHeadings]);
  EXPECT_GT(std::fabs(std::atan2(q.y, q.x)), f.params().cam_fov / 2 - 0.2f);
}

TEST(Target, HysteresisKeepsTargetBetweenNearlyEqualBranches) {
  // T-junction: corridor along x ending at a wall at x = 3, open branches up and down.
  // The leader goes up or down: two nearly equal branches; the target may not dither.
  SearchFilter f;
  ASSERT_TRUE(f.init(test_params()));
  const int W = f.window_cells();
  std::vector<std::pair<int, int>> walls;
  for (int x = 0; x < 15; ++x) {
    if (x < 13) walls.push_back({x, 8}), walls.push_back({x, 12});
  }
  for (int y = 0; y < W; ++y) walls.push_back({16, y});
  auto occ = occ_with(f, walls);
  f.debug_set_occ(occ.data());
  f.on_loss(track_at(0.75f, 0.25f, 1.0f, 0), Pose2{}, nullptr);
  int switches = 0;
  for (int k = 0; k < 60; ++k) {
    SearchOutput out = f.step(Pose2{}, nullptr, nullptr, 0, 0.1f);
    if (out.mode != Mode::SEARCH) break;
    switches += out.target_switched;
  }
  EXPECT_LE(switches, 2);
}

TEST(Target, InterceptIsFirstReachablePointOnTheLine) {
  // Leader 1.75 m ahead walking towards the robot at 0.3 m/s; robot 0.5 m/s facing it.
  SearchFilter f;
  const Params p = test_params();
  ASSERT_TRUE(f.init(p));
  auto occ = corridor(f);
  f.debug_set_occ(occ.data());
  f.on_loss(track_at(1.75f, 0.25f, -0.3f, 0), Pose2{}, nullptr);
  SearchOutput out = f.step(Pose2{}, nullptr, nullptr, 0, 0.1f);
  ASSERT_EQ(out.mode, Mode::SEARCH);
  ASSERT_TRUE(out.path_valid);
  EXPECT_EQ(out.target_heading, 4);
  EXPECT_FLOAT_EQ(out.intercept.y, out.target.y);
  const float s = 0.3f, dt = 0.1f;
  const int k = static_cast<int>(std::lround((out.target.x - out.intercept.x) / (s * dt)));
  auto reachable = [&](int kk) {
    const float x = out.target.x - kk * s * dt;
    return std::hypot(x, 0.25f) / p.v_full <= kk * dt + p.t_margin + 1e-3f;
  };
  EXPECT_GT(k, 0);
  EXPECT_TRUE(reachable(k));
  EXPECT_FALSE(reachable(k - 1));
}

// ------------------------------------------------------------------------------------------------
// Step 6: Dijkstra, vision-sector travel times, output path

TEST(Planner, PathGoesAroundWallAndFacesTarget) {
  SearchFilter f;
  Params p = test_params();
  ASSERT_TRUE(f.init(p));
  // Wall at odom x = 1.5 from y = -3 to y = 1.5; leader lost behind it, walking +y.
  std::vector<std::pair<int, int>> wall;
  for (int y = 4; y < 13; ++y) wall.push_back({13, y});
  auto occ = occ_with(f, wall);
  f.debug_set_occ(occ.data());
  f.on_loss(track_at(2.75f, -1.25f, 0.0f, 0.5f), Pose2{}, nullptr);
  SearchOutput out = f.step(Pose2{}, nullptr, nullptr, 0, 0.1f);
  ASSERT_TRUE(out.path_valid);
  const PathView& path = out.path;
  ASSERT_GT(path.size, 10);
  EXPECT_FLOAT_EQ(path.poses[0].x, 0.0f);
  EXPECT_FLOAT_EQ(path.poses[0].y, 0.0f);
  // The path stops `standoff` before the intercept and still faces the target.
  const Pose2& end = path.poses[path.size - 1];
  EXPECT_GT(std::hypot(end.x - out.intercept.x, end.y - out.intercept.y), 0.3f);
  EXPECT_LE(std::hypot(end.x - out.intercept.x, end.y - out.intercept.y), p.standoff + 1e-3f);
  for (int i = 0; i < path.size; ++i) {
    const Pose2& q = path.poses[i];
    const int c = f.cell_of({q.x, q.y});
    ASSERT_GE(c, 0);
    EXPECT_NE(f.occ()[c], kOccupied) << "pose " << i;
    if (i > 0) {
      const float step = std::hypot(q.x - path.poses[i - 1].x, q.y - path.poses[i - 1].y);
      EXPECT_LE(step, p.path_spacing + 1e-4f);
    }
    if (std::hypot(out.target.x - q.x, out.target.y - q.y) > 0.3f) {
      EXPECT_NEAR(wrap_angle_for_test(q.yaw - std::atan2(out.target.y - q.y, out.target.x - q.x)), 0, 1e-4);
    }
  }
  // The path passes the end of the wall (y > 1.5) and is shortcut (few corners).
  float ymax = -1e9;
  for (int i = 0; i < path.size; ++i) ymax = std::max(ymax, path.poses[i].y);
  EXPECT_GT(ymax, 1.5f);
}

TEST(Planner, WithinStandoffTurnsInPlaceTowardsTarget) {
  SearchFilter f;
  ASSERT_TRUE(f.init(test_params()));
  auto occ = occ_with(f, {});
  f.debug_set_occ(occ.data());
  // Leader stopped 0.75 m to the robot's left; the robot faces +x.
  f.on_loss(track_at(0.25f, 0.75f, 0.0f, 0.0f), Pose2{}, nullptr);
  SearchOutput out = f.step(Pose2{}, nullptr, nullptr, 0, 0.1f);
  ASSERT_TRUE(out.path_valid);
  ASSERT_EQ(out.path.size, 1);
  EXPECT_FLOAT_EQ(out.path.poses[0].x, 0.0f);
  EXPECT_NEAR(out.path.poses[0].yaw, std::atan2(out.target.y, out.target.x), 1e-4);
}

TEST(Planner, VisionSectorTravelTimeTurnsBeforeGoingBackwards) {
  // Leader behind the robot walking towards it. Facing away the robot first turns for
  // (pi - 1.4) / 0.5 = 3.5 s, so its intercept lies further along the leader's line.
  auto run = [](float yaw) {
    SearchFilter f;
    EXPECT_TRUE(f.init(test_params()));
    auto occ = occ_with(f, {});
    f.debug_set_occ(occ.data());
    f.on_loss(track_at(-1.75f, 0.25f, 0.3f, 0), Pose2{0, 0, yaw}, nullptr);
    return f.step(Pose2{0, 0, yaw}, nullptr, nullptr, 0, 0.1f);
  };
  const SearchOutput facing = run(kPi), away = run(0.0f);
  EXPECT_LT(facing.intercept.x, -1.0f);
  EXPECT_GT(away.intercept.x, facing.intercept.x + 0.2f);
}

// ------------------------------------------------------------------------------------------------
// Step 7: state machine TRACKING / SEARCH / GIVE_UP

namespace {
Observation seen_at(float x, float y, float vx, float vy) {
  Observation o;
  o.matched = o.has_track = true;
  o.track = track_at(x, y, vx, vy);
  return o;
}
}  // namespace

TEST(StateMachine, GraceThenSearchThenReacquire) {
  SearchFilter f;
  ASSERT_TRUE(f.init(test_params()));
  Observation hit = seen_at(1.5f, 0, 1.0f, 0), miss;
  EXPECT_EQ(f.step(Pose2{}, nullptr, nullptr, 0, 0.1f).mode, Mode::TRACKING);  // no track yet
  EXPECT_EQ(f.step(Pose2{}, nullptr, &hit, 1, 0.1f).mode, Mode::TRACKING);
  EXPECT_EQ(f.step(Pose2{}, nullptr, &miss, 1, 0.1f).mode, Mode::TRACKING);  // 0.1 s
  EXPECT_EQ(f.step(Pose2{}, nullptr, &miss, 1, 0.1f).mode, Mode::TRACKING);  // 0.2 s
  SearchOutput out = f.step(Pose2{}, nullptr, &miss, 1, 0.1f);              // 0.3 s = t_grace
  EXPECT_EQ(out.mode, Mode::SEARCH);
  EXPECT_EQ(out.lost, LossCause::kNotMatched);
  EXPECT_TRUE(out.path_valid);
  EXPECT_NEAR(out.time_since_loss, 0.3f, 1e-5);  // predicted from the last sighting
  // Catch-up prediction: the target is about 0.3 s * 1 m/s ahead of the sighting.
  EXPECT_NEAR(out.target.x, 1.75f, 0.26f);
  out = f.step(Pose2{}, nullptr, &hit, 1, 0.1f);
  EXPECT_EQ(out.mode, Mode::TRACKING);
  EXPECT_TRUE(out.reacquired);
  EXPECT_FALSE(out.path_valid);
}

// A slow or stuttering camera is not a loss: ticks without a frame do not count as misses.
TEST(StateMachine, SlowCameraStaysTracking) {
  SearchFilter f;
  ASSERT_TRUE(f.init(test_params()));
  Observation hit = seen_at(1.5f, 0, 0.0f, 0);
  for (int i = 0; i < 40; ++i) {  // 15 Hz ticks, a VISIBLE frame every 0.6 s
    const bool frame = i % 9 == 0;
    const SearchOutput out = f.step(Pose2{}, nullptr, &hit, frame ? 1 : 0, 1.0f / 15);
    EXPECT_EQ(out.mode, Mode::TRACKING) << i;
    EXPECT_EQ(out.lost, LossCause::kNone) << i;
  }
}

// Unmatched frames: perception's lost_duration decides, not the filter's own tick count.
TEST(StateMachine, LostDurationFromPerceptionTriggersSearch) {
  SearchFilter f;
  ASSERT_TRUE(f.init(test_params()));
  Observation hit = seen_at(1.5f, 0, 1.0f, 0), miss;
  f.step(Pose2{}, nullptr, &hit, 1, 0.1f);
  miss.lost_duration = 0.1f;
  EXPECT_EQ(f.step(Pose2{}, nullptr, &miss, 1, 0.1f).mode, Mode::TRACKING);
  // A late frame that perception says has not matched for 0.5 s: SEARCH (second miss in a row).
  miss.lost_duration = 0.5f;
  const SearchOutput out = f.step(Pose2{}, nullptr, &miss, 1, 0.1f);
  EXPECT_EQ(out.mode, Mode::SEARCH);
  EXPECT_EQ(out.lost, LossCause::kNotMatched);
  EXPECT_NEAR(out.lost_after, 0.5f, 1e-6);
  EXPECT_NEAR(out.time_since_loss, 0.5f, 1e-5);  // predicted from the last sighting
}

// 3 Hz camera: one unmatched frame is already past t_grace, but needs n_grace = 2 in a row.
TEST(StateMachine, SingleMissOnSlowCameraIsNotALoss) {
  SearchFilter f;
  ASSERT_TRUE(f.init(test_params()));
  Observation hit = seen_at(1.5f, 0, 1.0f, 0), miss;
  f.step(Pose2{}, nullptr, &hit, 1, 0.33f);
  miss.lost_duration = 0.33f;
  EXPECT_EQ(f.step(Pose2{}, nullptr, &miss, 1, 0.33f).mode, Mode::TRACKING);
  EXPECT_EQ(f.step(Pose2{}, nullptr, &hit, 1, 0.33f).mode, Mode::TRACKING);
  EXPECT_EQ(f.step(Pose2{}, nullptr, &miss, 1, 0.33f).mode, Mode::TRACKING);  // count restarted
  miss.lost_duration = 0.66f;
  EXPECT_EQ(f.step(Pose2{}, nullptr, &miss, 1, 0.33f).mode, Mode::SEARCH);
}

TEST(StateMachine, NoFramesForTsilentTriggersSearch) {
  SearchFilter f;
  ASSERT_TRUE(f.init(test_params()));
  Observation hit = seen_at(1.5f, 0, 1.0f, 0);
  f.step(Pose2{}, nullptr, &hit, 1, 0.1f);
  SearchOutput out;
  int ticks = 0;
  do {
    out = f.step(Pose2{}, nullptr, nullptr, 0, 0.1f);
    ++ticks;
  } while (out.mode == Mode::TRACKING && ticks < 50);
  EXPECT_EQ(out.mode, Mode::SEARCH);
  EXPECT_EQ(out.lost, LossCause::kSilent);
  EXPECT_NEAR(out.lost_after, test_params().t_silent, 0.051f);
}

TEST(StateMachine, GiveUpAfterTmaxThenGoToLastSeenAndRotate) {
  Params p = test_params();
  p.t_max = 2.0f;
  SearchFilter f;
  ASSERT_TRUE(f.init(p));
  Observation hit = seen_at(2.0f, 1.0f, 0.5f, 0);
  f.step(Pose2{}, nullptr, &hit, 1, 0.1f);
  SearchOutput out;
  int ticks = 0;
  do {
    out = f.step(Pose2{}, nullptr, nullptr, 0, 0.1f);
  } while (out.mode != Mode::GIVE_UP && ++ticks < 100);
  ASSERT_EQ(out.mode, Mode::GIVE_UP);
  EXPECT_NEAR(out.time_since_loss, 2.0f, 0.15f);
  ASSERT_TRUE(out.path_valid);
  const Pose2& end = out.path.poses[out.path.size - 1];
  EXPECT_NEAR(end.x, 2.0f, 1e-3);  // the last-seen point, kept in odom
  EXPECT_NEAR(end.y, 1.0f, 1e-3);
  // At the last-seen point: a single pose that turns in place.
  out = f.step(Pose2{2.0f, 1.0f, 0.3f}, nullptr, nullptr, 0, 0.1f);
  ASSERT_EQ(out.path.size, 1);
  EXPECT_FLOAT_EQ(out.path.poses[0].x, 2.0f);
  EXPECT_GT(out.path.poses[0].yaw, 0.3f + 0.5f);
  // A late detection still recovers.
  EXPECT_EQ(f.step(Pose2{2.0f, 1.0f, 0.3f}, nullptr, &hit, 1, 0.1f).mode, Mode::TRACKING);
}

TEST(StateMachine, GiveUpWhenMassLeavesTheWindow) {
  SearchFilter f;
  ASSERT_TRUE(f.init(test_params()));
  Observation hit = seen_at(3.5f, 0, 2.0f, 0);  // fast, close to the window edge
  f.step(Pose2{}, nullptr, &hit, 1, 0.1f);
  SearchOutput out;
  int ticks = 0;
  do {
    out = f.step(Pose2{}, nullptr, nullptr, 0, 0.1f);
  } while (out.mode == Mode::TRACKING || (out.mode == Mode::SEARCH && ++ticks < 200));
  EXPECT_EQ(out.mode, Mode::GIVE_UP);
  EXPECT_GT(out.out_of_window, 0.5f);
  EXPECT_LT(out.time_since_loss, test_params().t_max);
}

TEST(StateMachine, ResetReturnsToTrackingWithoutTrack) {
  SearchFilter f;
  ASSERT_TRUE(f.init(test_params()));
  f.on_loss(track_at(1, 0, 1, 0), Pose2{}, nullptr);
  EXPECT_EQ(f.mode(), Mode::SEARCH);
  f.reset();
  EXPECT_EQ(f.mode(), Mode::TRACKING);
  EXPECT_EQ(f.step(Pose2{}, nullptr, nullptr, 0, 1.0f).mode, Mode::TRACKING);
}

// ------------------------------------------------------------------------------------------------
// Smart skips: support boxes and Dijkstra early exit must not change any result.

namespace {
struct SkipRun {
  SearchFilter f;
  std::vector<SearchOutput> outs;
};
void run_scenario(SearchFilter& f, std::vector<SearchOutput>* outs, BlockedMode mode,
                  bool skips_off, bool check_boxes) {
  Params p;  // 15 m round window, default tail cut
  p.blocked_mode = mode;
  ASSERT_TRUE(f.init(p));
  f.debug_disable_skips(skips_off);
  std::vector<Seg> walls = {{-8, -1.5f, 12, -1.5f}, {-8, 1.5f, 4, 1.5f}, {5.2f, 1.5f, 12, 1.5f},
                            {4, 1.5f, 4, 6}, {5.2f, 1.5f, 5.2f, 6}};
  FakeScan scan;
  Pose2 robot;
  scan.make(robot, walls);
  f.on_loss(track_at(2.5f, 0.0f, 1.2f, 0.1f), robot, &scan.view);
  Observation miss;
  for (int i = 0; i < 120; ++i) {
    robot.x += 0.04f;
    robot.yaw = 0.4f * std::sin(0.07f * i);
    miss.robot = robot;
    scan.make(robot, walls);
    outs->push_back(f.step(robot, &scan.view, &miss, i % 4 ? 1 : 0, 0.1f));
    if (!check_boxes) continue;
    // Every nonzero state lies inside its support box.
    const int W = f.window_cells();
    const CellBox bb = f.debug_box_belief(), sb = f.debug_box_stopped();
    for (int c = 0; c < f.num_cells(); ++c) {
      const int x = c % W, y = c / W;
      bool any = false;
      for (int h = 0; h < kHeadings; ++h)
        any |= f.belief()[c * kHeadings + h] != 0.0f || f.maxprod()[c * kHeadings + h] != -INFINITY;
      if (any) {
        ASSERT_TRUE(x >= bb.x0 && x <= bb.x1 && y >= bb.y0 && y <= bb.y1) << "tick " << i;
      }
      if (f.stopped()[c] != 0.0f || f.maxprod_stopped()[c] != -INFINITY) {
        ASSERT_TRUE(x >= sb.x0 && x <= sb.x1 && y >= sb.y0 && y <= sb.y1) << "tick " << i;
      }
    }
  }
}
}  // namespace

TEST(Skips, SameResultAsFullPasses) {
  for (BlockedMode mode : {BlockedMode::DROP, BlockedMode::REDIRECT}) {
    SearchFilter a, b;
    std::vector<SearchOutput> oa, ob;
    run_scenario(a, &oa, mode, false, false);
    run_scenario(b, &ob, mode, true, false);
    ASSERT_EQ(oa.size(), ob.size());
    for (size_t i = 0; i < oa.size(); ++i) {
      ASSERT_EQ(oa[i].mode, ob[i].mode) << "tick " << i;
      ASSERT_EQ(oa[i].target.x, ob[i].target.x) << "tick " << i;
      ASSERT_EQ(oa[i].target.y, ob[i].target.y) << "tick " << i;
      ASSERT_EQ(oa[i].intercept.x, ob[i].intercept.x) << "tick " << i;
      ASSERT_EQ(oa[i].intercept.y, ob[i].intercept.y) << "tick " << i;
      ASSERT_EQ(oa[i].path.size, ob[i].path.size) << "tick " << i;
    }
    const size_t NS = a.num_cells() * kHeadings * sizeof(float), N = a.num_cells() * sizeof(float);
    EXPECT_EQ(std::memcmp(a.belief(), b.belief(), NS), 0);
    EXPECT_EQ(std::memcmp(a.maxprod(), b.maxprod(), NS), 0);
    EXPECT_EQ(std::memcmp(a.stopped(), b.stopped(), N), 0);
    EXPECT_EQ(std::memcmp(a.maxprod_stopped(), b.maxprod_stopped(), N), 0);
    EXPECT_EQ(a.out_of_window(), b.out_of_window());
  }
}

TEST(Skips, BoxCoversSupport) {
  SearchFilter f;
  std::vector<SearchOutput> outs;
  run_scenario(f, &outs, BlockedMode::DROP, false, true);
  // Early in the search the support is much smaller than the window.
  EXPECT_LT(outs.size(), 200u);
}
