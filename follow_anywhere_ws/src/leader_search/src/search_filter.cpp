// Lost-leader search core. See search_filter.hpp and docs/leader_search_spec.md.
#include "leader_search/search_filter.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

namespace leader_search {
namespace {

constexpr float kPi = 3.14159265358979f;
constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr float kDTheta = kPi / 4.0f;  // heading spacing, 45 degrees
// d(h): cell offset of heading h (0 = +x, counter-clockwise).
constexpr int kDX[kHeadings] = {1, 1, 0, -1, -1, -1, 0, 1};
constexpr int kDY[kHeadings] = {0, 1, 1, 1, 0, -1, -1, -1};

// Tail truncation (deviation, keeps the support boxes small): belief below Params::tail_cut is 0,
// max-product scores below kMinLog (m is renormalised to max 0 every tick) are -inf. The stay/move
// smear lets a sliver of mass advance one cell per tick (5 m/s); those tails mean nothing.
constexpr float kMinLog = -60.0f;
inline float cut_m(float v) { return v < kMinLog ? kNegInf : v; }

// Grows a CellBox over the cells where a pass wrote something nonzero (order-independent).
struct BoxAcc {
  int x0 = 1 << 30, y0 = 1 << 30, x1 = -1, y1 = -1;
  void add(int x, int y) {
    x0 = std::min(x0, x);
    y0 = std::min(y0, y);
    x1 = std::max(x1, x);
    y1 = std::max(y1, y);
  }
  CellBox box() const { return x1 < 0 ? CellBox{} : CellBox{x0, y0, x1, y1}; }
};
CellBox unite(const CellBox& a, const CellBox& b) {
  if (a.empty()) return b;
  if (b.empty()) return a;
  return {std::min(a.x0, b.x0), std::min(a.y0, b.y0), std::max(a.x1, b.x1), std::max(a.y1, b.y1)};
}

// 4-wide SIMD (GCC vector extensions: SSE on x86-64 without -march, NEON on the Jetson). A cell's
// 8 headings are two vectors: lo = h 0..3, hi = h 4..7.
typedef float v4f __attribute__((vector_size(16)));
typedef int32_t v4i __attribute__((vector_size(16)));
inline v4f ld4(const float* p) {
  v4f v;
  std::memcpy(&v, p, sizeof v);
  return v;
}
inline void st4(float* p, v4f v) { std::memcpy(p, &v, sizeof v); }
inline v4f splat(float x) { return v4f{x, x, x, x}; }
inline v4i splat_i(int32_t x) { return v4i{x, x, x, x}; }
inline v4f vmax(v4f a, v4f b) { return a > b ? a : b; }
// b[h-1] and b[h+1] over the 8 headings.
inline v4f prev_lo(v4f lo, v4f hi) { return __builtin_shuffle(hi, lo, v4i{3, 4, 5, 6}); }
inline v4f prev_hi(v4f lo, v4f hi) { return __builtin_shuffle(lo, hi, v4i{3, 4, 5, 6}); }
inline v4f next_lo(v4f lo, v4f hi) { return __builtin_shuffle(lo, hi, v4i{1, 2, 3, 4}); }
inline v4f next_hi(v4f lo, v4f hi) { return __builtin_shuffle(hi, lo, v4i{1, 2, 3, 4}); }
// Lane masks (-1 / 0) of the bits of an 8-bit heading mask.
inline v4i bits_lo(unsigned m) { return (splat_i(int32_t(m)) & v4i{1, 2, 4, 8}) != 0; }
inline v4i bits_hi(unsigned m) { return (splat_i(int32_t(m)) & v4i{16, 32, 64, 128}) != 0; }
inline float hsum(v4f a) { return (a[0] + a[1]) + (a[2] + a[3]); }
inline float hmax(v4f a) { return std::max(std::max(a[0], a[1]), std::max(a[2], a[3])); }
inline bool any_lane(v4i a) { return (a[0] | a[1] | a[2] | a[3]) != 0; }

inline float wrap(float a) { return std::remainder(a, 2.0f * kPi); }
inline float safe_log(float x) { return x > 0.0f ? std::log(x) : kNegInf; }
inline int hm1(int h) { return (h + kHeadings - 1) % kHeadings; }
inline int hp1(int h) { return (h + 1) % kHeadings; }
inline float max3(float a, float b, float c) { return std::max(a, std::max(b, c)); }

// Visits the cells of segment (x0,y0) -> (x1,y1), in window cell coordinates (Amanatides-Woo).
// Stops when f(cell) returns false or the segment leaves the W x W window.
template <class F>
void traverse(float x0, float y0, float x1, float y1, int W, F&& f) {
  int ix = static_cast<int>(std::floor(x0)), iy = static_cast<int>(std::floor(y0));
  const int ex = static_cast<int>(std::floor(x1)), ey = static_cast<int>(std::floor(y1));
  const float dx = x1 - x0, dy = y1 - y0;
  const int sx = dx > 0.0f ? 1 : -1, sy = dy > 0.0f ? 1 : -1;
  const float tdx = dx != 0.0f ? std::fabs(1.0f / dx) : kInf;
  const float tdy = dy != 0.0f ? std::fabs(1.0f / dy) : kInf;
  float tmx = dx != 0.0f ? (dx > 0.0f ? (ix + 1 - x0) : (x0 - ix)) * tdx : kInf;
  float tmy = dy != 0.0f ? (dy > 0.0f ? (iy + 1 - y0) : (y0 - iy)) * tdy : kInf;
  const int n = std::abs(ex - ix) + std::abs(ey - iy);
  for (int i = 0;; ++i) {
    if (ix < 0 || iy < 0 || ix >= W || iy >= W) return;
    if (!f(iy * W + ix)) return;
    if (i >= n) return;
    if (tmx < tmy) {
      tmx += tdx;
      ix += sx;
    } else {
      tmy += tdy;
      iy += sy;
    }
  }
}

// Fixed-capacity binary min-heap over (key, cell), storage owned by the filter.
struct HeapView {
  float* key;
  int32_t* cell;
  int32_t cap;
  int32_t size = 0;
  bool push(float k, int32_t c) {
    if (size >= cap) return false;
    int i = size++;
    while (i > 0) {
      const int parent = (i - 1) / 2;
      if (key[parent] <= k) break;
      key[i] = key[parent];
      cell[i] = cell[parent];
      i = parent;
    }
    key[i] = k;
    cell[i] = c;
    return true;
  }
  void pop(float* k, int32_t* c) {
    *k = key[0];
    *c = cell[0];
    const float lk = key[--size];
    const int32_t lc = cell[size];
    int i = 0;
    for (;;) {
      int child = 2 * i + 1;
      if (child >= size) break;
      if (child + 1 < size && key[child + 1] < key[child]) ++child;
      if (key[child] >= lk) break;
      key[i] = key[child];
      cell[i] = cell[child];
      i = child;
    }
    key[i] = lk;
    cell[i] = lc;
  }
};

}  // namespace

// Constants of one predict substep (spec Predict, step 2).
struct SearchFilter::TickConst {
  float q, l_stay_h, l_q;          // heading diffusion q; log(1-2q), log q
  float pm[kHeadings];             // p_m(h) = s*dt_s / L(h)
  float a, l_1ma, l_a;             // a = lambda_stop*dt_s (moving -> stopped)
  float g;                         // g = lambda_go*dt_s (stopped -> moving)
  bool mp_move[kHeadings];         // max-product: heading h advances one cell this substep
  // By number of open headings n = 0..8 (precomputed, no logf per cell):
  float go[kHeadings + 1], l_go[kHeadings + 1];  // g / n and its log
  float g_eff[kHeadings + 1], l_1mg[kHeadings + 1];  // g (0 if n = 0) and log(1 - g_eff)
};

// ------------------------------------------------------------------------------------------------
// Parameters

bool validate(const Params& p, std::string* err) {
  auto fail = [&](const char* why) {
    if (err) *err = why;
    return false;
  };
  if (!(p.rate_hz > 0.0f)) return fail("rate_hz must be > 0");
  if (p.window_cells < 4 || p.window_cells > 512) return fail("window_cells must be in [4, 512]");
  if (p.window_radius < 0.0f || 2.0f * std::ceil(p.window_radius / p.resolution) > 512.0f ||
      (p.window_radius > 0.0f && p.window_radius < 2.0f * p.resolution))
    return fail("window_radius must be 0 (square window) or in [2 r, 256 r]");
  if (!(p.resolution > 0.0f)) return fail("resolution must be > 0");
  if (!(p.s_min > 0.0f) || p.s_max < p.s_min) return fail("need 0 < s_min <= s_max");
  if (p.t_grace < 0.0f || p.t_max <= 0.0f) return fail("t_grace >= 0, t_max > 0");
  if (p.sigma_theta0 <= 0.0f || p.sigma_h < 0.0f) return fail("sigma_theta0 > 0, sigma_h >= 0");
  if (p.p_stop0 < 0.0f || p.p_stop0 > 1.0f) return fail("p_stop0 must be in [0, 1]");
  const float dt = 1.0f / p.rate_hz;
  if (p.lambda_stop < 0.0f || p.lambda_go < 0.0f || p.lambda_stop * dt > 1.0f ||
      p.lambda_go * dt > 1.0f)
    return fail("lambda_stop/lambda_go must be >= 0 and lambda*dt <= 1");
  if (!(p.cam_fov > 0.0f) || p.cam_fov > 2.0f * kPi || !(p.cam_range > 0.0f))
    return fail("cam_fov in (0, 2pi], cam_range > 0");
  for (float pd : p.p_d)
    if (pd < 0.0f || pd >= 1.0f) return fail("p_d[k] must be in [0, 1)");
  if (p.rho < 0.0f || p.delta_log < 0.0f) return fail("rho, delta_log must be >= 0");
  if (p.K_f < 0 || p.K_f > 1000 || p.t_margin < 0.0f) return fail("K_f in [0,1000], t_margin >= 0");
  if (!(p.v_full > 0.0f) || p.v_part < 0.0f || !(p.omega_max > 0.0f))
    return fail("v_full > 0, v_part >= 0, omega_max > 0");
  if (p.inflate_cells < 0 || !(p.path_spacing > 0.0f) || p.wall_cost < 0.0f || p.standoff < 0.0f || p.tail_cut < 0.0f || p.tail_cut > 1e-3f) return fail("inflate_cells >= 0, path_spacing > 0");
  return true;
}

bool load_params(const std::string& yaml_path, Params* out, std::string* err) {
  try {
    YAML::Node root = YAML::LoadFile(yaml_path);
    YAML::Node n = root;
    if (root.IsMap()) {
      for (const auto& kv : root) {
        if (kv.second.IsMap() && kv.second["ros__parameters"]) {
          n = kv.second["ros__parameters"];
          break;
        }
      }
    }
    Params p = *out;
#define LS_READ(name) \
  if (n[#name]) p.name = n[#name].as<decltype(p.name)>();
    LS_READ(rate_hz) LS_READ(window_cells) LS_READ(window_radius) LS_READ(resolution) LS_READ(t_grace)
    LS_READ(s_min) LS_READ(s_max) LS_READ(v_stop) LS_READ(sigma_theta0) LS_READ(sigma_h)
    LS_READ(stopped_enabled) LS_READ(p_stop0) LS_READ(lambda_stop) LS_READ(lambda_go)
    LS_READ(occ_memory) LS_READ(cam_fov) LS_READ(cam_range) LS_READ(cam_yaw) LS_READ(rho) LS_READ(delta_log)
    LS_READ(K_f) LS_READ(t_margin) LS_READ(v_full) LS_READ(v_part) LS_READ(omega_max)
    LS_READ(sector_full) LS_READ(sector_part) LS_READ(inflate_cells) LS_READ(wall_cost) LS_READ(path_spacing) LS_READ(standoff) LS_READ(tail_cut)
    LS_READ(t_max) LS_READ(p_min) LS_READ(p_out_max) LS_READ(giveup_reach)
#undef LS_READ
    if (n["blocked_mode"]) {
      const std::string m = n["blocked_mode"].as<std::string>();
      if (m == "DROP") {
        p.blocked_mode = BlockedMode::DROP;
      } else if (m == "REDIRECT") {
        p.blocked_mode = BlockedMode::REDIRECT;
      } else {
        if (err) *err = "blocked_mode must be DROP or REDIRECT";
        return false;
      }
    }
    if (n["p_d"]) {
      const YAML::Node pd = n["p_d"];
      if (!pd.IsSequence() || pd.size() != static_cast<size_t>(kPdBins)) {
        if (err) *err = "p_d must be a list of 10 values";
        return false;
      }
      for (int k = 0; k < kPdBins; ++k) p.p_d[k] = pd[k].as<float>();
    }
    if (!validate(p, err)) return false;
    *out = p;
    return true;
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return false;
  }
}

// ------------------------------------------------------------------------------------------------
// Setup

SearchFilter::SearchFilter() = default;
SearchFilter::~SearchFilter() = default;

bool SearchFilter::init(const Params& p, std::string* err) {
  if (!validate(p, err)) return false;
  p_ = p;
  // Round window of radius R: W = 2 ceil(R / r) cells across; 0 = square window_cells.
  W_ = p.window_radius > 0.0f ? 2 * static_cast<int32_t>(std::ceil(p.window_radius / p.resolution))
                              : p.window_cells;
  N_ = W_ * W_;
  r_ = p.resolution;
  const float max_len = 4.0f * W_ * r_;  // generous path length bound
  path_cap_ = std::max(N_, static_cast<int32_t>(std::ceil(max_len / p.path_spacing))) + 2;

  // One arena, carved into 64-byte aligned buffers.
  const size_t NS = static_cast<size_t>(N_) * kHeadings, N = static_cast<size_t>(N_);
  struct Slot {
    void** ptr;
    size_t bytes;
  } slots[] = {
      {reinterpret_cast<void**>(&b_), NS * sizeof(float)},
      {reinterpret_cast<void**>(&b2_), NS * sizeof(float)},
      {reinterpret_cast<void**>(&st_), N * sizeof(float)},
      {reinterpret_cast<void**>(&st2_), N * sizeof(float)},
      {reinterpret_cast<void**>(&m_), NS * sizeof(float)},
      {reinterpret_cast<void**>(&m2_), NS * sizeof(float)},
      {reinterpret_cast<void**>(&ms_), N * sizeof(float)},
      {reinterpret_cast<void**>(&ms2_), N * sizeof(float)},
      {reinterpret_cast<void**>(&edge_), NS},
      {reinterpret_cast<void**>(&occ_), N},
      {reinterpret_cast<void**>(&seen_), N},
      {reinterpret_cast<void**>(&blocked_), N},
      {reinterpret_cast<void**>(&near_), N},
      {reinterpret_cast<void**>(&inside_), N},
      {reinterpret_cast<void**>(&goal_), N},
      {reinterpret_cast<void**>(&emask_), N},
      {reinterpret_cast<void**>(&seen_scratch_), N},
      {reinterpret_cast<void**>(&leave_), N},
      {reinterpret_cast<void**>(&goals_), N * sizeof(int32_t)},
      {reinterpret_cast<void**>(&dist_), N * sizeof(float)},
      {reinterpret_cast<void**>(&parent_), N * sizeof(int32_t)},
      {reinterpret_cast<void**>(&heap_key_), (NS + 1) * sizeof(float)},
      {reinterpret_cast<void**>(&heap_cell_), (NS + 1) * sizeof(int32_t)},
      {reinterpret_cast<void**>(&chain_), N * sizeof(int32_t)},
      {reinterpret_cast<void**>(&corners_), (N + 2) * sizeof(Vec2)},
      {reinterpret_cast<void**>(&path_), static_cast<size_t>(path_cap_) * sizeof(Pose2)},
  };
  auto align = [](size_t x) { return (x + 63) & ~static_cast<size_t>(63); };
  size_t total = 64;
  for (const Slot& s : slots) total += align(s.bytes);
  arena_.reset(new (std::nothrow) unsigned char[total]);
  if (!arena_) {
    if (err) *err = "out of memory";
    return false;
  }
  arena_size_ = total;
  auto base = reinterpret_cast<uintptr_t>(arena_.get());
  uintptr_t at = (base + 63) & ~static_cast<uintptr_t>(63);
  for (const Slot& s : slots) {
    *s.ptr = reinterpret_cast<void*>(at);
    at += align(s.bytes);
  }
  centre_window(Pose2{});
  // Round window mask, fixed in window cells (the robot is always in cell (W/2, W/2)).
  for (int c = 0; c < N_; ++c) {
    const float dx = (c % W_ - W_ / 2) * r_, dy = (c / W_ - W_ / 2) * r_;
    inside_[c] = p.window_radius <= 0.0f || std::hypot(dx, dy) <= p.window_radius;
  }
  // Headings that leave the window from each cell (fixed in window cells).
  for (int c = 0; c < N_; ++c) {
    const int x = c % W_, y = c / W_;
    uint8_t l = 0;
    for (int h = 0; h < kHeadings; ++h) {
      const int tx = x + kDX[h], ty = y + kDY[h];
      if (tx < 0 || ty < 0 || tx >= W_ || ty >= W_ || !inside_[ty * W_ + tx]) l |= 1u << h;
    }
    leave_[c] = l;
  }
  reset();
  return true;
}

void SearchFilter::reset() {
  const int32_t NS = N_ * kHeadings;
  std::fill(b_, b_ + NS, 0.0f);
  std::fill(b2_, b2_ + NS, 0.0f);
  std::fill(st_, st_ + N_, 0.0f);
  std::fill(st2_, st2_ + N_, 0.0f);
  std::fill(m_, m_ + NS, kNegInf);
  std::fill(m2_, m2_ + NS, kNegInf);
  std::fill(ms_, ms_ + N_, kNegInf);
  std::fill(ms2_, ms2_ + N_, kNegInf);
  std::fill(edge_, edge_ + NS, kEdgeBlocked);
  std::fill(emask_, emask_ + N_, 0);
  std::fill(occ_, occ_ + N_, kUnknown);
  std::fill(seen_, seen_ + N_, 0);
  std::fill(goal_, goal_ + N_, 0);
  box_b_ = box_b2_ = box_s_ = box_s2_ = CellBox{};
  mode_ = Mode::TRACKING;
  s_ = oow_ = dropped_ = 0.0f;
  phase_o_ = phase_d_ = 0.5f;  // round, not floor, the max-product position
  time_since_seen_ = time_since_loss_ = 0.0f;
  has_track_ = has_target_ = clamped_ = false;
  target_state_ = -1;
}

// ------------------------------------------------------------------------------------------------
// Window geometry

Vec2 SearchFilter::window_origin() const { return {ox_ * r_, oy_ * r_}; }

int32_t SearchFilter::cell_of(Vec2 q) const {
  const int x = static_cast<int>(std::floor(q.x / r_)) - ox_;
  const int y = static_cast<int>(std::floor(q.y / r_)) - oy_;
  if (x < 0 || y < 0 || x >= W_ || y >= W_ || !inside_[y * W_ + x]) return -1;
  return y * W_ + x;
}

// Nearest point inside the window (square, then round), for points that fall outside it.
Vec2 SearchFilter::clamp_into_window(Vec2 q) const {
  const Vec2 o = window_origin();
  q.x = std::min(std::max(q.x, o.x + 0.5f * r_), o.x + (W_ - 0.5f) * r_);
  q.y = std::min(std::max(q.y, o.y + 0.5f * r_), o.y + (W_ - 0.5f) * r_);
  if (p_.window_radius > 0.0f) {
    const Vec2 cc = cell_centre((W_ / 2) * W_ + W_ / 2);
    const float d = std::hypot(q.x - cc.x, q.y - cc.y), rmax = p_.window_radius - r_;
    if (d > rmax) q = Vec2{cc.x + (q.x - cc.x) * rmax / d, cc.y + (q.y - cc.y) * rmax / d};
  }
  return q;
}

Vec2 SearchFilter::cell_centre(int32_t c) const {
  return {(ox_ + c % W_ + 0.5f) * r_, (oy_ + c / W_ + 0.5f) * r_};
}

CellBox SearchFilter::expand(const CellBox& b, int k) const {
  if (b.empty()) return b;
  return {std::max(0, b.x0 - k), std::max(0, b.y0 - k), std::min(W_ - 1, b.x1 + k),
          std::min(W_ - 1, b.y1 + k)};
}

// f(c, x, y) for the window cells in r: row by row, or in the test permutation (order_).
template <class F>
void SearchFilter::for_cells(const CellBox& r, F&& f) const {
  if (r.empty()) return;
  if (order_) {
    for (int i = 0; i < N_; ++i) {
      const int c = order_[i], x = c % W_, y = c / W_;
      if (x < r.x0 || x > r.x1 || y < r.y0 || y > r.y1 || !inside_[c]) continue;
      f(c, x, y);
    }
    return;
  }
  for (int y = r.y0; y <= r.y1; ++y)
    for (int x = r.x0; x <= r.x1; ++x)
      if (inside_[y * W_ + x]) f(y * W_ + x, x, y);
}

// Support boxes of the read buffers from their values (after whole-buffer writes).
void SearchFilter::boxes_from_values() {
  BoxAcc ab, as;
  for (int c = 0; c < N_; ++c) {
    const int x = c % W_, y = c / W_;
    for (int h = 0; h < kHeadings; ++h) {
      if (b_[c * kHeadings + h] != 0.0f || m_[c * kHeadings + h] != kNegInf) {
        ab.add(x, y);
        break;
      }
    }
    if (st_[c] != 0.0f || ms_[c] != kNegInf) as.add(x, y);
  }
  box_b_ = ab.box();
  box_s_ = as.box();
}

void SearchFilter::belief_marginal(float* out) const {
  for (int c = 0; c < N_; ++c) {
    float sum = st_[c];
    for (int h = 0; h < kHeadings; ++h) sum += b_[c * kHeadings + h];
    out[c] = sum;
  }
}

float SearchFilter::max_cell_mass() const {
  float best = 0.0f;
  const CellBox r = region(unite(box_b_, box_s_));
  for (int y = std::max(0, r.y0); y <= r.y1; ++y) {
    for (int x = r.x0; x <= r.x1; ++x) {
      const int c = y * W_ + x;
      float sum = st_[c];
      for (int h = 0; h < kHeadings; ++h) sum += b_[c * kHeadings + h];
      best = std::max(best, sum);
    }
  }
  return best;
}

void SearchFilter::swap_all() {
  std::swap(b_, b2_);
  std::swap(st_, st2_);
  std::swap(m_, m2_);
  std::swap(ms_, ms2_);
  std::swap(box_b_, box_b2_);
  std::swap(box_s_, box_s2_);
}

void SearchFilter::centre_window(const Pose2& robot) {
  ox_ = static_cast<int32_t>(std::floor(robot.x / r_)) - W_ / 2;
  oy_ = static_cast<int32_t>(std::floor(robot.y / r_)) - W_ / 2;
}

// Spec tick 1: whole-cell shift so the robot stays in the centre cell. Mass that falls off goes
// to out_of_window. Pull: new(x, y) = old(x + dx, y + dy).
void SearchFilter::shift_window(const Pose2& robot) {
  const int dx = static_cast<int>(std::floor(robot.x / r_)) - W_ / 2 - ox_;
  const int dy = static_cast<int>(std::floor(robot.y / r_)) - W_ / 2 - oy_;
  if (dx == 0 && dy == 0) return;
  auto inside = [&](int x, int y) {
    return x >= 0 && y >= 0 && x < W_ && y < W_ && inside_[y * W_ + x];
  };
  // Reduction in natural order over the supports: old cells that leave the window.
  const CellBox old = region(unite(box_b_, box_s_));
  for (int y = std::max(0, old.y0); y <= old.y1; ++y) {
    for (int x = old.x0; x <= old.x1; ++x) {
      if (inside(x - dx, y - dy)) continue;
      const int c = y * W_ + x;
      float sum = st_[c];
      for (int h = 0; h < kHeadings; ++h) sum += b_[c * kHeadings + h];
      oow_ += sum;
    }
  }
  // New cells that can hold mass (the old support moved by -d) plus stale data in the write side.
  CellBox moved = old;
  if (!moved.empty()) {
    moved = CellBox{std::max(0, old.x0 - dx), std::max(0, old.y0 - dy),
                    std::min(W_ - 1, old.x1 - dx), std::min(W_ - 1, old.y1 - dy)};
  }
  const CellBox r = region(unite(moved, unite(box_b2_, box_s2_)));
  BoxAcc acc_b, acc_s;
  for_cells(r, [&](int c, int x, int y) {
    const int sx = x + dx, sy = y + dy;
    const bool in = inside(sx, sy);  // never move mass onto the window corners (c is inside)
    const int s = sy * W_ + sx;
    bool any = false;
    for (int h = 0; h < kHeadings; ++h) {
      b2_[c * kHeadings + h] = in ? b_[s * kHeadings + h] : 0.0f;
      m2_[c * kHeadings + h] = in ? m_[s * kHeadings + h] : kNegInf;
      any |= b2_[c * kHeadings + h] != 0.0f || m2_[c * kHeadings + h] != kNegInf;
    }
    st2_[c] = in ? st_[s] : 0.0f;
    ms2_[c] = in ? ms_[s] : kNegInf;
    if (any) acc_b.add(x, y);
    if (st2_[c] != 0.0f || ms2_[c] != kNegInf) acc_s.add(x, y);
  });
  for (int c = 0; c < N_; ++c) {  // occupancy moves with the window (bytes, cheap)
    const int x = c % W_ + dx, y = c / W_ + dy;
    blocked_[c] = x >= 0 && y >= 0 && x < W_ && y < W_ ? occ_[y * W_ + x]
                                                       : static_cast<uint8_t>(kUnknown);
  }
  std::memcpy(occ_, blocked_, static_cast<size_t>(N_));
  const CellBox old_b = box_b_, old_s = box_s_;
  box_b2_ = acc_b.box();
  box_s2_ = acc_s.box();
  swap_all();
  // The other side still holds the old window's values inside the old supports.
  box_b2_ = old_b;
  box_s2_ = old_s;
  if (!skips_) box_b2_ = box_s2_ = full_box();
  ox_ += dx;
  oy_ += dy;
}

// Raycasts the scan: cells along a beam are free, the hit cell occupied. Occupied wins over free,
// so the result does not depend on beam order. Cells the scan does not reach are unknown, or with
// occ_memory keep their last observed state (deviation: the lidar has blind sectors, and walls
// that drop out of one scan would otherwise open for a tick and let the belief leak through).
void SearchFilter::rasterize(const Pose2& robot, const ScanView* scan) {
  if (!scan || !scan->ranges) return;  // no new scan: keep the last one (shifted)
  uint8_t* cur = blocked_;  // scratch: this scan only
  std::fill(cur, cur + N_, kUnknown);
  const Pose2& sb = scan->sensor_in_base;
  const float cy = std::cos(robot.yaw), sy = std::sin(robot.yaw);
  const float sxw = robot.x + cy * sb.x - sy * sb.y;
  const float syw = robot.y + sy * sb.x + cy * sb.y;
  const float gx0 = sxw / r_ - ox_, gy0 = syw / r_ - oy_;
  for (int i = 0; i < scan->count; ++i) {
    const float range = scan->ranges[i];
    if (std::isnan(range) || range < scan->range_min) continue;
    const bool hit = std::isfinite(range) && range < scan->range_max;
    const float len = hit ? range : scan->range_max;
    const float a = robot.yaw + sb.yaw + scan->angle_min + i * scan->angle_increment;
    const float gx1 = gx0 + len * std::cos(a) / r_, gy1 = gy0 + len * std::sin(a) / r_;
    traverse(gx0, gy0, gx1, gy1, W_, [&](int c) {
      if (cur[c] != kOccupied) cur[c] = kFree;
      return true;
    });
    if (hit) {
      const int hx = static_cast<int>(std::floor(gx1)), hy = static_cast<int>(std::floor(gy1));
      if (hx >= 0 && hy >= 0 && hx < W_ && hy < W_) cur[hy * W_ + hx] = kOccupied;
    }
  }
  for (int c = 0; c < N_; ++c)
    if (cur[c] != kUnknown || !p_.occ_memory) occ_[c] = cur[c];
}

// Edge (c, h) is open when c + d(h) is inside, not occupied and, for diagonals, both orthogonal
// neighbours are not occupied. Unknown counts as free. Edges out of an occupied cell are closed
// too (deviation: otherwise mass inside a wall cell leaks through it).
void SearchFilter::build_edge_mask() { build_edge_mask(full_box()); }

// Only cells near the support need edges: elsewhere there is no mass to move.
void SearchFilter::build_edge_mask(const CellBox& r) {
  // One byte per cell: bit h = edge (c, h) open. nb = occupied neighbours (square window).
  for_cells(r, [&](int c, int x, int y) {
    if (occ_[c] == kOccupied) {
      emask_[c] = 0;  // the leader cannot be inside a wall: its mass drains away
      return;
    }
    unsigned nb = 0;
    for (int h = 0; h < kHeadings; ++h) {
      const int tx = x + kDX[h], ty = y + kDY[h];
      if (tx >= 0 && ty >= 0 && tx < W_ && ty < W_ && occ_[ty * W_ + tx] == kOccupied) nb |= 1u << h;
    }
    const unsigned rot_l = ((nb << 1) | (nb >> 7)) & 0xFFu;  // bit h = nb bit h-1
    const unsigned rot_r = ((nb >> 1) | (nb << 7)) & 0xFFu;  // bit h = nb bit h+1
    // Orthogonal: target occupied. Diagonal: target or either orthogonal neighbour occupied.
    const unsigned blocked = (nb & 0x55u) | ((nb | rot_l | rot_r) & 0xAAu);
    emask_[c] = static_cast<uint8_t>(~(blocked | leave_[c]) & 0xFFu);
  });
}

const uint8_t* SearchFilter::edge_open() const {
  for (int c = 0; c < N_; ++c) {
    for (int h = 0; h < kHeadings; ++h) {
      uint8_t e = kEdgeBlocked;
      if ((emask_[c] >> h) & 1u)
        e = kEdgeOpen;
      else if (occ_[c] != kOccupied && ((leave_[c] >> h) & 1u))
        e = kEdgeLeaves;
      edge_[c * kHeadings + h] = e;
    }
  }
  return edge_;
}

void SearchFilter::debug_set_occ(const uint8_t* occ) {
  std::memcpy(occ_, occ, static_cast<size_t>(N_));
  build_edge_mask();
}

// ------------------------------------------------------------------------------------------------
// On loss

void SearchFilter::on_loss(const TrackInit& t, const Pose2& robot, const ScanView* scan) {
  // Without a scan the last occupancy is kept if the window does not move.
  const int32_t ox = ox_, oy = oy_;
  std::memcpy(blocked_, occ_, static_cast<size_t>(N_));
  reset();
  centre_window(robot);
  if (!scan && ox == ox_ && oy == oy_) std::memcpy(occ_, blocked_, static_cast<size_t>(N_));
  rasterize(robot, scan);
  build_edge_mask();

  // Clamp p into the window (warn via SearchOutput::clamped).
  clamped_ = cell_of(t.p) < 0;
  const Vec2 p = clamped_ ? clamp_into_window(t.p) : t.p;

  // s = clamp(|v|, s_min, s_max), theta = atan2(v_y, v_x).
  const float vn = std::hypot(t.v.x, t.v.y);
  s_ = std::min(std::max(vn, p_.s_min), p_.s_max);
  const float theta = std::atan2(t.v.y, t.v.x);
  float wh[kHeadings], whs = 0.0f;
  for (int h = 0; h < kHeadings; ++h) {
    const float d = wrap(h * kDTheta - theta);
    wh[h] = vn > 1e-3f ? std::exp(-d * d / (2.0f * p_.sigma_theta0 * p_.sigma_theta0)) : 1.0f;
    whs += wh[h];
  }
  // Cells the leader reaches from p only by going around a wall get no b0 (deviation: otherwise
  // the Gaussian spills through walls, and mass behind a wall is never observed and never
  // decays). Geodesic distance from p's cell over non-occupied cells vs the straight line.
  // N(c; p, Sigma + r^2 I), cut at 5 sigma.
  const float a = t.cov[0] + r_ * r_, b = 0.5f * (t.cov[1] + t.cov[2]), d = t.cov[3] + r_ * r_;
  const float sigma_max = std::sqrt(0.5f * (a + d) + std::hypot(0.5f * (a - d), b));
  plan_dijkstra(cell_of(p), nullptr, 0, 1.1f * 5.0f * sigma_max + r_);
  const float det = std::max(a * d - b * b, 1e-9f);
  const float ia = d / det, ib = -b / det, id = a / det;
  float gs = 0.0f;
  for (int c = 0; c < N_; ++c) {
    const Vec2 q = cell_centre(c);
    const float ex = q.x - p.x, ey = q.y - p.y;
    float g = std::exp(-0.5f * (ia * ex * ex + 2.0f * ib * ex * ey + id * ey * ey));
    if (occ_[c] == kOccupied || !inside_[c]) g = 0.0f;  // not inside walls or off the window
    if (!(dist_[c] <= 1.1f * std::hypot(ex, ey) + r_)) g = 0.0f;  // only around a wall
    st2_[c] = g;  // scratch
    gs += g;
  }
  if (!(gs > 0.0f)) {  // degenerate: everything on p's cell
    std::fill(st2_, st2_ + N_, 0.0f);
    st2_[cell_of(p)] = 1.0f;
    gs = 1.0f;
  }
  const float ps = !p_.stopped_enabled ? 0.0f : (vn < p_.v_stop ? 0.5f : p_.p_stop0);
  for (int c = 0; c < N_; ++c) {
    const float g = st2_[c] / gs;
    for (int h = 0; h < kHeadings; ++h) {
      const float v = cut_b_((1.0f - ps) * g * wh[h] / whs);
      b_[c * kHeadings + h] = v;
      m_[c * kHeadings + h] = cut_m(safe_log(v));  // m0 = log b0
    }
    st_[c] = cut_b_(ps * g);
    ms_[c] = cut_m(safe_log(st_[c]));
  }
  std::fill(st2_, st2_ + N_, 0.0f);
  boxes_from_values();
  box_b2_ = box_s2_ = CellBox{};  // reset() cleared the other buffers
  oow_ = 0.0f;
  mode_ = Mode::SEARCH;
  last_track_ = t;
  last_seen_ = t.p;
  has_track_ = true;
}

// ------------------------------------------------------------------------------------------------
// Predict (spec tick 2-4): n_sub substeps of heading diffusion + move, on b and on m.

void SearchFilter::predict(float dt) {
  const int n_sub = std::max(1, static_cast<int>(std::ceil(s_ * dt / r_)));
  const float dts = dt / n_sub;
  TickConst k;
  k.q = std::min(0.25f, p_.sigma_h * p_.sigma_h * dts / (2.0f * kDTheta * kDTheta));
  k.l_stay_h = safe_log(1.0f - 2.0f * k.q);
  k.l_q = safe_log(k.q);
  for (int h = 0; h < kHeadings; ++h) {
    const float L = (h % 2 == 0) ? r_ : r_ * std::sqrt(2.0f);
    k.pm[h] = std::min(1.0f, s_ * dts / L);
  }
  k.a = p_.stopped_enabled ? std::min(1.0f, p_.lambda_stop * dts) : 0.0f;
  k.l_a = safe_log(k.a);
  k.l_1ma = safe_log(1.0f - k.a);
  k.g = p_.stopped_enabled ? std::min(1.0f, p_.lambda_go * dts) : 0.0f;
  for (int n = 0; n <= kHeadings; ++n) {
    k.go[n] = n > 0 ? k.g / n : 0.0f;
    k.l_go[n] = safe_log(k.go[n]);
    k.g_eff[n] = n > 0 ? k.g : 0.0f;
    k.l_1mg[n] = safe_log(1.0f - k.g_eff[n]);
  }
  // Max-product moves at exactly speed s (deviation, see move()): a phase per move length L
  // gains s*dt_s/L each substep and the headings of that length advance one cell when it wraps.
  const float adv_o = s_ * dts / r_, adv_d = adv_o / std::sqrt(2.0f);
  for (int i = 0; i < n_sub; ++i) {
    phase_o_ += adv_o;
    phase_d_ += adv_d;
    const bool mo = phase_o_ >= 1.0f, md = phase_d_ >= 1.0f;
    if (mo) phase_o_ -= 1.0f;
    if (md) phase_d_ -= 1.0f;
    for (int h = 0; h < kHeadings; ++h) k.mp_move[h] = (h % 2 == 0) ? mo : md;
    int64_t t0 = 0;
    stage_begin(&t0);
    heading_diffusion(k);
    stage_end(kStDiffuse, &t0);
    move(k);
    stage_end(kStMove, &t0);
  }
}

// b'(c,h) = (1-2q) b(c,h) + q b(c,h-1) + q b(c,h+1); m' = max of the logs. SIMD: the 8 headings
// of a cell are rotated in two 4-lane vectors. Runs over box_b_ and box_b2_ (stale values).
void SearchFilter::heading_diffusion(const TickConst& k) {
  const v4f qs = splat(1.0f - 2.0f * k.q), q = splat(k.q);
  const v4f ls = splat(k.l_stay_h), lq = splat(k.l_q), ninf = splat(kNegInf), zero = splat(0.0f);
  BoxAcc acc;
  for_cells(region(unite(box_b_, box_b2_)), [&](int c, int x, int y) {
    const int base = c * kHeadings;
    const v4f blo = ld4(b_ + base), bhi = ld4(b_ + base + 4);
    const v4f mlo = ld4(m_ + base), mhi = ld4(m_ + base + 4);
    const v4f nblo = qs * blo + q * prev_lo(blo, bhi) + q * next_lo(blo, bhi);
    const v4f nbhi = qs * bhi + q * prev_hi(blo, bhi) + q * next_hi(blo, bhi);
    const v4f nmlo = vmax(vmax(ls + mlo, lq + prev_lo(mlo, mhi)), lq + next_lo(mlo, mhi));
    const v4f nmhi = vmax(vmax(ls + mhi, lq + prev_hi(mlo, mhi)), lq + next_hi(mlo, mhi));
    st4(b2_ + base, nblo);
    st4(b2_ + base + 4, nbhi);
    st4(m2_ + base, nmlo);
    st4(m2_ + base + 4, nmhi);
    if (any_lane((nblo != zero) | (nbhi != zero) | (nmlo != ninf) | (nmhi != ninf))) acc.add(x, y);
  });
  box_b2_ = acc.box();
  std::swap(b_, b2_);
  std::swap(m_, m2_);
  std::swap(box_b_, box_b2_);
}

// Move (pull): with bm = (1-a) b' (a = stop fraction),
//   b''(c,h) = (1-p_m) bm(c,h) + p_m bm(c-d(h),h) open(c-d(h),h) + redirect + go share of stopped.
// REDIRECT pulls p_m bm(c,g) from closed g = h-+1 into open h, split over g's open neighbours.
// stopped''(c) = (1-g) stopped(c) + a sum_h b'(c,h).
// Max-product (deviation from the spec): with the stay/move split the never-moving path is always
// the single most likely one (0.8^n > anything that moves), so argmax(m) would stay at the loss
// point. Instead m moves deterministically at speed s: when mp_move[h], m''(c,h) = m'(c-d(h),h)
// (closed edge: -inf, or REDIRECT from closed h+-1 at the same cell), otherwise m''(c,h) = m'(c,h).
// Stop/go and log(1-a) are applied as for b.
// SIMD over the 8 headings of a cell (interior cells, DROP); the window border and REDIRECT use the
// scalar path. Both are pure functions of the cell, so the result does not depend on cell order.
void SearchFilter::move(const TickConst& k) {
  const bool redirect = p_.blocked_mode == BlockedMode::REDIRECT;
  auto n_redirect = [&](uint8_t mask, int g) {
    return int((mask >> hm1(g)) & 1u) + int((mask >> hp1(g)) & 1u);
  };
  const float keep = 1.0f - k.a;

  // Accounting (natural order, cells with mass): mass that leaves the window or is dropped.
  const CellBox acc_r = region(box_b_);
  for (int y = std::max(0, acc_r.y0); y <= acc_r.y1; ++y) {
    for (int x = acc_r.x0; x <= acc_r.x1; ++x) {
      const int c = y * W_ + x;
      if (!inside_[c]) continue;
      const uint8_t open = emask_[c], leaves = occ_[c] == kOccupied ? 0 : leave_[c];
      if (open == 0xFF) continue;
      for (int h = 0; h < kHeadings; ++h) {
        if ((open >> h) & 1u) continue;
        const float moved = k.pm[h] * keep * b_[c * kHeadings + h];
        if ((leaves >> h) & 1u) {
          oow_ += moved;
        } else if (!redirect || n_redirect(open, h) == 0) {
          dropped_ += moved;
        }
      }
    }
  }

  // Per-substep vectors.
  v4f stay_lo, stay_hi, pull_lo, pull_hi;  // (1 - p_m) and p_m * keep per heading
  for (int h = 0; h < 4; ++h) {
    stay_lo[h] = 1.0f - k.pm[h];
    stay_hi[h] = 1.0f - k.pm[h + 4];
    pull_lo[h] = k.pm[h] * keep;
    pull_hi[h] = k.pm[h + 4] * keep;
  }
  const v4i mv_lo = v4i{k.mp_move[0], k.mp_move[1], k.mp_move[2], k.mp_move[3]} != 0;
  const v4i mv_hi = v4i{k.mp_move[4], k.mp_move[5], k.mp_move[6], k.mp_move[7]} != 0;
  const v4f vkeep = splat(keep), l1ma = splat(k.l_1ma), ninf = splat(kNegInf), zero = splat(0.0f);
  const v4f tail = splat(p_.tail_cut), minlog = splat(kMinLog);
  int src_off[kHeadings];  // state offset of c - d(h), heading h
  for (int h = 0; h < kHeadings; ++h) src_off[h] = -(kDY[h] * W_ + kDX[h]) * kHeadings + h;

  // Reach one cell (pulls from c - d(h)); also clear the stale write boxes.
  const CellBox r =
      region(unite(unite(expand(box_b_, 1), box_s_), unite(box_b2_, box_s2_)));
  BoxAcc acc_b, acc_s;
  for_cells(r, [&](int c, int x, int y) {
    const uint8_t open = emask_[c];
    const int n_open = __builtin_popcount(open);
    const float g_eff = k.g_eff[n_open], go = k.go[n_open], l_go = k.l_go[n_open];
    const float* b = b_ + c * kHeadings;
    const float* m = m_ + c * kHeadings;
    float bsum, mmax;
    bool any_b = false;
    const bool interior = x > 0 && y > 0 && x < W_ - 1 && y < W_ - 1;
    if (interior && !redirect) {
      const v4f blo = ld4(b), bhi = ld4(b + 4), mlo = ld4(m), mhi = ld4(m + 4);
      // Gather the pulled states b(c - d(h), h), m(c - d(h), h) and whether that edge is open.
      float gb[kHeadings], gm[kHeadings];
      unsigned in = 0;
      const int base = c * kHeadings;
      for (int h = 0; h < kHeadings; ++h) {
        gb[h] = b_[base + src_off[h]];
        gm[h] = m_[base + src_off[h]];
        in |= ((emask_[c - kDY[h] * W_ - kDX[h]] >> h) & 1u) << h;
      }
      const v4i in_lo = bits_lo(in), in_hi = bits_hi(in), op_lo = bits_lo(open), op_hi = bits_hi(open);
      const v4f vgo = splat(go * st_[c]), vlgo = splat(l_go + ms_[c]);
      v4f nblo = stay_lo * (vkeep * blo), nbhi = stay_hi * (vkeep * bhi);
      nblo += in_lo ? pull_lo * ld4(gb) : zero;
      nbhi += in_hi ? pull_hi * ld4(gb + 4) : zero;
      nblo += op_lo ? vgo : zero;
      nbhi += op_hi ? vgo : zero;
      v4f nmlo = mv_lo ? (in_lo ? l1ma + ld4(gm) : ninf) : l1ma + mlo;
      v4f nmhi = mv_hi ? (in_hi ? l1ma + ld4(gm + 4) : ninf) : l1ma + mhi;
      nmlo = op_lo ? vmax(nmlo, vlgo) : nmlo;
      nmhi = op_hi ? vmax(nmhi, vlgo) : nmhi;
      nblo = nblo < tail ? zero : nblo;
      nbhi = nbhi < tail ? zero : nbhi;
      nmlo = nmlo < minlog ? ninf : nmlo;
      nmhi = nmhi < minlog ? ninf : nmhi;
      st4(b2_ + base, nblo);
      st4(b2_ + base + 4, nbhi);
      st4(m2_ + base, nmlo);
      st4(m2_ + base + 4, nmhi);
      any_b = any_lane((nblo != zero) | (nbhi != zero) | (nmlo != ninf) | (nmhi != ninf));
      bsum = hsum(blo + bhi);
      mmax = hmax(vmax(mlo, mhi));
    } else {
      bsum = 0.0f;
      mmax = kNegInf;
      for (int h = 0; h < kHeadings; ++h) {
        float nb = (1.0f - k.pm[h]) * (keep * b[h]);
        float nm = k.mp_move[h] ? kNegInf : k.l_1ma + m[h];
        const int sx = x - kDX[h], sy = y - kDY[h];
        if (sx >= 0 && sy >= 0 && sx < W_ && sy < W_) {
          const int s = sy * W_ + sx;
          if ((emask_[s] >> h) & 1u) {
            nb += k.pm[h] * keep * b_[s * kHeadings + h];
            if (k.mp_move[h]) nm = k.l_1ma + m_[s * kHeadings + h];
          }
        }
        if ((open >> h) & 1u) {
          if (redirect) {
            for (int g : {hm1(h), hp1(h)}) {
              if (((open >> g) & 1u) || ((leave_[c] >> g) & 1u && occ_[c] != kOccupied)) continue;
              const float share = k.pm[g] / n_redirect(open, g);
              nb += share * keep * b[g];
              if (k.mp_move[g])
                nm = std::max(nm, safe_log(1.0f / n_redirect(open, g)) + k.l_1ma + m[g]);
            }
          }
          nb += go * st_[c];
          nm = std::max(nm, l_go + ms_[c]);
        }
        nb = cut_b_(nb);
        nm = cut_m(nm);
        b2_[c * kHeadings + h] = nb;
        m2_[c * kHeadings + h] = nm;
        any_b |= nb != 0.0f || nm != kNegInf;
      }
      bsum = (b[0] + b[1]) + (b[2] + b[3]) + ((b[4] + b[5]) + (b[6] + b[7]));
      for (int h = 0; h < kHeadings; ++h) mmax = std::max(mmax, m[h]);
    }
    const float ns = cut_b_((1.0f - g_eff) * st_[c] + k.a * bsum);
    const float nms = cut_m(std::max(k.l_1mg[n_open] + ms_[c], k.l_a + mmax));
    st2_[c] = ns;
    ms2_[c] = nms;
    if (any_b) acc_b.add(x, y);
    if (ns != 0.0f || nms != kNegInf) acc_s.add(x, y);
  });
  box_b2_ = acc_b.box();
  box_s2_ = acc_s.box();
  swap_all();
}

// ------------------------------------------------------------------------------------------------
// Observe (spec tick 5): raycast the D455 FOV through occ; seen cells without a re-ID match are
// scaled by (1 - p_d[range bin]), m by log(1 - p_d). Occluded/out-of-FOV cells are copied.

// Marks the cells the D455 sees from robot: seen_[c] = 1 + range bin, 0 = not seen. Rays stop at
// occupied cells.
void SearchFilter::observe_rays(const Pose2& robot) {
  std::fill(seen_, seen_ + N_, 0);
  const float yaw = robot.yaw + p_.cam_yaw;
  const float gx0 = robot.x / r_ - ox_, gy0 = robot.y / r_ - oy_;
  const float step = 0.5f * r_ / p_.cam_range;  // half a cell at full range
  const int n_rays = static_cast<int>(std::ceil(p_.cam_fov / step)) + 1;
  for (int i = 0; i < n_rays; ++i) {
    const float a = yaw - 0.5f * p_.cam_fov + p_.cam_fov * i / (n_rays - 1);
    const float gx1 = gx0 + p_.cam_range * std::cos(a) / r_;
    const float gy1 = gy0 + p_.cam_range * std::sin(a) / r_;
    traverse(gx0, gy0, gx1, gy1, W_, [&](int c) {
      if (occ_[c] == kOccupied) return false;
      if (seen_[c]) return true;
      const Vec2 q = cell_centre(c);
      const float d = std::hypot(q.x - robot.x, q.y - robot.y);
      if (d <= p_.cam_range) seen_[c] = static_cast<uint8_t>(1 + std::min(kPdBins - 1, int(d)));
      return true;
    });
  }
}

// Factor (1 - p_d) and its log per seen_ value (index 0 = not seen: 1 and 0, exact no-ops).
void SearchFilter::observation_factors(float* f, float* lf) const {
  f[0] = 1.0f;
  lf[0] = 0.0f;
  for (int k = 0; k < kPdBins; ++k) {
    f[k + 1] = 1.0f - p_.p_d[k];
    lf[k + 1] = std::log(f[k + 1]);
  }
}

// Observation update alone (tests): b *= 1 - p_d, m += log(1 - p_d) on seen cells.
void SearchFilter::observe(const Pose2& robot) {
  observe_rays(robot);
  float fac[kPdBins + 1], lfac[kPdBins + 1];
  observation_factors(fac, lfac);
  BoxAcc acc_b, acc_s;
  const CellBox r = region(unite(unite(box_b_, box_b2_), unite(box_s_, box_s2_)));
  for_cells(r, [&](int c, int x, int y) {
    const float f = fac[seen_[c]], lf = lfac[seen_[c]];
    bool any = false;
    for (int h = 0; h < kHeadings; ++h) {
      const int j = c * kHeadings + h;
      b2_[j] = b_[j] * f;
      m2_[j] = m_[j] + lf;
      any |= b2_[j] != 0.0f || m2_[j] != kNegInf;
    }
    st2_[c] = st_[c] * f;
    ms2_[c] = ms_[c] + lf;
    if (any) acc_b.add(x, y);
    if (st2_[c] != 0.0f || ms2_[c] != kNegInf) acc_s.add(x, y);
  });
  box_b2_ = acc_b.box();
  box_s2_ = acc_s.box();
  swap_all();
}

// Spec tick 5 + 6 in one write pass: observation factor (cam = null: none), then divide b, stopped
// and out_of_window by their total and shift m so max(m) = 0. The reduction (natural order over
// the supports) also finds the argmax of m for select_target.
void SearchFilter::observe_normalise(const Pose2* cam) {
  if (cam) {
    observe_rays(*cam);
  } else {
    std::fill(seen_, seen_ + N_, 0);
  }
  float fac[kPdBins + 1], lfac[kPdBins + 1];
  observation_factors(fac, lfac);
  const int NS = N_ * kHeadings;
  float total = oow_, mmax = kNegInf;
  int best = -1;
  const CellBox rb = region(box_b_), rs = region(box_s_);
  for (int y = std::max(0, rb.y0); y <= rb.y1; ++y) {
    for (int c = y * W_ + rb.x0; c <= y * W_ + rb.x1; ++c) {
      const v4f f = splat(fac[seen_[c]]), lf = splat(lfac[seen_[c]]);
      const int j = c * kHeadings;
      const v4f mlo = ld4(m_ + j) + lf, mhi = ld4(m_ + j + 4) + lf;
      total += hsum(ld4(b_ + j) * f + ld4(b_ + j + 4) * f);
      if (hmax(vmax(mlo, mhi)) > mmax) {  // rare: find the first maximal heading
        for (int h = 0; h < kHeadings; ++h) {
          const float mv = h < 4 ? mlo[h] : mhi[h - 4];
          if (mv > mmax) mmax = mv, best = j + h;
        }
      }
    }
  }
  for (int y = std::max(0, rs.y0); y <= rs.y1; ++y) {
    for (int c = y * W_ + rs.x0; c <= y * W_ + rs.x1; ++c) {
      total += st_[c] * fac[seen_[c]];
      const float mv = ms_[c] + lfac[seen_[c]];
      if (mv > mmax) mmax = mv, best = NS + c;
    }
  }
  last_total_ = total;
  best_state_ = best;
  const bool scale = total > 0.0f && std::isfinite(total);
  const float shift = std::isfinite(mmax) ? mmax : 0.0f;
  BoxAcc acc_b, acc_s;
  const CellBox r = region(unite(unite(box_b_, box_b2_), unite(box_s_, box_s2_)));
  const v4f inv = splat(scale ? 1.0f / total : 1.0f), vshift = splat(shift);
  const v4f zero = splat(0.0f), ninf = splat(kNegInf);
  for_cells(r, [&](int c, int x, int y) {
    const float fs = fac[seen_[c]];
    const v4f f = splat(fs), lf = splat(lfac[seen_[c]]);
    const int j = c * kHeadings;
    const v4f blo = ld4(b_ + j) * f * inv, bhi = ld4(b_ + j + 4) * f * inv;
    const v4f mlo = (ld4(m_ + j) + lf) - vshift, mhi = (ld4(m_ + j + 4) + lf) - vshift;
    st4(b2_ + j, blo);
    st4(b2_ + j + 4, bhi);
    st4(m2_ + j, mlo);
    st4(m2_ + j + 4, mhi);
    const bool any = any_lane((blo != zero) | (bhi != zero) | (mlo != ninf) | (mhi != ninf));
    const float lfs = lfac[seen_[c]];
    const float v = st_[c] * fs;
    st2_[c] = scale ? v * (1.0f / total) : v;
    ms2_[c] = (ms_[c] + lfs) - shift;
    if (any) acc_b.add(x, y);
    if (st2_[c] != 0.0f || ms2_[c] != kNegInf) acc_s.add(x, y);
  });
  box_b2_ = acc_b.box();
  box_s2_ = acc_s.box();
  if (scale) oow_ /= total;
  swap_all();
}

const char* SearchFilter::stage_name(int s) {
  static const char* names[kNumStages] = {"shift", "rasterize", "edge mask", "heading diffusion",
                                          "move", "observe + normalise", "-", "target", "dijkstra",
                                          "intercept", "path"};
  return s >= 0 && s < kNumStages ? names[s] : "?";
}

void SearchFilter::stage_begin(int64_t* t0) const {
  if (timing_) *t0 = std::chrono::steady_clock::now().time_since_epoch().count();
}

void SearchFilter::stage_end(int stage, int64_t* t0) {
  if (!timing_) return;
  const int64_t t1 = std::chrono::steady_clock::now().time_since_epoch().count();
  stage_us_[stage] += std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::duration(t1 - *t0)).count();
  *t0 = t1;
}

void SearchFilter::filter_tick(const Pose2& robot, const ScanView* scan,
                               const Observation* looked, float dt, SearchOutput* out) {
  int64_t t0 = 0;
  stage_begin(&t0);
  dropped_ = 0.0f;
  shift_window(robot);
  stage_end(kStShift, &t0);
  rasterize(robot, scan);
  stage_end(kStRaster, &t0);
  const int n_sub = std::max(1, static_cast<int>(std::ceil(s_ * dt / r_)));
  build_edge_mask(region(expand(unite(box_b_, box_s_), n_sub + 1)));
  stage_end(kStEdges, &t0);
  predict(dt);  // times move itself
  stage_begin(&t0);
  observe_normalise(looked ? &looked->robot : nullptr);
  if (looked && out) out->looked = true;
  stage_end(kStObserve, &t0);
}

void SearchFilter::debug_filter_tick(const Pose2& robot, const ScanView* scan,
                                     const Observation* looked, float dt) {
  filter_tick(robot, scan, looked, dt, nullptr);
}

// ------------------------------------------------------------------------------------------------
// Target (spec tick 7): argmax of m (moving and stopped states) with hysteresis.

bool SearchFilter::select_target(SearchOutput* out) {
  const int NS = N_ * kHeadings;
  // Argmax of m: found by observe_normalise() in its reduction (natural order, first maximum).
  const int best = best_state_;
  if (best < 0) return false;
  const float best_score = best < NS ? m_[best] : ms_[best - NS];
  auto cell_of_state = [&](int j) { return j < NS ? j / kHeadings : j - NS; };

  int chosen = best;
  bool switched = false;
  const int pc = has_target_ ? cell_of(target_) : -1;
  if (pc >= 0 && !seen_[pc]) {
    // Best state within rho of the previous target.
    int near = -1;
    float near_score = kNegInf;
    const int rc = static_cast<int>(std::ceil(p_.rho / r_));
    const int px = pc % W_, py = pc / W_;
    for (int y = std::max(0, py - rc); y <= std::min(W_ - 1, py + rc); ++y) {
      for (int x = std::max(0, px - rc); x <= std::min(W_ - 1, px + rc); ++x) {
        const int c = y * W_ + x;
        const Vec2 q = cell_centre(c);
        if (std::hypot(q.x - target_.x, q.y - target_.y) > p_.rho + 1e-4f) continue;
        for (int h = 0; h < kHeadings; ++h)
          if (m_[c * kHeadings + h] > near_score) near_score = m_[c * kHeadings + h], near = c * kHeadings + h;
        if (ms_[c] > near_score) near_score = ms_[c], near = NS + c;
      }
    }
    if (near >= 0 && !(best_score > near_score + p_.delta_log)) {
      chosen = near;
    } else {
      switched = true;
    }
  } else {
    switched = has_target_;
  }
  if (switched) {
    const Vec2 q = cell_centre(cell_of_state(chosen));
    switched = std::hypot(q.x - target_.x, q.y - target_.y) > p_.rho;
  }
  target_state_ = chosen;
  target_ = cell_centre(cell_of_state(chosen));
  has_target_ = true;
  out->target = target_;
  out->target_heading = chosen < NS ? static_cast<int8_t>(chosen % kHeadings) : -1;
  out->target_switched = switched;
  return true;
}

// ------------------------------------------------------------------------------------------------
// Robot-side planning (spec tick 8-9).

// 8-connected Dijkstra over blocked_ (occ inflated by inflate_cells), cost in metres. Entering a
// cell next to a blocked one costs wall_cost more (deviation: soft clearance; hard inflation by a
// whole 0.5 m cell would close doors).
void SearchFilter::plan_dijkstra(int32_t start, const int32_t* goals, int32_t n_goals,
                                 float max_dist) {
  const int k = p_.inflate_cells;
  // wall_[c] = blocked and inside the window (the window rim is no wall); near_ = next to a wall.
  if (k == 0) {
    for (int c = 0; c < N_; ++c) {  // branch-free, vectorises
      const uint8_t occd = occ_[c] == kOccupied;
      blocked_[c] = occd | (inside_[c] ^ 1u);
      seen_scratch_[c] = occd & inside_[c];
    }
  } else {
    for (int c = 0; c < N_; ++c) {
      const int x = c % W_, y = c / W_;
      uint8_t bl = 0;
      for (int yy = std::max(0, y - k); yy <= std::min(W_ - 1, y + k) && !bl; ++yy)
        for (int xx = std::max(0, x - k); xx <= std::min(W_ - 1, x + k) && !bl; ++xx)
          bl = occ_[yy * W_ + xx] == kOccupied;
      blocked_[c] = bl || !inside_[c];  // plan inside the round window only
      seen_scratch_[c] = bl && inside_[c];
    }
  }
  const uint8_t* wl = seen_scratch_;
  for (int y = 0; y < W_; ++y) {
    const int y0 = std::max(0, y - 1), y1 = std::min(W_ - 1, y + 1);
    for (int x = 0; x < W_; ++x) {
      const int x0 = std::max(0, x - 1), x1 = std::min(W_ - 1, x + 1);
      uint8_t nr = 0;
      for (int yy = y0; yy <= y1; ++yy)
        for (int xx = x0; xx <= x1; ++xx) nr |= wl[yy * W_ + xx];
      near_[y * W_ + x] = nr & (blocked_[y * W_ + x] ^ 1u);
    }
  }
  std::fill(dist_, dist_ + N_, kInf);
  std::fill(parent_, parent_ + N_, -1);
  if (start < 0) return;
  // Early exit once every goal is settled (only the goals' distances and chains are read then).
  int remaining = 0;
  if (skips_) {
    for (int i = 0; i < n_goals; ++i) {
      if (!goal_[goals[i]]) ++remaining;
      goal_[goals[i]] = 1;
    }
  }
  const bool early = remaining > 0;
  // A* towards the goals: h = distance from the cell centre to the segment through the first and
  // last goal. Every step costs at least its straight length, so h is consistent and the goals'
  // distances are exact (also for all cells if the search runs to the end).
  const bool astar = n_goals > 0;
  Vec2 ga, gb;
  if (astar) {
    ga = cell_centre(goals[0]);
    gb = cell_centre(goals[n_goals - 1]);
  }
  const float sx = gb.x - ga.x, sy = gb.y - ga.y, s2 = sx * sx + sy * sy;
  auto heur = [&](int c) {
    if (!astar) return 0.0f;
    const Vec2 q = cell_centre(c);
    float u = s2 > 0.0f ? ((q.x - ga.x) * sx + (q.y - ga.y) * sy) / s2 : 0.0f;
    u = std::min(1.0f, std::max(0.0f, u));
    return std::hypot(q.x - (ga.x + u * sx), q.y - (ga.y + u * sy));
  };
  blocked_[start] = 0;
  HeapView heap{heap_key_, heap_cell_, N_ * kHeadings + 1};
  dist_[start] = 0.0f;
  heap.push(heur(start), start);
  const float diag = r_ * std::sqrt(2.0f);
  while (heap.size > 0) {
    float key;
    int32_t c;
    heap.pop(&key, &c);
    if (key > dist_[c] + heur(c)) continue;  // stale entry
    const float d = dist_[c];
    if (d > max_dist) break;
    if (early && goal_[c] && --remaining == 0) break;
    const int x = c % W_, y = c / W_;
    for (int h = 0; h < kHeadings; ++h) {
      const int nx = x + kDX[h], ny = y + kDY[h];
      if (nx < 0 || ny < 0 || nx >= W_ || ny >= W_) continue;
      const int n = ny * W_ + nx;
      if (blocked_[n]) continue;
      if (h % 2 == 1 && (blocked_[y * W_ + nx] || blocked_[ny * W_ + x])) continue;
      const float nd = d + (h % 2 == 0 ? r_ : diag) + (near_[n] ? p_.wall_cost : 0.0f);
      if (nd < dist_[n]) {
        dist_[n] = nd;
        parent_[n] = c;
        heap.push(nd + heur(n), n);
      }
    }
  }
  for (int i = 0; i < n_goals && skips_; ++i) goal_[goals[i]] = 0;
  // Beyond max_dist: unreachable for the caller.
  if (std::isfinite(max_dist))
    for (int c = 0; c < N_; ++c)
      if (dist_[c] > max_dist) dist_[c] = kInf, parent_[c] = -1;
}

// Cells of the target's constant-velocity line p_k, k = 0..K_f, up to a wall or the window edge
// (the intercept candidates of choose_intercept).
int32_t SearchFilter::line_cells(int32_t* cells) const {
  const int NS = N_ * kHeadings;
  const bool stopped = target_state_ >= NS;
  const float dt = 1.0f / p_.rate_hz;
  const float ang = stopped ? 0.0f : (target_state_ % kHeadings) * kDTheta;
  const float ux = std::cos(ang), uy = std::sin(ang);
  int n = 0;
  for (int k = 0; k <= (stopped ? 0 : p_.K_f); ++k) {
    const int c = cell_of(Vec2{target_.x + k * s_ * dt * ux, target_.y + k * s_ * dt * uy});
    if (c < 0 || occ_[c] == kOccupied) break;
    if (n == 0 || cells[n - 1] != c) cells[n++] = c;
  }
  return n;
}

// Straight segment over cells that are not blocked; with avoid_near also not next to a wall
// (the end cells excepted).
bool SearchFilter::line_of_sight(Vec2 a, Vec2 b, bool avoid_near) const {
  bool ok = true;
  int last = -1;
  const float gx1 = b.x / r_ - ox_, gy1 = b.y / r_ - oy_;
  const int end = cell_of(b), start = cell_of(a);
  traverse(a.x / r_ - ox_, a.y / r_ - oy_, gx1, gy1, W_, [&](int c) {
    last = c;
    if (c == start || c == end) return true;
    if (blocked_[c] || (avoid_near && near_[c])) return ok = false;
    return true;
  });
  return ok && last == end;
}

// Vision-sector travel time to cell: path length d from Dijkstra, first-leg bearing beta from the
// furthest cell on the parent chain that the robot sees directly.
// d = |robot -> leg| + Dijkstra distance leg -> cell + |cell centre -> pk|.
float SearchFilter::travel_time(const Pose2& robot, int32_t cell, Vec2 pk) const {
  if (!std::isfinite(dist_[cell])) return kInf;
  const Vec2 rp{robot.x, robot.y};
  int leg_cell = cell;
  for (int c = cell; c >= 0; c = parent_[c]) {
    if (line_of_sight(rp, cell_centre(c))) {
      leg_cell = c;
      break;
    }
  }
  const Vec2 leg = leg_cell == cell ? pk : cell_centre(leg_cell);
  const Vec2 cc = cell_centre(cell);
  float d = std::hypot(leg.x - rp.x, leg.y - rp.y);
  if (leg_cell != cell)
    d += dist_[cell] - dist_[leg_cell] + std::hypot(pk.x - cc.x, pk.y - cc.y);
  if (d < 1e-3f) return 0.0f;
  const float beta = std::fabs(wrap(std::atan2(leg.y - rp.y, leg.x - rp.x) - robot.yaw));
  const float turn = (beta - p_.sector_full) / p_.omega_max;
  if (beta < p_.sector_full) return d / p_.v_full;
  if (beta < p_.sector_part) {
    const float strafe = p_.v_part > 0.0f ? d / p_.v_part : kInf;
    return std::min(strafe, turn + d / p_.v_full);
  }
  return turn + d / p_.v_full;
}

// p_k = c* + k s dt u(h*), k = 0..K_f, up to the last free cell before a wall or the window edge.
// Smallest k with travel time <= k dt + t_margin; else the last point.
int32_t SearchFilter::choose_intercept(const Pose2& robot, SearchOutput* out) {
  const int NS = N_ * kHeadings;
  const bool stopped = target_state_ >= NS;
  const float dt = 1.0f / p_.rate_hz;
  const float ang = stopped ? 0.0f : (target_state_ % kHeadings) * kDTheta;
  const float ux = std::cos(ang), uy = std::sin(ang);
  const int kmax = stopped ? 0 : p_.K_f;
  int32_t goal = -1, last = -1;
  Vec2 goal_p, last_p;
  for (int k = 0; k <= kmax; ++k) {
    const Vec2 pk{target_.x + k * s_ * dt * ux, target_.y + k * s_ * dt * uy};
    const int c = cell_of(pk);
    if (c < 0 || occ_[c] == kOccupied) break;
    if (!std::isfinite(dist_[c])) continue;
    last = c;
    last_p = pk;
    if (travel_time(robot, c, pk) <= k * dt + p_.t_margin) {
      goal = c;
      goal_p = pk;
      break;
    }
  }
  if (goal < 0 && last >= 0) goal = last, goal_p = last_p;
  if (goal < 0) {  // nothing on the line is reachable: nearest reachable cell to the target
    float best = kInf;
    for (int c = 0; c < N_; ++c) {
      if (!std::isfinite(dist_[c])) continue;
      const Vec2 q = cell_centre(c);
      const float e = std::hypot(q.x - target_.x, q.y - target_.y);
      if (e < best) best = e, goal = c;
    }
    if (goal >= 0) goal_p = cell_centre(goal);
  }
  if (goal >= 0) out->intercept = goal_p;
  return goal;
}

// Parent chain -> string pulling on blocked_ -> resample every path_spacing; yaw faces look_at.
void SearchFilter::build_path(const Pose2& robot, int32_t goal_cell, Vec2 goal, Vec2 look_at,
                              float standoff, SearchOutput* out) {
  out->path = PathView{path_, 0};
  out->path_valid = false;
  if (goal_cell < 0 || !std::isfinite(dist_[goal_cell])) return;
  int n = 0;
  for (int c = goal_cell; c >= 0 && n < N_; c = parent_[c]) chain_[n++] = c;
  std::reverse(chain_, chain_ + n);

  const Vec2 rp{robot.x, robot.y};
  int nc = 0;
  corners_[nc++] = rp;
  Vec2 anchor = rp;
  for (int j = 1; j < n; ++j) {
    if (!line_of_sight(anchor, cell_centre(chain_[j]), true)) {
      anchor = cell_centre(chain_[j - 1]);
      corners_[nc++] = anchor;
    }
  }
  corners_[nc++] = goal;
  // Stop standoff short of the goal (deviation: the robot only has to see the predicted leader;
  // driving onto it runs into the leader as an obstacle). Within standoff: turn in place.
  float total = 0.0f;
  for (int i = 1; i < nc; ++i)
    total += std::hypot(corners_[i].x - corners_[i - 1].x, corners_[i].y - corners_[i - 1].y);
  float keep = std::max(0.0f, total - standoff);
  for (int i = 1; i < nc; ++i) {
    const Vec2 a = corners_[i - 1], b = corners_[i];
    const float len = std::hypot(b.x - a.x, b.y - a.y);
    if (len >= keep) {
      const float f = len > 1e-6f ? keep / len : 0.0f;
      corners_[i] = Vec2{a.x + f * (b.x - a.x), a.y + f * (b.y - a.y)};
      nc = i + 1;
      break;
    }
    keep -= len;
  }
  goal = corners_[nc - 1];

  int np = 0;
  auto emit = [&](Vec2 q, float seg_yaw) {
    if (np >= path_cap_) return;
    const float lx = look_at.x - q.x, ly = look_at.y - q.y;
    const float yaw = std::hypot(lx, ly) > 0.2f ? std::atan2(ly, lx) : seg_yaw;
    path_[np++] = Pose2{q.x, q.y, yaw};
  };
  float seg_yaw = robot.yaw;
  if (nc == 2 && std::hypot(goal.x - rp.x, goal.y - rp.y) < 1e-3f) nc = 1;  // turn in place
  float carry = 0.0f;  // distance already travelled past the last emitted pose
  emit(rp, seg_yaw);
  for (int i = 1; i < nc; ++i) {
    const Vec2 a = corners_[i - 1], b = corners_[i];
    const float len = std::hypot(b.x - a.x, b.y - a.y);
    if (len < 1e-6f) continue;
    seg_yaw = std::atan2(b.y - a.y, b.x - a.x);
    float t = p_.path_spacing - carry;
    for (; t <= len; t += p_.path_spacing)
      emit(Vec2{a.x + (b.x - a.x) * t / len, a.y + (b.y - a.y) * t / len}, seg_yaw);
    carry = len - (t - p_.path_spacing);
  }
  if (np > 0) {
    const Pose2& e = path_[np - 1];
    if (std::hypot(e.x - goal.x, e.y - goal.y) > 1e-3f) emit(goal, seg_yaw);
  }
  out->path = PathView{path_, np};
  out->path_valid = np > 0;
}

// GIVE_UP: path to the last-seen point (kept in odom, clamped into the window), then rotate.
void SearchFilter::give_up_path(const Pose2& robot, SearchOutput* out) {
  if (std::hypot(last_seen_.x - robot.x, last_seen_.y - robot.y) <= p_.giveup_reach) {
    path_[0] = Pose2{robot.x, robot.y, robot.yaw + 1.0f};
    out->path = PathView{path_, 1};
    out->path_valid = true;
    return;
  }
  const Vec2 g = cell_of(last_seen_) < 0 ? clamp_into_window(last_seen_) : last_seen_;
  int goal = cell_of(g);
  plan_dijkstra(cell_of(Vec2{robot.x, robot.y}), &goal, goal >= 0 ? 1 : 0);
  Vec2 gp = g;
  if (goal < 0 || !std::isfinite(dist_[goal])) {
    float best = kInf;
    goal = -1;
    for (int c = 0; c < N_; ++c) {
      if (!std::isfinite(dist_[c])) continue;
      const Vec2 q = cell_centre(c);
      const float e = std::hypot(q.x - g.x, q.y - g.y);
      if (e < best) best = e, goal = c;
    }
    if (goal >= 0) gp = cell_centre(goal);
  }
  out->intercept = gp;
  build_path(robot, goal, gp, g, 0.0f, out);  // GIVE_UP: all the way to the last-seen point
}

// ------------------------------------------------------------------------------------------------
// Tick and state machine.

SearchOutput SearchFilter::step(const Pose2& robot, const ScanView* scan, const Observation* obs,
                                int32_t obs_count, float dt) {
  SearchOutput out;
  if (timing_) std::fill(stage_us_, stage_us_ + kNumStages, 0.0);
  const Observation* match = nullptr;
  const Observation* look = nullptr;
  for (int i = 0; i < obs_count; ++i) (obs[i].matched ? match : look) = &obs[i];

  // Reacquired: back to TRACKING at once; the tracker is reseeded from the detection.
  if (match) {
    if (match->has_track) {
      last_track_ = match->track;
      last_seen_ = match->track.p;
      has_track_ = true;
    }
    out.reacquired = mode_ != Mode::TRACKING;
    mode_ = Mode::TRACKING;
    time_since_seen_ = 0.0f;
    has_target_ = false;
    out.mode = mode_;
    return out;
  }

  float dt_eff = dt;
  if (mode_ == Mode::TRACKING) {
    time_since_seen_ += dt;
    if (!has_track_ || time_since_seen_ < p_.t_grace) {
      out.mode = mode_;
      return out;
    }
    const float since_seen = time_since_seen_;
    on_loss(last_track_, robot, scan);
    out.clamped = clamped_;
    dt_eff = since_seen;  // predict from the last sighting to now
    scan = nullptr;             // already rasterized
  }

  time_since_loss_ += dt_eff;
  if (mode_ == Mode::SEARCH) {
    filter_tick(robot, scan, look, dt_eff, &out);
    const float mcm = max_cell_mass();
    if (time_since_loss_ > p_.t_max || mcm < p_.p_min || oow_ > p_.p_out_max) {
      mode_ = Mode::GIVE_UP;
    } else {
      int64_t t0 = 0;
      stage_begin(&t0);
      const bool has_target = select_target(&out);
      stage_end(kStTarget, &t0);
      if (has_target) {
        const int rc = cell_of(Vec2{robot.x, robot.y});
        plan_dijkstra(rc, goals_, line_cells(goals_));
        stage_end(kStDijkstra, &t0);
        const int goal = choose_intercept(robot, &out);
        stage_end(kStIntercept, &t0);
        build_path(robot, goal, out.intercept, target_, p_.standoff, &out);
        stage_end(kStPath, &t0);
      }
    }
    out.max_cell_mass = mcm;
  } else {  // GIVE_UP: keep the window and occupancy current, drive to the last-seen point
    shift_window(robot);
    rasterize(robot, scan);
  }
  if (mode_ == Mode::GIVE_UP) give_up_path(robot, &out);
  out.mode = mode_;
  out.out_of_window = oow_;
  out.dropped_mass = dropped_;
  out.time_since_loss = time_since_loss_;
  return out;
}

}  // namespace leader_search
