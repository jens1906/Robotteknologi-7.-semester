// Lost-leader search: a fixed-size, double-buffered Bayes filter on a directed grid graph.
// Spec: docs/leader_search_spec.md. Symbols in comments follow the spec (q, p_m, p_d, s, r, ...).
//
// Pure C++17, no ROS. All memory is allocated in init(); step() never allocates or throws.
// Grid: W x W cells of size r, axis-aligned with odom, centred on the robot (whole-cell shifts).
// With window_radius > 0 only the cells within that radius of the robot's cell are in the window.
// State index: idx = (y * W + x) * 8 + h. Heading 0 = +x, counter-clockwise in 45 degree steps.
#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <string>

namespace leader_search {

constexpr int kHeadings = 8;
constexpr int kPdBins = 10;  // p_d range bins of 1 m: [0,1), [1,2), ... [9,10)

enum class BlockedMode : uint8_t { DROP = 0, REDIRECT = 1 };
enum class Mode : uint8_t { TRACKING = 0, SEARCH = 1, GIVE_UP = 2 };

// Values of the occupancy window.
enum : uint8_t { kFree = 0, kOccupied = 1, kUnknown = 2 };
// Values of the edge mask.
enum : uint8_t { kEdgeBlocked = 0, kEdgeOpen = 1, kEdgeLeaves = 2 };

struct Params {
  float rate_hz = 10.0f;       // search tick rate, dt = 1 / rate_hz
  float window_radius = 15.0f; // [m] round window around the robot; 0 = square of window_cells
  int32_t window_cells = 20;   // W of the square window (window_radius = 0)
  float resolution = 0.5f;     // r [m]
  float t_grace = 0.3f;        // [s] without a re-ID match before SEARCH (perception's lost_duration)
  int32_t n_grace = 2;         // and at least this many unmatched frames in a row (slow cameras)
  float t_silent = 1.0f;       // [s] without any camera frame before SEARCH (perception stalled)
  float s_min = 0.3f;          // [m/s] leader speed clamp
  float s_max = 2.0f;
  float v_stop = 0.15f;        // [m/s] below: leader treated as stopped at loss
  float sigma_theta0 = 0.3f;   // [rad] initial heading spread
  float sigma_h = 0.6f;        // [rad/sqrt(s)] heading diffusion rate
  bool stopped_enabled = true;
  float p_stop0 = 0.1f;        // initial stopped mass
  float lambda_stop = 0.2f;    // [1/s] moving -> stopped
  float lambda_go = 0.5f;      // [1/s] stopped -> moving
  BlockedMode blocked_mode = BlockedMode::DROP;
  bool occ_memory = true;      // occupancy keeps the last observed state of unscanned cells
  float cam_fov = 1.52f;       // [rad] D455 horizontal FOV
  float cam_range = 6.0f;      // [m] re-ID range
  float cam_yaw = 0.0f;        // [rad] camera mount yaw in base_link
  float p_d[kPdBins] = {0.6f, 0.6f, 0.6f, 0.6f, 0.6f, 0.6f, 0.6f, 0.6f, 0.6f, 0.6f};
  float rho = 1.0f;            // [m] target hysteresis radius
  float delta_log = 0.69f;     // target hysteresis margin (log)
  int32_t K_f = 30;            // intercept horizon [ticks]
  float t_margin = 0.5f;       // [s] intercept slack
  float v_full = 0.5f;         // [m/s] full-vision sector speed (MPC speed_limit[0])
  float v_part = 0.2f;         // [m/s] partial-vision sector speed (MPC speed_limit[1])
  float omega_max = 0.5f;      // [rad/s] turn rate (MPC speed_limit[3])
  float sector_full = 1.4f;    // [rad] MPC full-vision sector edge
  float sector_part = 2.3f;    // [rad] MPC partial-vision sector edge
  int32_t inflate_cells = 0;   // planner obstacle inflation
  float wall_cost = 1.0f;      // [m] planner: extra cost of entering a cell next to a wall
  float path_spacing = 0.1f;   // [m] output path resampling
  float tail_cut = 1e-6f;      // belief below this is set to 0 (skips empty cells; 0 = exact)
  float standoff = 1.0f;       // [m] SEARCH path ends this far before the intercept (look, not reach)
  float t_max = 20.0f;         // [s] give up after this long in SEARCH
  float p_min = 0.01f;         // give up when the largest cell mass is below this
  float p_out_max = 0.5f;      // give up when out_of_window exceeds this
  float giveup_reach = 0.3f;   // [m] at the last-seen point: rotate in place
};

// Checks ranges and consistency. Writes a reason to *err when false.
bool validate(const Params& p, std::string* err);
// Reads a YAML file: either flat keys or nested `<node>: ros__parameters:`. Missing keys keep
// their defaults. Init-time only (allocates).
bool load_params(const std::string& yaml_path, Params* out, std::string* err);

struct Vec2 {
  float x = 0.0f, y = 0.0f;
};
struct Pose2 {
  float x = 0.0f, y = 0.0f, yaw = 0.0f;
};

// Last tracker estimate, odom frame.
struct TrackInit {
  Vec2 p;                                // position
  Vec2 v;                                // velocity over ground
  float cov[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // position covariance, row-major 2x2
};

// One LaserScan, non-owning. Beams are in the sensor frame; sensor_in_base places it on the robot.
struct ScanView {
  const float* ranges = nullptr;
  int32_t count = 0;
  float angle_min = 0.0f, angle_increment = 0.0f, range_min = 0.0f, range_max = 0.0f;
  Pose2 sensor_in_base;
};

// One camera frame's re-ID result. matched = leader identified (reacquired).
struct Observation {
  bool matched = false;
  bool has_track = false;  // matched with a usable position/velocity
  TrackInit track;         // valid when has_track
  Pose2 robot;             // robot pose in odom at the frame
  // Unmatched frames: perception's time since its last match (PerceptionLeader.lost_duration).
  // NaN = unknown; then the filter counts its own ticks instead.
  float lost_duration = std::numeric_limits<float>::quiet_NaN();
};

struct PathView {
  const Pose2* poses = nullptr;  // filter-owned, valid until the next step()
  int32_t size = 0;
};

enum class LossCause : uint8_t {
  kNone = 0,
  kNotMatched,  // camera frames kept coming without a match for t_grace
  kSilent,      // no camera frame at all for t_silent
};

struct SearchOutput {
  Mode mode = Mode::TRACKING;
  bool reacquired = false;       // SEARCH/GIVE_UP -> TRACKING this tick
  LossCause lost = LossCause::kNone;  // TRACKING -> SEARCH this tick, and why
  float lost_after = 0.0f;       // [s] since the last match when it happened
  bool target_switched = false;  // hysteresis accepted a new candidate this tick
  bool clamped = false;          // loss point was outside the window and clamped (warn)
  bool path_valid = false;
  PathView path;                 // odom frame, robot -> intercept, yaw faces the target
  Vec2 target;                   // target state's cell centre, odom
  int8_t target_heading = -1;    // -1 = stopped state
  Vec2 intercept;                // odom
  bool looked = false;           // an unmatched camera frame was applied this tick
  float max_cell_mass = 0.0f;
  float out_of_window = 0.0f;
  float dropped_mass = 0.0f;     // DROP (or REDIRECT without an open h+-1) this tick
  float time_since_loss = 0.0f;
};

// Inclusive rectangle of window cells; x0 > x1 = empty. The filter keeps one per buffer: outside it
// the buffer is exactly 0 (belief) and -inf (max-product), so passes can skip it.
struct CellBox {
  int32_t x0 = 0, y0 = 0, x1 = -1, y1 = -1;
  bool empty() const { return x0 > x1 || y0 > y1; }
};

// Per-stage time of the last step() in microseconds (profiling, off unless enabled).
enum Stage : int {
  kStShift, kStRaster, kStEdges, kStDiffuse, kStMove, kStObserve, kStNormalise, kStTarget,
  kStDijkstra, kStIntercept, kStPath, kNumStages
};

class SearchFilter {
 public:
  SearchFilter();
  ~SearchFilter();
  SearchFilter(const SearchFilter&) = delete;
  SearchFilter& operator=(const SearchFilter&) = delete;

  // Allocates every buffer from p (the only allocating call). False if p is invalid.
  bool init(const Params& p, std::string* err = nullptr);

  // TRACKING -> SEARCH (spec "On loss"): centres the window on robot, rasterizes scan (null:
  // keep the last occupancy if the window did not move, else unknown), seeds b0, stopped, m0.
  // Also called by step() after t_grace.
  void on_loss(const TrackInit& t, const Pose2& robot, const ScanView* scan);

  // One tick (spec passes 1-9 plus the state machine). obs: camera frames since the last tick.
  SearchOutput step(const Pose2& robot, const ScanView* scan, const Observation* obs,
                    int32_t obs_count, float dt);

  void reset();  // -> TRACKING, buffers zeroed, no reallocation

  // Read-only access for debug output and tests. Buffers are the current read buffers.
  const Params& params() const { return p_; }
  int32_t window_cells() const { return W_; }
  int32_t num_cells() const { return N_; }
  float resolution() const { return r_; }
  Vec2 window_origin() const;  // odom coords of the outer corner of cell (0, 0)
  int32_t cell_of(Vec2 odom) const;  // -1 outside (also outside the round window)
  Vec2 clamp_into_window(Vec2 odom) const;
  Vec2 cell_centre(int32_t c) const;
  void belief_marginal(float* out_n) const;  // sum over h plus stopped, size N
  const float* belief() const { return b_; }
  const float* stopped() const { return st_; }
  const float* maxprod() const { return m_; }
  const float* maxprod_stopped() const { return ms_; }
  const uint8_t* edge_open() const;  // N*8 kEdge* values, expanded from the bit masks
  const uint8_t* occ() const { return occ_; }
  const uint8_t* seen() const { return seen_; }
  float out_of_window() const { return oow_; }
  float dropped_mass() const { return dropped_; }
  float speed() const { return s_; }
  Mode mode() const { return mode_; }
  size_t arena_bytes() const { return arena_size_; }

  // Tests: iterate cells in this order inside every pass (nullptr = natural order). The array
  // must hold a permutation of [0, N) and outlive the calls. Reductions stay in natural order.
  void debug_set_cell_order(const int32_t* order) { order_ = order; }
  // Tests: replace the occupancy window (N values) and rebuild the edge mask.
  void debug_set_occ(const uint8_t* occ);
  // Tests: run only the belief part of a tick (shift, predict, max-product, observe, normalise).
  void debug_filter_tick(const Pose2& robot, const ScanView* scan, const Observation* looked,
                         float dt);
  // Profiling: time each stage of step() (steady_clock; no allocation).
  void enable_stage_timing(bool on) { timing_ = on; }
  const double* stage_us() const { return stage_us_; }
  static const char* stage_name(int s);
  // Tests: run every pass over the whole window and Dijkstra to the end (results must not change).
  void debug_disable_skips(bool off) { skips_ = !off; }
  // Tests: support boxes of the belief (b, m) and of the stopped state (st, ms).
  CellBox debug_box_belief() const { return box_b_; }
  CellBox debug_box_stopped() const { return box_s_; }
  // Tests: only the observation update from this robot pose (no normalisation).
  void debug_observe(const Pose2& robot) { observe(robot); }
  // Total of b + stopped + out_of_window just before the last normalisation (should be ~1).
  float last_total() const { return last_total_; }

 private:
  struct TickConst;

  int32_t ord(int32_t i) const { return order_ ? order_[i] : i; }
  void swap_all();
  void centre_window(const Pose2& robot);
  void shift_window(const Pose2& robot);
  void rasterize(const Pose2& robot, const ScanView* scan);
  void build_edge_mask();
  void build_edge_mask(const CellBox& region);
  float cut_b_(float v) const { return v < p_.tail_cut ? 0.0f : v; }
  CellBox full_box() const { return CellBox{0, 0, W_ - 1, W_ - 1}; }
  CellBox expand(const CellBox& b, int k) const;
  CellBox region(const CellBox& b) const { return skips_ ? b : full_box(); }
  template <class F>
  void for_cells(const CellBox& r, F&& f) const;  // f(c, x, y) over window cells in r
  void boxes_from_values();
  void predict(float dt);
  void heading_diffusion(const TickConst& k);
  void move(const TickConst& k);
  void observe(const Pose2& cam_robot);
  void observe_rays(const Pose2& robot);
  void observation_factors(float* f, float* lf) const;
  void observe_normalise(const Pose2* cam);
  bool select_target(SearchOutput* out);
  // Stops once all goal cells are settled (n_goals > 0) or beyond max_dist.
  void plan_dijkstra(int32_t start, const int32_t* goals = nullptr, int32_t n_goals = 0,
                     float max_dist = std::numeric_limits<float>::infinity());
  int32_t line_cells(int32_t* cells) const;  // the target's constant-velocity line (intercept)
  int32_t choose_intercept(const Pose2& robot, SearchOutput* out);
  void build_path(const Pose2& robot, int32_t goal_cell, Vec2 goal, Vec2 look_at, float standoff,
                  SearchOutput* out);
  bool line_of_sight(Vec2 a, Vec2 b, bool avoid_near = false) const;
  float travel_time(const Pose2& robot, int32_t cell, Vec2 pk) const;
  void filter_tick(const Pose2& robot, const ScanView* scan, const Observation* looked,
                   float dt, SearchOutput* out);
  void give_up_path(const Pose2& robot, SearchOutput* out);
  float max_cell_mass() const;

  Params p_;
  int32_t W_ = 0, N_ = 0;
  float r_ = 0.0f;
  int32_t ox_ = 0, oy_ = 0;  // window origin in whole cells of the odom grid
  const int32_t* order_ = nullptr;

  // Arena and the buffers carved from it.
  std::unique_ptr<unsigned char[]> arena_;
  size_t arena_size_ = 0;
  float *b_ = nullptr, *b2_ = nullptr;    // belief, N*8
  float *st_ = nullptr, *st2_ = nullptr;  // stopped, N
  float *m_ = nullptr, *m2_ = nullptr;    // max-product (log), N*8
  float *ms_ = nullptr, *ms2_ = nullptr;  // max-product of stopped (log), N
  uint8_t* edge_ = nullptr;               // N*8, debug expansion of emask_ (edge_open())
  uint8_t* emask_ = nullptr;              // N, bit h = edge (c, h) open
  uint8_t* seen_scratch_ = nullptr;       // N, planner scratch: wall cells inside the window
  uint8_t* leave_ = nullptr;              // N, bit h = c + d(h) is off the window (fixed)
  uint8_t* occ_ = nullptr;                // N
  uint8_t* seen_ = nullptr;               // N
  uint8_t* blocked_ = nullptr;            // N, planner obstacles (inflated)
  uint8_t* near_ = nullptr;               // N, planner: free cell next to a blocked one
  uint8_t* inside_ = nullptr;             // N, 1 = cell inside the round window
  uint8_t* goal_ = nullptr;               // N, Dijkstra goal flags (scratch)
  int32_t* goals_ = nullptr;              // N, Dijkstra goal list (scratch)
  CellBox box_b_, box_b2_, box_s_, box_s2_;  // support of b/m, b2/m2, st/ms, st2/ms2
  bool skips_ = true;
  bool timing_ = false;
  double stage_us_[kNumStages] = {};
  void stage_begin(int64_t* t0) const;
  void stage_end(int stage, int64_t* t0);
  float* dist_ = nullptr;                 // N
  int32_t* parent_ = nullptr;             // N
  float* heap_key_ = nullptr;             // N*8
  int32_t* heap_cell_ = nullptr;          // N*8
  int32_t* chain_ = nullptr;              // N, cell path robot -> goal
  Vec2* corners_ = nullptr;               // N+2, string-pulled polyline
  Pose2* path_ = nullptr;                 // path_cap_
  int32_t path_cap_ = 0;

  // Filter state.
  Mode mode_ = Mode::TRACKING;
  float s_ = 0.0f;  // leader speed s
  float phase_o_ = 0.0f, phase_d_ = 0.0f;  // max-product move phases (orthogonal, diagonal)
  float oow_ = 0.0f, dropped_ = 0.0f, last_total_ = 1.0f;
  float time_since_seen_ = 0.0f, time_since_loss_ = 0.0f;
  float time_since_frame_ = 0.0f;
  int32_t misses_ = 0;             // unmatched camera frames since the last match  // since the last camera frame of any kind (TRACKING)
  bool has_track_ = false;
  TrackInit last_track_;
  Vec2 last_seen_;
  bool has_target_ = false;
  Vec2 target_;
  int32_t target_state_ = -1;
  int32_t best_state_ = -1;  // argmax of m from the last observe_normalise()  // index into m (N*8) or N*8 + c for stopped
  bool clamped_ = false;
};

}  // namespace leader_search
