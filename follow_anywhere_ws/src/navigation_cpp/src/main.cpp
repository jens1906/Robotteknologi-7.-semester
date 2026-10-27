#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <osqp.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "perception_msgs/msg/detection_array.hpp"
#include "rclcpp/rclcpp.hpp"
#include "unitree_api/msg/request.hpp"

using perception_msgs::msg::Detection;
using perception_msgs::msg::DetectionArray;
using std::chrono::duration;
using std::chrono::steady_clock;
using unitree_api::msg::Request;

using SpMat = Eigen::SparseMatrix<OSQPFloat, Eigen::ColMajor, OSQPInt>;
using Triplet = Eigen::Triplet<OSQPFloat, OSQPInt>;

OSQPCscMatrix to_csc(SpMat &M) {
  M.makeCompressed();
  OSQPCscMatrix csc{};
  csc.m = M.rows();
  csc.n = M.cols();
  csc.p = M.outerIndexPtr();
  csc.i = M.innerIndexPtr();
  csc.x = M.valuePtr();
  csc.nzmax = M.nonZeros();
  csc.nz = -1;
  return csc;
}

constexpr int API_STOP_MOVE = 1003;
constexpr int API_MOVE = 1008;

constexpr int CLASS_LEADER = Detection::CLASS_LEADER;

static double time_now() {
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

constexpr int MAX_OBSTACLES = 10;
constexpr int PREDICTION_HORIZON = 20;

struct Command {
  double vx = 0.0;
  double vy = 0.0;
  double wz = 0.0;

  bool is_zero(double eps = 1e-6) const {
    return std::abs(vx) < eps && std::abs(vy) < eps && std::abs(wz) < eps;
  }

  bool is_valid(double eps = 1e-6) const {
    return !is_zero(eps) && std::isfinite(vx) && std::isfinite(vy) &&
           std::isfinite(wz);
  }
};

struct Obstacle {
  Eigen::Vector2d pol =
      Eigen::Vector2d::Zero(); // position in polar coordinates (r, theta)
  Eigen::Vector2d pos =
      Eigen::Vector2d::Zero(); // position in Cartesian coordinates (x, y)
  Eigen::Vector2d vel =
      Eigen::Vector2d::Zero(); // velocity in Cartesian coordinates (vx, vy)
  double std = 0.0;            // standard deviation of the obstacle position
  double spd = 0.0;            // speed of the obstacle
};

Obstacle from_detection(const Detection &detection) {
  Obstacle obs;
  obs.pol = Eigen::Vector2d(detection.range, detection.bearing);
  obs.pos = Eigen::Vector2d(detection.range * std::cos(detection.bearing),
                            detection.range * std::sin(detection.bearing));
  obs.vel = Eigen::Vector2d(detection.velocity.x, detection.velocity.y);
  obs.std = detection.velocity_std;
  obs.spd = obs.vel.norm();
  if (obs.vel.norm() < detection.velocity_std * 2.0) {
    obs.vel = Eigen::Vector2d::Zero();
  }
  return obs;
}

struct ObstacleList {
  std::array<Obstacle, MAX_OBSTACLES> obstacles;
  int size = 0;
  void clear() { size = 0; }
  void fill_from_detections(const DetectionArray &detections,
                            std::array<uint8_t, MAX_OBSTACLES> order) {
    clear();
    for (int i = 0; i < detections.detections.size() && i < MAX_OBSTACLES;
         ++i) {
      if (order[i] == 255)
        break;
      obstacles[i] = from_detection(detections.detections[order[i]]);
      size++;
    }
  }
};

class LeaderFollower : public rclcpp::Node {
private:
  bool has_leader = false;
  uint32_t leader_id = 0;
  double last_seen = 0.0;
  Detection leader;

  bool last_path_valid = false;
  Eigen::Matrix<double, 3, PREDICTION_HORIZON> last_path;

  ObstacleList obstacles;

  Command cmd;

  bool moving = false;

  double dt;
  rclcpp::TimerBase::SharedPtr timer;
  rclcpp::Publisher<Request>::SharedPtr pub;
  rclcpp::Subscription<DetectionArray>::SharedPtr sub;

  std::vector<Triplet> A_c, P_c;
  std::vector<OSQPFloat> l, u, q;

  OSQPSolver *solver = nullptr;

public:
  LeaderFollower() : Node("leader_follower") {
    // Initialize parameters
    pub = create_publisher<Request>("/api/sport/request", 10);
    sub = create_subscription<DetectionArray>(
        "/perception/detections", 10,
        [this](const DetectionArray::SharedPtr msg) {
          this->detections_callback(msg);
        });
    const double rate = declare_parameter("rate", 10.0);
    dt = 1.0 / rate;
    timer = create_wall_timer(duration<double>(dt),
                              [this]() { this->timer_callback(); });

    constexpr int N = PREDICTION_HORIZON, M = MAX_OBSTACLES;
    A_c.reserve(3 + N * (18 + 4 * M));
    P_c.reserve(N * (16 + M)); // no delta = N * (3 + 3 + 1 + M)
    l.reserve(3 + N * (6 + 2 * M));
    u.reserve(3 + N * (6 + 2 * M));
    q.reserve(3 * (N + 1) + 3 * N + M * N);
  }

  ~LeaderFollower() {
    if (solver)
      osqp_cleanup(solver);
  }

  void detections_callback(const DetectionArray::SharedPtr msg) {
    const double now = time_now();

    std::array<uint8_t, MAX_OBSTACLES> d_order;
    std::array<double, MAX_OBSTACLES> d_distances;

    int count = 0;
    for (size_t i = 0; i < msg->detections.size(); ++i) {
      const Detection &det = msg->detections[i];

      if (det.class_id == CLASS_LEADER) {
        has_leader = true;
        leader_id = det.id;
        leader = det;
        last_seen = now;
      } else if (has_leader && det.id == leader_id) {
        leader = det;
        last_seen = now;
      } else {
        Obstacle o = from_detection(det);
        double dist = std::hypot(o.pos.x(), o.pos.y());

        // find sorted insertion point
        int k = count;
        while (k > 0 && d_distances[k - 1] > dist)
          --k;
        if (k >= MAX_OBSTACLES)
          continue; // farther than everything kept

        // shift [k, last) right by one; drops the farthest when full
        int last = std::min(count, MAX_OBSTACLES - 1);
        std::copy_backward(d_distances.begin() + k, d_distances.begin() + last,
                           d_distances.begin() + last + 1);
        std::copy_backward(d_order.begin() + k, d_order.begin() + last,
                           d_order.begin() + last + 1);
        d_distances[k] = dist;
        d_order[k] = i;
        if (count < MAX_OBSTACLES)
          ++count;
      }
    }

    for (int j = count; j < MAX_OBSTACLES; ++j) {
      d_order[j] = 255;
      d_distances[j] = std::numeric_limits<double>::infinity();
    }

    obstacles.fill_from_detections(*msg, d_order);
  }

  void timer_callback() {
    compute_command();

    if (cmd.is_valid()) {
      char buf[128];
      std::snprintf(buf, sizeof(buf), "{\"x\": %.4f, \"y\": %.4f, \"z\": %.4f}",
                    cmd.vx, cmd.vy, cmd.wz);
      send(API_MOVE, buf);
      moving = true;
    } else if (moving) {
      stop();
    }
  }

  void send(int api_id, const std::string &parameter = "") {
    Request msg;
    msg.header.identity.id =
        std::chrono::duration_cast<std::chrono::microseconds>(
            steady_clock::now().time_since_epoch())
            .count();
    msg.header.identity.api_id = api_id;
    msg.parameter = parameter;
    pub->publish(msg);
  }

  void stop() {
    send(API_STOP_MOVE);
    moving = false;
  }

  Eigen::Matrix<double, 3, 3> B(double theta) {
    Eigen::Matrix<double, 3, 3> B;
    const double cos_t = std::cos(theta);
    const double sin_t = std::sin(theta);
    B << cos_t, -sin_t, 0.0, sin_t, cos_t, 0.0, 0.0, 0.0, 1.0;
    return B;
  }

  void compute_command() {
    const double now = time_now();
    cmd = Command();

    if (!has_leader)
      return;
    if ((now - last_seen) > 1.0) {
      has_leader = false;
      last_path_valid = false;
      return;
    }

    const double dt = this->dt;

    const int NX = 3 * (PREDICTION_HORIZON + 1), NU = 3 * PREDICTION_HORIZON,
              M = MAX_OBSTACLES;
    const int n_vars = NX + NU + M * PREDICTION_HORIZON, N = PREDICTION_HORIZON;
    auto xi = [&](int r, int k) { return 3 * k + r; }; // x section starts at 0
    auto ui = [&](int r, int k) {
      return NX + 3 * k + r;
    }; // NX = 3*(N+1), where u starts
    auto si = [&](int i, int k) {
      return NX + NU + M * k + i;
    }; // NU = 3*N, where s starts

    A_c.clear();
    l.clear();
    u.clear();
    P_c.clear();
    q.assign(n_vars, 0.0);

    auto add_row = [&](OSQPFloat lo, OSQPFloat hi) {
      l.push_back(lo);
      u.push_back(hi);
      return static_cast<OSQPInt>(l.size() - 1);
    };

    const Eigen::Vector2d Q = Eigen::Vector2d(8.0, 0.5);
    const Eigen::Vector3d R = Eigen::Vector3d(7.5, 7.5, 0.5);
    const Eigen::Vector3d S =
        Eigen::Vector3d(3.0, 3.0, 1.0); // Δu weights: vx, vy, wz
    Eigen::Vector3d u_prev(last_path(0, 0), last_path(1, 0), last_path(2, 0));
    if (!last_path_valid)
      u_prev.setZero();

    const Eigen::Vector3d x0 = Eigen::Vector3d::Zero();
    Eigen::Vector2d p = Eigen::Vector2d::Zero();
    double t = 0.0;

    double hard_margin = 1.0;
    double soft_margin = 2.0;
    double leader_margin = 2.0;
    double margin_cost = 50.0;

    const int num_obstacles = obstacles.size;

    std::array<double, 4> leader_arr = {leader.range * std::cos(leader.bearing),
                                        leader.range * std::sin(leader.bearing),
                                        leader.velocity.x, leader.velocity.y};
    double leader_speed = std::hypot(leader_arr[2], leader_arr[3]);

    if (leader_speed < leader.velocity_std * 2.0) {
      leader_arr[2] = 0.0;
      leader_arr[3] = 0.0;
    }

    std::array<double, 6> speed_limit = {
        -1.0, 1.0,  -1.0,
        1.0,  -1.0, 1.0}; // vx_min, vx_max, vy_min, vy_max, wz_min, wz_max

    for (int r = 0; r < 3; ++r) {
      A_c.emplace_back(add_row(x0[r], x0[r]), xi(r, 0), 1.0);
    }

    for (int k = 0; k < PREDICTION_HORIZON; ++k) {
      Eigen::Matrix<double, 3, 3> Bm = B(t);
      for (int r = 0; r < 3; ++r) {
        int row_idx = add_row(0.0, 0.0);
        A_c.emplace_back(row_idx, xi(r, k + 1), 1.0); // 0.0 <= x_{k+1} <= 0.0
        A_c.emplace_back(row_idx, xi(r, k),
                         -1.0); // 0.0 <= x_{k+1} - x_k <= 0.0
        for (int c = 0; c < 3; ++c) {
          A_c.emplace_back(
              row_idx, ui(c, k),
              -dt * Bm(r, c)); // 0.0 <= x_{k+1} - x_k - dt*B*u_k <= 0.0
        }
      }
      for (int r = 0; r < 3; ++r) {
        A_c.emplace_back(add_row(speed_limit[2 * r], speed_limit[2 * r + 1]),
                         ui(r, k), 1.0); // lower_spd_lim <= u_k <= top_spd_lim
      }
      if (last_path_valid) {
        int idx = std::min(k + 1, PREDICTION_HORIZON - 1);
        double cos_t = std::cos(t);
        double sin_t = std::sin(t);
        p(0) += last_path(0, idx) * dt * cos_t - last_path(1, idx) * dt * sin_t;
        p(1) += last_path(0, idx) * dt * sin_t + last_path(1, idx) * dt * cos_t;
        t += last_path(2, idx) * dt;
      }
      leader_arr[0] += leader_arr[2] * dt;
      leader_arr[1] += leader_arr[3] * dt;
      Eigen::Vector2d leader_pos(leader_arr[0], leader_arr[1]);
      Eigen::Vector2d leader_rel_pos = leader_pos - p;
      double leader_dist = std::hypot(leader_rel_pos(0), leader_rel_pos(1));
      Eigen::Vector2d leader_dir = leader_rel_pos / leader_dist;
      double leader_cost = leader_dir.dot(leader_pos) - leader_margin;
      double leader_heading = std::atan2(leader_dir(1), leader_dir(0));
      double leader_heading_err =
          std::remainder(leader_heading - t, 2.0 * M_PI);
      leader_heading = t + leader_heading_err;
      int px = xi(0, k + 1), py = xi(1, k + 1);
      q[px] += -Q(0) * leader_dir(0) * leader_cost;
      q[py] += -Q(0) * leader_dir(1) * leader_cost;
      q[xi(2, k + 1)] += -Q(1) * leader_heading;

      P_c.emplace_back(px, px, Q(0) * leader_dir(0) * leader_dir(0));
      P_c.emplace_back(px, py, Q(0) * leader_dir(0) * leader_dir(1));
      P_c.emplace_back(py, py, Q(0) * leader_dir(1) * leader_dir(1));

      for (int r = 0; r < 3; ++r) {
        P_c.emplace_back(ui(r, k), ui(r, k), R(r)); // cost for control input
      }

      P_c.emplace_back(xi(2, k + 1), xi(2, k + 1), Q(1)); // cost for heading
                                                          //
      for (int r = 0; r < 3; ++r) {
        const int a = ui(r, k);
        P_c.emplace_back(a, a, S(r));
        if (k == 0) {
          q[a] += -S(r) * u_prev(r);
        } else {
          const int b = ui(r, k - 1);
          P_c.emplace_back(b, b, S(r));
          P_c.emplace_back(b, a, -S(r)); // upper triangle: b < a
        }
      }

      for (int i = 0; i < MAX_OBSTACLES; ++i) {
        P_c.emplace_back(si(i, k), si(i, k),
                         margin_cost); // cost for soft margin
        if (i < num_obstacles) {
          Obstacle &obs = obstacles.obstacles[i];
          Eigen::Vector2d obs_pos = obs.pos;
          Eigen::Vector2d obs_vel = obs.vel;
          obs_pos(0) += obs_vel(0) * dt * (k + 1);
          obs_pos(1) += obs_vel(1) * dt * (k + 1);
          Eigen::Vector2d obs_mov = obs_pos;
          obs_pos(0) -= p(0);
          obs_pos(1) -= p(1);
          double obs_dist = std::hypot(obs_pos(0), obs_pos(1));
          Eigen::Vector2d obs_dir = obs_pos / obs_dist;
          int row_idx = add_row(soft_margin - obs_dir.dot(obs_mov), OSQP_INFTY);
          // soft_margin - obs_dir.dot(obs_mov) <= -obs_dir.dot(x_{k+1}) +
          // s_{i,k} <= inf
          A_c.emplace_back(row_idx, xi(0, k + 1), -obs_dir(0));
          A_c.emplace_back(row_idx, xi(1, k + 1), -obs_dir(1));
          A_c.emplace_back(row_idx, si(i, k), 1.0);
        } else {
          int row_idx = add_row(-OSQP_INFTY, OSQP_INFTY);
          A_c.emplace_back(row_idx, xi(0, k + 1),
                           0.0); // dummy constraint for unused obstacles
          A_c.emplace_back(row_idx, xi(1, k + 1), 0.0);
          A_c.emplace_back(row_idx, si(i, k), 0.0);
        }
        int s_row_idx = add_row(0.0, soft_margin - hard_margin);
        // 0.0 <= s_{i,k} <= soft_margin - hard_margin
        A_c.emplace_back(s_row_idx, si(i, k),
                         1.0); // s_{i,k} = 0 for unused obstacles
      }
    }

    const int n_rows = static_cast<int>(l.size());
    SpMat A(n_rows, n_vars);
    A.setFromTriplets(A_c.begin(), A_c.end());
    SpMat P(n_vars, n_vars);
    P.setFromTriplets(P_c.begin(),
                      P_c.end()); // only upper-triangle entries were added

    OSQPCscMatrix A_csc = to_csc(A);
    OSQPCscMatrix P_csc = to_csc(P);

    if (solver) {
      osqp_update_data_vec(solver, q.data(), l.data(), u.data());
      osqp_update_data_mat(solver, P.valuePtr(), nullptr, P.nonZeros(),
                           A.valuePtr(), nullptr, A.nonZeros());
    } else {
      OSQPSettings settings;
      osqp_set_default_settings(&settings);
      settings.time_limit = 0.5 * dt;
      settings.verbose = 0;
      if (osqp_setup(&solver, &P_csc, q.data(), &A_csc, l.data(), u.data(),
                     n_rows, n_vars, &settings) != 0) {
        RCLCPP_WARN(get_logger(), "OSQP setup failed.");
        if (solver) {
          osqp_cleanup(solver);
          solver = nullptr;
        }
        last_path_valid = false;
        return;
      }
    }

    osqp_solve(solver);

    const OSQPInt status = solver->info->status_val;
    if (status == OSQP_SOLVED || status == OSQP_SOLVED_INACCURATE) {
      const OSQPFloat *z = solver->solution->x;
      for (int k = 0; k < N; ++k)
        for (int r = 0; r < 3; ++r)
          last_path(r, k) = z[ui(r, k)];
      last_path_valid = true;
      cmd.vx = z[ui(0, 0)];
      cmd.vy = z[ui(1, 0)];
      cmd.wz = z[ui(2, 0)];
    } else {
      RCLCPP_WARN(get_logger(), "OSQP: %s", solver->info->status);
      last_path_valid = false;
    }
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LeaderFollower>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
