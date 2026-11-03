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

#include "follow_anywhere_interfaces/msg/perception_leader.hpp"
#include "follow_anywhere_interfaces/msg/perception_obstacle_array.hpp"
#include "rclcpp/rclcpp.hpp"
#include "unitree_api/msg/request.hpp"

using follow_anywhere_interfaces::msg::PerceptionLeader;
using follow_anywhere_interfaces::msg::PerceptionObstacle;
using follow_anywhere_interfaces::msg::PerceptionObstacleArray;
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

static double time_now() {
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

constexpr int MAX_OBSTACLES = 10;
constexpr int PREDICTION_HORIZON = 15;

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
  Eigen::Vector2d near_pol =
      Eigen::Vector2d::Zero(); // position in polar coordinates (r, theta)
  Eigen::Vector2d near_pos =
      Eigen::Vector2d::Zero(); // position in Cartesian coordinates (x, y)
  Eigen::Vector2d los_pol =
      Eigen::Vector2d::Zero(); // position in polar coordinates (r, theta)
  Eigen::Vector2d los_pos =
      Eigen::Vector2d::Zero(); // position in Cartesian coordinates (x, y)
  Eigen::Vector2d vel =
      Eigen::Vector2d::Zero(); // velocity in Cartesian coordinates (vx, vy)
  double pos_std = 0.0;        // standard deviation of the obstacle position
  double vel_std = 0.0;
  double spd = 0.0; // speed of the obstacle
};

struct Leader {
  bool has_leader = false;
  Eigen::Vector2d pol =
      Eigen::Vector2d::Zero(); // position in polar coordinates (r, theta)
  Eigen::Vector2d pos =
      Eigen::Vector2d::Zero(); // position in Cartesian coordinates (x, y)
  Eigen::Vector2d vel =
      Eigen::Vector2d::Zero(); // velocity in Cartesian coordinates (vx, vy)
  double vel_std = 0.0;        // standard deviation of the leader's velocity
  double pos_std = 0.0;        // standard deviation of the leader's position
  double spd = 0.0;            // speed of the leader
};

Obstacle from_obstacle(const PerceptionObstacle &obstacle) {
  Obstacle obs;
  obs.near_pos =
      Eigen::Vector2d(obstacle.closest_point.x, obstacle.closest_point.y);
  obs.near_pol = Eigen::Vector2d(
      obs.near_pos.norm(), std::atan2(obs.near_pos.y(), obs.near_pos.x()));
  obs.los_pos = Eigen::Vector2d(obstacle.line_of_sight_point.x,
                                obstacle.line_of_sight_point.y);
  obs.los_pol = Eigen::Vector2d(obs.los_pos.norm(),
                                std::atan2(obs.los_pos.y(), obs.los_pos.x()));
  obs.vel = Eigen::Vector2d(obstacle.velocity.x, obstacle.velocity.y);
  obs.vel_std = obstacle.velocity_uncertainty;
  obs.pos_std = obstacle.position_uncertainty;
  obs.spd = obs.vel.norm();
  if (obs.vel.norm() < obstacle.velocity_uncertainty * 2.0) {
    obs.vel = Eigen::Vector2d::Zero();
  }
  return obs;
}

struct ObstacleList {
  std::array<Obstacle, MAX_OBSTACLES> obstacles;
  int size = 0;
  void clear() { size = 0; }
  void fill_from_obstacles(const PerceptionObstacleArray &msg,
                           std::array<uint8_t, MAX_OBSTACLES> order) {
    clear();
    for (size_t i = 0; i < msg.obstacles.size() && i < MAX_OBSTACLES; ++i) {
      if (order[i] == 255)
        break;
      obstacles[i] = from_obstacle(msg.obstacles[order[i]]);
      size++;
    }
  }
};

class LeaderFollower : public rclcpp::Node {
private:
  Leader leader;

  bool last_path_valid = false;
  Eigen::Matrix<double, 3, PREDICTION_HORIZON> last_path;

  ObstacleList obstacles;
  rclcpp::Time obstacles_stamp;
  rclcpp::Time leader_stamp;

  Command cmd;

  bool moving = false;

  double dt;
  rclcpp::TimerBase::SharedPtr timer;
  rclcpp::Publisher<Request>::SharedPtr pub;
  rclcpp::Subscription<PerceptionLeader>::SharedPtr leader_sub;
  rclcpp::Subscription<PerceptionObstacleArray>::SharedPtr obstacles_sub;

  std::vector<Triplet> A_c, P_c;
  std::vector<OSQPFloat> l, u, q;

  OSQPSolver *solver = nullptr;

public:
  LeaderFollower() : Node("leader_follower") {
    // Initialize parameters
    pub = create_publisher<Request>("/api/sport/request", 10);
    leader_sub = create_subscription<PerceptionLeader>(
        "/perception/leader", 10,
        [this](const PerceptionLeader::SharedPtr msg) {
          this->leader_callback(msg);
        });
    obstacles_sub = create_subscription<PerceptionObstacleArray>(
        "/perception/obstacles", 10,
        [this](const PerceptionObstacleArray::SharedPtr msg) {
          this->obstacles_callback(msg);
        });
    const double rate = declare_parameter("rate", 10.0);
    dt = 1.0 / rate;
    timer = create_wall_timer(duration<double>(dt),
                              [this]() { this->timer_callback(); });

    constexpr int N = PREDICTION_HORIZON, M = MAX_OBSTACLES;
    constexpr int QP_VARS = 3 * (N + 1) + 3 * N + 3 * M * N + 2 * N;
    constexpr int QP_ROWS = 3 + N * (11 + 5 * M);
    constexpr int QP_A_NNZ = 3 + N * (28 + 10 * M);
    constexpr int QP_P_NNZ = N * (18 + 3 * M);

    A_c.reserve(QP_A_NNZ);
    P_c.reserve(QP_P_NNZ);
    l.reserve(QP_ROWS);
    u.reserve(QP_ROWS);
    q.reserve(QP_VARS);
  }

  ~LeaderFollower() {
    if (solver)
      osqp_cleanup(solver);
  }

  void leader_callback(const PerceptionLeader::SharedPtr msg) {
    Eigen::Vector2d p(msg->closest_point.x, msg->closest_point.y);
    if (msg->status != PerceptionLeader::VISIBLE || !std::isfinite(p.x()) ||
        !std::isfinite(p.y())) {
      leader.has_leader = false;
      if (msg->lost_duration > 1.0) {
        last_path_valid = false;
      }
      return;
    }
    leader.pos = p;
    leader_stamp = msg->header.stamp;
    leader.pol = Eigen::Vector2d(p.norm(), std::atan2(p.y(), p.x()));
    leader.vel = Eigen::Vector2d(msg->velocity.x, msg->velocity.y);
    leader.vel_std = msg->velocity_uncertainty;
    leader.pos_std = msg->position_uncertainty;
    leader.spd = leader.vel.norm();
    if (leader.spd < leader.vel_std * 2.0) {
      leader.vel = Eigen::Vector2d::Zero();
    }
    leader.has_leader = true;
  }

  void obstacles_callback(const PerceptionObstacleArray::SharedPtr msg) {
    std::array<uint8_t, MAX_OBSTACLES> d_order;
    std::array<double, MAX_OBSTACLES> d_distances;

    int count = 0;
    for (size_t i = 0; i < msg->obstacles.size(); ++i) {
      const PerceptionObstacle &obstacle = msg->obstacles[i];
      if (!std::isfinite(obstacle.closest_point.x) ||
          !std::isfinite(obstacle.closest_point.y))
        continue;
      Obstacle o = from_obstacle(obstacle);
      double dist = std::hypot(o.near_pos.x(), o.near_pos.y());

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

    for (int j = count; j < MAX_OBSTACLES; ++j) {
      d_order[j] = 255;
      d_distances[j] = std::numeric_limits<double>::infinity();
    }

    obstacles.fill_from_obstacles(*msg, d_order);
    obstacles_stamp = msg->header.stamp;
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

  // Seconds from the perception image to when the command takes effect (one
  // control step later). Unset stamps are treated as fresh.
  double message_age(const rclcpp::Time &stamp) {
    if (stamp.nanoseconds() == 0)
      return dt;
    return std::max(0.0, (now() - stamp).seconds()) + dt;
  }

  Eigen::Matrix<double, 3, 3> B(double theta) {
    Eigen::Matrix<double, 3, 3> B;
    const double cos_t = std::cos(theta);
    const double sin_t = std::sin(theta);
    B << cos_t, -sin_t, 0.0, sin_t, cos_t, 0.0, 0.0, 0.0, 1.0;
    return B;
  }

  void compute_command() {
    cmd = Command();

    const double dt = this->dt;

    const int N = PREDICTION_HORIZON, M = MAX_OBSTACLES;
    const int NX = 3 * (N + 1), NU = NX + 3 * N, NS = NU + M * N;
    const int NL = NS + N * M, NM = NL + 2 * N;
    const int n_vars = NM + M * N; // x, u, s, l, m, h
    auto xi = [&](int r, int k) { return 3 * k + r; };
    auto ui = [&](int r, int k) { return NX + 3 * k + r; };
    auto si = [&](int i, int k) { return NU + M * k + i; };
    auto li = [&](int i, int k) { return NS + M * k + i; };
    auto mi = [&](int i, int k) { return NL + 2 * k + i; };
    auto hi = [&](int i, int k) { return NM + M * k + i; };

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

    const Eigen::Vector2d Q = Eigen::Vector2d(8.0, 0.25);
    const Eigen::Vector3d R = Eigen::Vector3d(5.0, 5.0, 1.0);
    const Eigen::Vector3d S =
        Eigen::Vector3d(3.0, 3.0, 1.0); // Δu weights: vx, vy, wz

    Eigen::Vector3d u_prev = Eigen::Vector3d::Zero();
    if (last_path_valid)
      u_prev = last_path.col(0);

    const Eigen::Vector3d x0 = Eigen::Vector3d::Zero();
    Eigen::Vector2d p = Eigen::Vector2d::Zero();
    double t = 0.0;

    double hard_margin = 1.0;
    double soft_margin = 2.0;
    double leader_margin = 2.0;
    double margin_cost = 50.0;
    double margin_cost_linear = 10.0;
    // exact penalty on breaking the hard margin; keeps the QP always feasible
    double hard_cost_quadratic = 1000.0;
    double hard_cost_linear = 1000.0;
    double max_vel_margin = 0.5; // cap on uncertainty growth along the horizon
    double min_sector_speed = 0.1; // below this the nominal has no direction
    double max_obstacle_age = 0.5;

    double los_margin = 1.0;
    double los_cost_linear = 10.0;
    double los_cost_quadratic = 20.0;

    const double phi = 1.4; // full-vision half-angle (~80°)
    const double sphi = std::sin(phi), cphi = std::cos(phi);
    const double w_cone_q = 30.0, w_cone_l = 6.0;  // outside full-vision cone
    const double w_back_q = 80.0, w_back_l = 16.0; // moving backwards

    const bool use_leader = leader.has_leader;
    const double leader_age = use_leader ? message_age(leader_stamp) : 0.0;

    int num_obstacles = obstacles.size;
    const double obs_age = message_age(obstacles_stamp);
    if (num_obstacles > 0 && obs_age > max_obstacle_age + dt) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "Obstacles are %.2f s old, ignoring them.", obs_age);
      num_obstacles = 0;
    }

    std::array<double, 6> leader_arr = {leader.pos.x(), leader.pos.y(),
                                        leader.vel.x(), leader.vel.y(),
                                        leader.pos_std, leader.vel_std};
    leader_arr[0] += leader_arr[2] * leader_age;
    leader_arr[1] += leader_arr[3] * leader_age;
    std::array<double, 4> speed_limit = {
        0.5, 0.2, 0.0,
        0.5}; // full_vision, partial_vision, poor_vision, heading_limit

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

      A_c.emplace_back(add_row(-speed_limit[3], speed_limit[3]), ui(2, k),
                       1.0); // wz limit
      if (last_path_valid) {
        int idx = std::min(k + 1, PREDICTION_HORIZON - 1);
        double cos_t = std::cos(t);
        double sin_t = std::sin(t);
        p(0) += last_path(0, idx) * dt * cos_t - last_path(1, idx) * dt * sin_t;
        p(1) += last_path(0, idx) * dt * sin_t + last_path(1, idx) * dt * cos_t;
        t += last_path(2, idx) * dt;
        double abs_dir =
            std::abs(std::atan2(last_path(1, idx), last_path(0, idx)));
        if (std::hypot(last_path(0, idx), last_path(1, idx)) < min_sector_speed)
          abs_dir = 0.0;     // standing still: solver noise has no direction
        if (abs_dir < 1.4) { // +/- 80 degrees full vision
          A_c.emplace_back(add_row(-speed_limit[2], speed_limit[0]), ui(0, k),
                           1.0); // vx limit
          A_c.emplace_back(add_row(-speed_limit[0], speed_limit[0]), ui(1, k),
                           1.0); // vy limit
        } else if (abs_dir <
                   2.3) { // +/- 130 degrees partial vision 100%-60% vision
          A_c.emplace_back(add_row(-speed_limit[2], speed_limit[1]), ui(0, k),
                           1.0); // vx limit
          A_c.emplace_back(add_row(-speed_limit[1], speed_limit[1]), ui(1, k),
                           1.0); // vy limit
        } else {                 // walking backwards
          A_c.emplace_back(add_row(-speed_limit[2], speed_limit[2]), ui(0, k),
                           1.0); // vx limit
          A_c.emplace_back(add_row(-speed_limit[2], speed_limit[2]), ui(1, k),
                           1.0); // vy limit
        }
      } else {
        A_c.emplace_back(add_row(-speed_limit[2], speed_limit[1]), ui(0, k),
                         1.0); // vx limit
        A_c.emplace_back(add_row(-speed_limit[1], speed_limit[1]), ui(1, k),
                         1.0); // vy limit
      }
      const int vx = ui(0, k), vy = ui(1, k);
      const int m_cone = mi(0, k), m_back = mi(1, k);

      // m_cone >= -sinφ·vx + cosφ·vy  and  m_cone >= -sinφ·vx - cosφ·vy
      // (distance of v outside the ±φ cone; outward normals of the two edges)
      int r = add_row(0.0, OSQP_INFTY);
      A_c.emplace_back(r, m_cone, 1.0);
      A_c.emplace_back(r, vx, sphi);
      A_c.emplace_back(r, vy, -cphi);

      r = add_row(0.0, OSQP_INFTY);
      A_c.emplace_back(r, m_cone, 1.0);
      A_c.emplace_back(r, vx, sphi);
      A_c.emplace_back(r, vy, cphi);

      // m_back >= -vx
      r = add_row(0.0, OSQP_INFTY);
      A_c.emplace_back(r, m_back, 1.0);
      A_c.emplace_back(r, vx, 1.0);

      // m >= 0
      A_c.emplace_back(add_row(0.0, OSQP_INFTY), m_cone, 1.0);
      A_c.emplace_back(add_row(0.0, OSQP_INFTY), m_back, 1.0);

      // cost: quadratic + linear (linear term = penalty even at small speeds)
      P_c.emplace_back(m_cone, m_cone, w_cone_q);
      P_c.emplace_back(m_back, m_back, w_back_q);
      q[m_cone] += w_cone_l;
      q[m_back] += w_back_l;

      leader_arr[0] += leader_arr[2] * dt;
      leader_arr[1] += leader_arr[3] * dt;
      leader_arr[4] += leader_arr[5] * dt;

      // Without a visible leader the leader terms are switched off (weights 0,
      // same sparsity) and only obstacle avoidance remains.
      Eigen::Vector2d leader_dir = Eigen::Vector2d::UnitX();
      double leader_dist = 0.0;
      const double Qd = use_leader ? Q(0) : 0.0;
      const double Qh = use_leader ? Q(1) : 0.0;
      double leader_cost = 0.0, leader_heading = 0.0;
      if (use_leader) {
        Eigen::Vector2d leader_pos(leader_arr[0], leader_arr[1]);
        Eigen::Vector2d leader_rel_pos = leader_pos - p;
        leader_dist = std::hypot(leader_rel_pos(0), leader_rel_pos(1));
        if (leader_dist > 1e-6)
          leader_dir = leader_rel_pos / leader_dist;
        leader_pos -= leader_dir * leader_arr[4];
        leader_cost = leader_dir.dot(leader_pos) - leader_margin;
        leader_heading = std::atan2(leader_dir(1), leader_dir(0));
        double leader_heading_err =
            std::remainder(leader_heading - t, 2.0 * M_PI);
        leader_heading = t + leader_heading_err;
      }
      int px = xi(0, k + 1), py = xi(1, k + 1);
      q[px] += -Qd * leader_dir(0) * leader_cost;
      q[py] += -Qd * leader_dir(1) * leader_cost;
      q[xi(2, k + 1)] += -Qh * leader_heading;

      P_c.emplace_back(px, px, Qd * leader_dir(0) * leader_dir(0));
      P_c.emplace_back(px, py, Qd * leader_dir(0) * leader_dir(1));
      P_c.emplace_back(py, py, Qd * leader_dir(1) * leader_dir(1));

      for (int r = 0; r < 3; ++r) {
        P_c.emplace_back(ui(r, k), ui(r, k), R(r)); // cost for control input
      }

      P_c.emplace_back(xi(2, k + 1), xi(2, k + 1), Qh); // cost for heading
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
        q[si(i, k)] += margin_cost_linear;
        const int h_ik = hi(i, k);
        P_c.emplace_back(h_ik, h_ik, hard_cost_quadratic);
        q[h_ik] += hard_cost_linear;
        const int l_ik = li(i, k);
        P_c.emplace_back(l_ik, l_ik, los_cost_quadratic);
        q[l_ik] += los_cost_linear;

        if (i < num_obstacles) {
          Obstacle &obs = obstacles.obstacles[i];
          const double t_pred = obs_age + dt * (k + 1);
          const double look_ahead = 1.0;
          const Eigen::Vector2d obs_vel = obs.vel;
          const Eigen::Vector2d a =
              obs.near_pos + obs_vel * t_pred; // predicted, absolute
          const Eigen::Vector2d ab =
              obs_vel * look_ahead; // swept segment a -> a+ab

          double tau = 0.0;
          const double ab2 = ab.squaredNorm();
          if (ab2 > 1e-9)
            tau = std::clamp((p - a).dot(ab) / ab2, 0.0, 1.0);
          Eigen::Vector2d obs_mov =
              a + tau * ab; // closest point of segment to p
          Eigen::Vector2d obs_pos = obs_mov - p;
          const double obs_dist = obs_pos.norm();

          Eigen::Vector2d obs_dir;
          if (obs_dist > 1e-6)
            obs_dir = obs_pos / obs_dist;
          else if (ab2 > 1e-9)
            obs_dir = Eigen::Vector2d(-ab.y(), ab.x())
                          .normalized(); // on the segment: push sideways
          else
            obs_dir = Eigen::Vector2d::UnitX();

          double obs_margin =
              obs.pos_std + std::min(obs.vel_std * t_pred, max_vel_margin);

          int row_idx = add_row(soft_margin + obs_margin - obs_dir.dot(obs_mov),
                                OSQP_INFTY);
          // soft_margin - obs_dir.dot(obs_mov) <= -obs_dir.dot(x_{k+1}) +
          // s_{i,k} <= inf
          A_c.emplace_back(row_idx, xi(0, k + 1), -obs_dir(0));
          A_c.emplace_back(row_idx, xi(1, k + 1), -obs_dir(1));
          A_c.emplace_back(row_idx, si(i, k), 1.0);
          A_c.emplace_back(row_idx, h_ik, 1.0);

          obs_pos = obs.los_pos;
          obs_pos(0) += obs_vel(0) * t_pred;
          obs_pos(1) += obs_vel(1) * t_pred;
          obs_mov = obs_pos;
          obs_pos(0) -= p(0);
          obs_pos(1) -= p(1);

          const double along = leader_dir.dot(obs_pos);
          if (use_leader && std::isfinite(along) && along > 0.0 &&
              along < leader_dist) {
            const Eigen::Vector2d n_left(-leader_dir(1), leader_dir(0));
            const double off = n_left.dot(obs_pos);
            double side = off > 0.0 ? 1.0 : -1.0;
            Eigen::Vector2d m = side * n_left;

            double los_margin_i = los_margin + obs_margin;
            int los_row = add_row(los_margin_i - m.dot(obs_mov), OSQP_INFTY);
            A_c.emplace_back(los_row, xi(0, k + 1), -m(0));
            A_c.emplace_back(los_row, xi(1, k + 1), -m(1));
            A_c.emplace_back(los_row, l_ik, 1.0);
          } else {
            int los_row =
                add_row(-OSQP_INFTY, OSQP_INFTY); // dummy, keeps sparsity fixed
            A_c.emplace_back(los_row, xi(0, k + 1), 0.0);
            A_c.emplace_back(los_row, xi(1, k + 1), 0.0);
            A_c.emplace_back(los_row, l_ik, 0.0);
          }
        } else {
          int row_idx = add_row(-OSQP_INFTY, OSQP_INFTY);
          A_c.emplace_back(row_idx, xi(0, k + 1),
                           0.0); // dummy constraint for unused obstacles
          A_c.emplace_back(row_idx, xi(1, k + 1), 0.0);
          A_c.emplace_back(row_idx, si(i, k), 0.0);
          A_c.emplace_back(row_idx, h_ik, 0.0);

          int los_row =
              add_row(-OSQP_INFTY, OSQP_INFTY); // dummy, keeps sparsity fixed
          A_c.emplace_back(los_row, xi(0, k + 1), 0.0);
          A_c.emplace_back(los_row, xi(1, k + 1), 0.0);
          A_c.emplace_back(los_row, l_ik, 0.0);
        }
        int s_row_idx = add_row(0.0, soft_margin - hard_margin);
        // 0.0 <= s_{i,k} <= soft_margin - hard_margin
        A_c.emplace_back(s_row_idx, si(i, k),
                         1.0); // s_{i,k} = 0 for unused obstacles

        A_c.emplace_back(add_row(0.0, OSQP_INFTY), l_ik, 1.0);
        A_c.emplace_back(add_row(0.0, OSQP_INFTY), h_ik, 1.0); // h_{i,k} >= 0
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
    if (status == OSQP_SOLVED_INACCURATE)
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "OSQP: solved inaccurate");
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
