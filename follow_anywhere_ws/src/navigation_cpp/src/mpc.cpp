#include "mpc.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Sparse>

#include "osqp.h"

namespace navigation_cpp {

using SpMat = Eigen::SparseMatrix<OSQPFloat, Eigen::ColMajor, OSQPInt>;
using Triplet = Eigen::Triplet<OSQPFloat, OSQPInt>;

namespace {

// Points an OSQP csc matrix at a compressed Eigen sparse matrix (no copy).
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

} // namespace

Obstacle make_obstacle(double range, double bearing, double vx, double vy,
                       double std) {
  Obstacle o;
  o.polar = Eigen::Vector2d(range, bearing);
  // convert obs to Cartesian coordinates
  o.cart =
      Eigen::Vector2d(range * std::cos(bearing), range * std::sin(bearing));
  o.vel = Eigen::Vector2d(vx, vy);
  o.std = std;
  o.spd = o.vel.norm();
  return o;
}

void ObstacleList::push_nearest(const Obstacle &o) {
  if (size == MAX_OBSTACLES && o.polar[0] >= items[size - 1].polar[0]) {
    return;
  }
  // shift farther obstacles back by one, dropping the last one when full
  int i = size < MAX_OBSTACLES ? size++ : size - 1;
  for (; i > 0 && items[i - 1].polar[0] > o.polar[0]; --i) {
    items[i] = items[i - 1];
  }
  items[i] = o;
}

Eigen::Matrix2d A(const Leader &leader, const Eigen::Vector3d &speed) {
  const double t = leader.bearing;
  const double d = leader.range;
  const double vx = speed[0];
  const double vy = speed[1];
  const double vsin = vx * std::sin(t) - vy * std::cos(t);
  const double vcos = vx * std::cos(t) + vy * std::sin(t);

  Eigen::Matrix2d a;
  a << 0.0, vsin, vsin / d, vcos / d;
  return a;
}

Eigen::Matrix<double, 2, 3> B(const Leader &leader) {
  const double t = leader.bearing;
  const double d = leader.range;

  Eigen::Matrix<double, 2, 3> b;
  b << -std::cos(t), -std::sin(t), 0.0, std::sin(t) / d, -std::cos(t) / d, -1.0;
  return b;
}

std::optional<Eigen::MatrixXd>
compute_mpc(const Leader &leader, const ObstacleList &obstacles,
            const std::optional<Eigen::MatrixXd> &last_path, double dt) {
  const double margin = 1.0; // safety margin for obstacles
  const int N = 20;          // prediction horizon

  // A = A(leader, speed)
  const Eigen::Matrix<double, 2, 3> Bm = B(leader);

  const Eigen::Vector2d Q(8.0, 0.5);      // np.diag([8.0, 0.5])
  const Eigen::Vector3d R(7.5, 7.5, 0.5); // np.diag([7.5, 7.5, 0.5])

  // u = cp.Variable((3, N)); x = cp.Variable((2, N + 1))
  // stacked into one vector z = [x[:,0], ..., x[:,N], u[:,0], ..., u[:,N-1]]
  const int nx = 2 * (N + 1);
  const int n = nx + 3 * N;
  auto xi = [](int r, int k) { return r + 2 * k; };
  auto ui = [nx](int r, int k) { return nx + r + 3 * k; };

  // constraints: row by row into A_c with bounds l <= A_c z <= u
  std::vector<Triplet> A_c;
  std::vector<OSQPFloat> l, u;
  auto add_row = [&](OSQPFloat lo, OSQPFloat hi) {
    l.push_back(lo);
    u.push_back(hi);
    return static_cast<OSQPInt>(l.size() - 1);
  };

  // cost: sum quad_form(x[:,k+1], Q) + quad_form(u[:,k], R)
  // OSQP minimises 1/2 z'Pz, so P = 2 * blkdiag(...)
  std::vector<Triplet> P_c;

  // current state and reference
  // constraints += [x[:, 0] == [leader.range - 2.0, leader.bearing]]
  const Eigen::Vector2d x0(leader.range - 2.0, leader.bearing);
  for (int r = 0; r < 2; ++r) {
    A_c.emplace_back(add_row(x0[r], x0[r]), xi(r, 0), 1.0);
  }

  // accumulate the displacement over the prediction horizon
  Eigen::Vector2d vp = Eigen::Vector2d::Zero();
  // p = 0; p is linear in u, so keep it as its coefficients on u: p = Pk @
  // u.flatten('F')
  Eigen::MatrixXd Pk = Eigen::MatrixXd::Zero(2, 3 * N);
  double t = 0.0;

  const int n_obs = obstacles.size;
  Eigen::Vector2d leader_speed(leader.vx, leader.vy);
  const double sinb = std::sin(leader.bearing);
  const double cosb = std::cos(leader.bearing);
  const Eigen::Vector2d leader_change(
      leader_speed[0] * dt * cosb + leader_speed[1] * dt * sinb,
      (-leader_speed[0] * dt * sinb + leader_speed[1] * dt * cosb) /
          leader.range);
  // NOTE: same as python: zeroed after leader_change is computed, so this has
  // no effect
  if (leader_speed.norm() < leader.velocity_std * 2) {
    leader_speed.setZero();
  }

  for (int k = 0; k < N; ++k) {
    // constraints += [x[:, k + 1] == x[:, k] + dt * (B @ u[:, k])]
    for (int r = 0; r < 2; ++r) {
      const OSQPInt row = add_row(leader_change[r], leader_change[r]);
      A_c.emplace_back(row, xi(r, k + 1), 1.0);
      A_c.emplace_back(row, xi(r, k), -1.0);
      for (int j = 0; j < 3; ++j) {
        if (Bm(r, j) != 0.0) {
          A_c.emplace_back(row, ui(j, k), -dt * Bm(r, j));
        }
      }
    }

    // -max_vel <= u[i, k] <= max_vel
    const double max_vel[3] = {1.0, 1.0, 1.0};
    for (int i = 0; i < 3; ++i) {
      A_c.emplace_back(add_row(-max_vel[i], max_vel[i]), ui(i, k), 1.0);
    }

    // cost += cp.quad_form(x[:, k + 1], Q) + cp.quad_form(u[:, k], R)
    for (int r = 0; r < 2; ++r) {
      P_c.emplace_back(xi(r, k + 1), xi(r, k + 1), 2.0 * Q[r]);
    }
    for (int r = 0; r < 3; ++r) {
      P_c.emplace_back(ui(r, k), ui(r, k), 2.0 * R[r]);
    }

    if (n_obs > 0) {
      // NOTE: same as python: Rk uses t from before this step's update
      double cos_t = std::cos(t);
      double sin_t = std::sin(t);
      Eigen::Matrix2d Rk;
      Rk << cos_t, -sin_t, sin_t, cos_t;
      // accumulate the displacement over the prediction horizon
      if (last_path) {
        const Eigen::MatrixXd &lp = *last_path;
        const int idx = std::min(k + 1, static_cast<int>(lp.cols()) - 1);
        t += lp(2, idx) * dt; // accumulate the change in orientation over the
                              // prediction horizon
        cos_t = std::cos(t);
        sin_t = std::sin(t);
        vp +=
            Eigen::Vector2d(lp(0, idx) * dt * cos_t - lp(1, idx) * dt * sin_t,
                            lp(0, idx) * dt * sin_t + lp(1, idx) * dt * cos_t);
      }
      // p = p + dt * (Rk @ u[:2, k])
      Pk.block(0, 3 * k, 2, 2) += dt * Rk;
      for (int i = 0; i < n_obs; ++i) {
        const Obstacle &o = obstacles.items[i];
        // set the velocity to zero for obstacles with low speed relative to
        // their uncertainty
        const Eigen::Vector2d vel =
            o.spd < o.std * 2 ? Eigen::Vector2d::Zero().eval() : o.vel;
        // shift the obstacle by the estimated displacement
        const Eigen::Vector2d moved = o.cart + vel * ((k + 1) * dt);
        const Eigen::Vector2d shifted = moved - vp;

        // compute the distance to the shifted obstacle
        const double distance = shifted.norm();
        // NOTE: python divides by zero here (nan); skip that obstacle instead
        if (distance < 1e-9) {
          continue;
        }
        // direction vector to the shifted obstacle
        const Eigen::Vector2d direction = shifted / distance;
        // directions[i] @ (moved_obstacles[i] - p) >= margin
        // <=> -(directions[i] @ Pk) @ u >= margin - directions[i] @
        // moved_obstacles[i]
        const OSQPInt row = add_row(margin - direction.dot(moved), OSQP_INFTY);
        const Eigen::RowVectorXd coef = -direction.transpose() * Pk;
        for (int c = 0; c < 3 * (k + 1); ++c) {
          if (coef[c] != 0.0) {
            A_c.emplace_back(row, nx + c, coef[c]);
          }
        }
      }
    }
  }

  // problem = cp.Problem(cp.Minimize(cost), constraints); problem.solve()
  const OSQPInt m = static_cast<OSQPInt>(l.size());
  SpMat P(n, n), Acon(m, n);
  P.setFromTriplets(P_c.begin(),
                    P_c.end()); // diagonal, so already upper triangular
  Acon.setFromTriplets(A_c.begin(), A_c.end());
  OSQPCscMatrix P_csc = to_csc(P);
  OSQPCscMatrix A_csc = to_csc(Acon);
  std::vector<OSQPFloat> q(n, 0.0);

  OSQPSettings settings;
  osqp_set_default_settings(&settings);
  settings.verbose = 0;
  settings.warm_starting = 1;
  settings.polishing = 1;
  settings.eps_abs = 1e-4;
  settings.eps_rel = 1e-4;
  settings.max_iter = 4000;

  OSQPSolver *solver = nullptr;
  if (osqp_setup(&solver, &P_csc, q.data(), &A_csc, l.data(), u.data(), m, n,
                 &settings) != 0) {
    osqp_cleanup(solver);
    return std::nullopt;
  }

  // warm start u from the last solution
  if (last_path && last_path->rows() == 3 && last_path->cols() == N) {
    std::vector<OSQPFloat> z0(n, 0.0);
    for (int k = 0; k < N; ++k) {
      for (int r = 0; r < 3; ++r) {
        z0[ui(r, k)] = (*last_path)(r, k);
      }
    }
    osqp_warm_start(solver, z0.data(), nullptr);
  }

  osqp_solve(solver);

  std::optional<Eigen::MatrixXd> result;
  const OSQPInt status = solver->info->status_val;
  if (status == OSQP_SOLVED || status == OSQP_SOLVED_INACCURATE) {
    Eigen::MatrixXd u_value(3, N);
    for (int k = 0; k < N; ++k) {
      for (int r = 0; r < 3; ++r) {
        u_value(r, k) = solver->solution->x[ui(r, k)];
      }
    }
    result = u_value;
  }
  osqp_cleanup(solver);
  return result;
}

} // namespace navigation_cpp
