#include "tmp_lidar/assignment.hpp"

#include <algorithm>
#include <limits>

namespace tmp_lidar
{

std::vector<int> solveAssignment(const std::vector<std::vector<double>> & cost, double forbidden)
{
  const int rows = static_cast<int>(cost.size());
  if (rows == 0) {return {};}
  const int cols = static_cast<int>(cost[0].size());
  std::vector<int> result(static_cast<std::size_t>(rows), -1);
  if (cols == 0) {return result;}

  // Square matrix padded with dummy entries; a forbidden pair costs more than leaving both unassigned.
  const int n = std::max(rows, cols);
  const double big = forbidden;
  auto at = [&](int r, int c) {
      if (r >= rows || c >= cols) {return 0.5 * big;}
      return std::min(cost[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)], big);
    };

  // e-maxx formulation, 1-based.
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> u(static_cast<std::size_t>(n + 1), 0.0), v(static_cast<std::size_t>(n + 1), 0.0);
  std::vector<int> p(static_cast<std::size_t>(n + 1), 0), way(static_cast<std::size_t>(n + 1), 0);
  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(static_cast<std::size_t>(n + 1), inf);
    std::vector<bool> used(static_cast<std::size_t>(n + 1), false);
    do {
      used[static_cast<std::size_t>(j0)] = true;
      const int i0 = p[static_cast<std::size_t>(j0)];
      double delta = inf;
      int j1 = 0;
      for (int j = 1; j <= n; ++j) {
        if (used[static_cast<std::size_t>(j)]) {continue;}
        const double cur = at(i0 - 1, j - 1) - u[static_cast<std::size_t>(i0)] - v[static_cast<std::size_t>(j)];
        if (cur < minv[static_cast<std::size_t>(j)]) {
          minv[static_cast<std::size_t>(j)] = cur;
          way[static_cast<std::size_t>(j)] = j0;
        }
        if (minv[static_cast<std::size_t>(j)] < delta) {
          delta = minv[static_cast<std::size_t>(j)];
          j1 = j;
        }
      }
      for (int j = 0; j <= n; ++j) {
        if (used[static_cast<std::size_t>(j)]) {
          u[static_cast<std::size_t>(p[static_cast<std::size_t>(j)])] += delta;
          v[static_cast<std::size_t>(j)] -= delta;
        } else {
          minv[static_cast<std::size_t>(j)] -= delta;
        }
      }
      j0 = j1;
    } while (p[static_cast<std::size_t>(j0)] != 0);
    do {
      const int j1 = way[static_cast<std::size_t>(j0)];
      p[static_cast<std::size_t>(j0)] = p[static_cast<std::size_t>(j1)];
      j0 = j1;
    } while (j0 != 0);
  }

  for (int j = 1; j <= n; ++j) {
    const int r = p[static_cast<std::size_t>(j)] - 1;
    const int c = j - 1;
    if (r < rows && c < cols && cost[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] < forbidden) {
      result[static_cast<std::size_t>(r)] = c;
    }
  }
  return result;
}

}  // namespace tmp_lidar
