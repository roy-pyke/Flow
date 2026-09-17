#pragma once
#include <cstddef>
#include <vector>

namespace flow {
struct Diagnostics {
  std::size_t steps = 0;
  double initial_mass = 0, final_mass = 0, relative_mass_drift = 0;
  double minimum = 0, maximum = 0;
  double workspace_ms = 0, kernel_ms = 0;
  std::vector<double> step_sizes, masses, minima, maxima, energies;
};
void step(const double* input, double* output, std::size_t nx, std::size_t ny,
          double dx, double dy, double kappa, double dt);
Diagnostics solve(const double* initial, const double* times, double* frames,
                  std::size_t nt, std::size_t nx, std::size_t ny,
                  double dx, double dy, double kappa, double dt);
}  // namespace flow
