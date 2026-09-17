#include "diffusion.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace flow {
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
// This order matches NumPy's conservative face-flux additions/subtractions.
// All reads use input; only output is written. No boundary face flux is added.
void step(const double* input, double* output, std::size_t nx, std::size_t ny,
          double dx, double dy, double kappa, double dt) {
  if (kappa == 0 || dt == 0) {
    std::copy(input, input + nx * ny, output);
    return;
  }
  const double cx = kappa * dt / (dx * dx), cy = kappa * dt / (dy * dy);
  for (std::size_t y = 0; y < ny; ++y) {
    for (std::size_t x = 0; x < nx; ++x) {
      const std::size_t i = y * nx + x;
      double value = input[i];
      if (x + 1 < nx) value += cx * (input[i + 1] - input[i]);
      if (x > 0) value -= cx * (input[i] - input[i - 1]);
      if (y + 1 < ny) value += cy * (input[i + nx] - input[i]);
      if (y > 0) value -= cy * (input[i] - input[i - nx]);
      output[i] = value;
    }
  }
}

struct Summary { double mass, minimum, maximum, energy; };
static Summary summarize(const double* data, std::size_t count, double area) {
  // Compensated sums keep diagnostics meaningful for large grids.
  double sum = 0, correction = 0, squared = 0, correction2 = 0;
  double minimum = data[0], maximum = data[0];
  for (std::size_t i = 0; i < count; ++i) {
    const double value = data[i];
    if (!std::isfinite(value)) throw std::runtime_error("Nonfinite result during numerical integration.");
    const double add = value - correction, next = sum + add;
    correction = (next - sum) - add; sum = next;
    const double add2 = value * value - correction2, next2 = squared + add2;
    correction2 = (next2 - squared) - add2; squared = next2;
    minimum = std::min(minimum, value); maximum = std::max(maximum, value);
  }
  return {sum * area, minimum, maximum, 0.5 * squared * area};
}

Diagnostics solve(const double* initial, const double* times, double* frames,
                  std::size_t nt, std::size_t nx, std::size_t ny,
                  double dx, double dy, double kappa, double dt) {
  Diagnostics diag;
  const std::size_t count = nx * ny;
  auto started = Clock::now();
  std::vector<double> current(initial, initial + count), next(count);
  diag.workspace_ms = elapsed(started);
  started = Clock::now();
  const auto initial_stats = summarize(current.data(), count, dx * dy);
  diag.initial_mass = initial_stats.mass;
  diag.minimum = initial_stats.minimum; diag.maximum = initial_stats.maximum;
  double time = 0;
  for (std::size_t output = 0; output < nt; ++output) {
    const double target = times[output];
    const double origin = time;
    std::size_t segment_steps = 0;
    const double tolerance = 8 * std::numeric_limits<double>::epsilon() * std::max(std::numeric_limits<double>::min(), std::abs(target));
    while (target - time > tolerance) {
      const double actual_dt = std::min(dt, target - time);
      if (time + actual_dt == time) throw std::runtime_error("Time step cannot advance floating-point time.");
      step(current.data(), next.data(), nx, ny, dx, dy, kappa, actual_dt);
      current.swap(next);
      ++segment_steps;
      time = std::min(target, origin + segment_steps * dt);
      ++diag.steps;
      diag.step_sizes.push_back(actual_dt);
    }
    time = target;
    std::copy(current.begin(), current.end(), frames + output * count);
    const auto stats = summarize(current.data(), count, dx * dy);
    diag.final_mass = stats.mass;
    diag.masses.push_back(stats.mass); diag.minima.push_back(stats.minimum);
    diag.maxima.push_back(stats.maximum); diag.energies.push_back(stats.energy);
    diag.minimum = std::min(diag.minimum, stats.minimum);
    diag.maximum = std::max(diag.maximum, stats.maximum);
    const double denom = diag.initial_mass == 0 ? 1.0 : std::abs(diag.initial_mass);
    diag.relative_mass_drift = std::max(diag.relative_mass_drift, std::abs(stats.mass - diag.initial_mass) / denom);
  }
  diag.kernel_ms = elapsed(started);
  return diag;
}
}  // namespace flow
