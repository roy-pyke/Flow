#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include "build_info.h"
#include "diffusion.h"

namespace py = pybind11;
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point started) {
  return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}
static void check_array(const py::array& array, int ndim, const char* name) {
  if (!array.dtype().is(py::dtype::of<double>())) throw py::type_error(std::string(name) + " must have native float64 dtype; no implicit cast is performed.");
  if (array.ndim() != ndim || !(array.flags() & py::array::c_style)) throw py::value_error(std::string(name) + " must be C-contiguous with the required dimensions.");
  const auto* data = static_cast<const double*>(array.data());
  if (reinterpret_cast<std::uintptr_t>(data) % alignof(double)) throw py::value_error(std::string(name) + " must be aligned for float64 access.");
  for (py::ssize_t i = 0; i < array.size(); ++i) if (!std::isfinite(data[i])) throw py::value_error(std::string(name) + " must be finite.");
}
static void check_parameters(const py::array& field, double dx, double dy, double kappa, double dt, bool single) {
  check_array(field, 2, "initial");
  if (field.shape(0) < 2 || field.shape(1) < 2) throw py::value_error("Each grid dimension must be at least two.");
  if (!std::isfinite(dx) || !std::isfinite(dy) || dx <= 0 || dy <= 0 || !std::isfinite(kappa) || kappa < 0) throw py::value_error("Grid spacings must be finite and positive; kappa must be finite and nonnegative.");
  const double limit = 1.0 / (2.0 * kappa * (1.0/(dx*dx) + 1.0/(dy*dy)));
  if (!std::isfinite(dt) || dt < 0 || (!single && dt == 0) || dt > limit*(1.0+1e-12)) throw py::value_error("Time step must satisfy the explicit diffusion stability bound.");
}

PYBIND11_MODULE(flow_cpp, m) {
  m.doc() = "Flow C++17 single-threaded conservative diffusion; no fast-math.";
  m.def("build_info", []() {
    py::dict d;
    d["source_hash"] = FLOW_SOURCE_HASH; d["compiler"] = FLOW_COMPILER;
    d["build_type"] = FLOW_BUILD_TYPE; d["compiler_flags"] = FLOW_BUILD_FLAGS;
    d["pybind11"] = FLOW_PYBIND_VERSION; d["version"] = "0.2.0";
    d["threads"] = 1; d["fast_math"] = false;
    return d;
  });
  m.def("diffusion_step", [](const py::array& initial, double dx, double dy, double kappa, double dt) {
    check_parameters(initial, dx, dy, kappa, dt, true);
    py::array_t<double> output({initial.shape(0), initial.shape(1)});
    const auto* input = static_cast<const double*>(initial.data());
    auto* result = output.mutable_data();
    const auto nx = initial.shape(1), ny = initial.shape(0);
    { py::gil_scoped_release release; flow::step(input, result, nx, ny, dx, dy, kappa, dt); }
    return output;
  }, py::arg("initial").noconvert(), py::arg("dx"), py::arg("dy"), py::arg("kappa"), py::arg("dt"));
  m.def("solve", [](const py::array& initial, const py::array& times, double dx, double dy, double kappa, double dt) {
    const auto started = Clock::now();
    check_parameters(initial, dx, dy, kappa, dt, false);
    check_array(times, 1, "times");
    if (times.size() == 0) throw py::value_error("Output times cannot be empty.");
    const auto* ts = static_cast<const double*>(times.data());
    for (py::ssize_t i = 0; i < times.size(); ++i) if (ts[i] < 0 || (i && ts[i] <= ts[i-1])) throw py::value_error("Output times must be nonnegative and strictly increasing.");
    const auto nt = times.shape(0), ny = initial.shape(0), nx = initial.shape(1);
    if (nt > std::numeric_limits<py::ssize_t>::max()/initial.size()/static_cast<py::ssize_t>(sizeof(double))) throw py::value_error("Output shape overflows addressable memory.");
    const double validation_ms = elapsed(started);
    const auto allocated = Clock::now();
    py::array_t<double> frames({nt, ny, nx});
    const double output_allocation_ms = elapsed(allocated);
    auto* result = frames.mutable_data();
    const auto* input = static_cast<const double*>(initial.data());
    flow::Diagnostics diag;
    { py::gil_scoped_release release; diag = flow::solve(input, ts, result, nt, nx, ny, dx, dy, kappa, dt); }
    py::dict d;
    d["internal_steps"] = diag.steps; d["dt_max_s"] = dt;
    d["cfl_limit_s"] = kappa == 0 ? py::none() : py::cast(1.0/(2.0*kappa*(1.0/(dx*dx)+1.0/(dy*dy))));
    d["initial_mass"] = diag.initial_mass; d["final_mass"] = diag.final_mass;
    d["relative_mass_drift"] = diag.relative_mass_drift;
    d["min_concentration"] = diag.minimum; d["max_concentration"] = diag.maximum;
    d["mass_by_output"] = diag.masses; d["min_by_output"] = diag.minima;
    d["max_by_output"] = diag.maxima; d["energy_by_output"] = diag.energies;
    d["actual_step_sizes_s"] = diag.step_sizes;
    d["diagnostic_scope"] = "initial_and_output_frames";
    d["dtype"] = "float64"; d["boundary"] = "zero_flux";
    d["kernel_ms"] = diag.kernel_ms; d["workspace_allocation_ms"] = diag.workspace_ms;
    d["output_allocation_ms"] = output_allocation_ms; d["binding_validation_ms"] = validation_ms;
    d["binding_ms"] = elapsed(started);
    d["output_bytes"] = frames.nbytes(); d["workspace_bytes"] = 2*initial.nbytes();
    return py::make_tuple(frames, d);
  }, py::arg("initial").noconvert(), py::arg("times").noconvert(), py::arg("dx"), py::arg("dy"), py::arg("kappa"), py::arg("dt"));
}
