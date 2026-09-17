"""Generate V2 numerical evidence from actual production solvers.

Run the native benchmark separately before this command; its measurements are
not silently regenerated while other experiments compete for CPU resources.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Keep the standalone experiment process reproducible before numerical imports.
for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[_name] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import scipy

from backend.app.numerics.validation import run_validation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/numerics"
NAMES = {"explicit_euler": "Forward Euler", "backward_euler": "Backward Euler",
         "crank_nicolson": "Crank–Nicolson", "crank_nicolson_rannacher": "CN + Rannacher",
         "advection_explicit": "Explicit upwind", "imex_euler": "IMEX Euler"}


def write_json(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_csv(name, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
                         for key, value in row.items()} for row in rows)


def save(fig, name):
    for ax in fig.axes:
        if ax.get_xscale() == "log":
            lo, hi = ax.get_xlim()
            ticks = [m * 10.0 ** exponent for exponent in range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
                     for m in [1, 2, 5] if lo <= m * 10.0 ** exponent <= hi]
            if len(ticks) > 7:
                ticks = ticks[::2]
            ax.set_xticks(ticks, [f"{value:g}" for value in ticks], fontsize=9)
            ax.xaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=180)
    plt.close(fig)


def validation_figures(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    rows = []
    for method, points in data["temporal"]["methods"].items():
        rows.extend({"method": method, **point} for point in points)
        for ax, norm in zip(axes, ["l2", "linf"]):
            ax.loglog([p["dt_s"] for p in points], [p[norm] for p in points], "o-", label=NAMES[method])
            ax.set(xlabel="Time step (s)", ylabel=f"{norm.upper()} error", title=f"Fixed grid · {norm.upper()}")
            ax.grid(True, which="both", alpha=.2)
    axes[0].legend(fontsize=9)
    save(fig, "temporal_convergence")
    write_csv("temporal_convergence.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    rows = []
    groups = {"semidiscrete": data["spatial"]["semidiscrete"], **data["spatial"]["production"]}
    for name, points in groups.items():
        rows.extend({"study": name, **point} for point in points)
        axes[0].loglog([p["h_m"] for p in points], [p["l2"] for p in points], "o-", label=NAMES.get(name, name))
    points = data["transport"]["spatial"]
    rows.extend({"study": "upwind_transport", **point} for point in points)
    axes[1].loglog([p["h_m"] for p in points], [p["l2"] for p in points], "o-", label="Upwind L2")
    axes[1].loglog([p["h_m"] for p in points], [p["linf"] for p in points], "s--", label="Upwind Linf")
    damped = data["transport"]["spatial_diffusion"]
    rows.extend({"study": "upwind_transport_with_diffusion", **point} for point in damped)
    axes[1].loglog([p["h_m"] for p in damped], [p["l2"] for p in damped], "^-", label="Transport + diffusion L2")
    for ax in axes:
        ax.set(xlabel="Grid spacing (m)", ylabel="Concentration error")
        ax.grid(True, which="both", alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_title("Diffusion: second order\nSemi-discrete T=.2; production T=.001")
    axes[1].set_title("Transport: first-order upwind\nPure and physically diffusive cases")
    save(fig, "spatial_convergence")
    write_csv("spatial_convergence.csv", rows)

    stability = data["stability"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    rows = []
    amp = stability["amplification"]
    for method in ["explicit_euler", "backward_euler", "crank_nicolson"]:
        axes[0].plot(amp["z"], amp[method], label=NAMES[method])
    axes[0].axhline(0, color="gray", lw=.7)
    axes[0].set(xlabel="z = Δt × eigenvalue", ylabel="Amplification factor", ylim=(-1.2, 1.2), title="Linear stability vs sign")
    for case in stability["cases"]:
        times = stability["output_times_s"]
        for i, t in enumerate(times):
            rows.append({"method": case["label"], "time_s": t, "dt_s": case["dt_s"],
                         "min": case["min_history"][i], "max": case["max_history"][i],
                         "energy": case["energy_history"][i], "mass_drift": case["mass_drift"]})
        axes[1].plot(times, case["min_history"], "o-", label=NAMES[case["label"]])
        axes[2].semilogy(times, case["energy_history"], "o-", label=NAMES[case["label"]])
    axes[1].set(xlabel="Time (s)", ylabel="Minimum concentration", title="Nonnegative spike initial condition")
    axes[2].set(xlabel="Time (s)", ylabel="Σ(c − mean)² × cell area", title="Variance energy decay")
    for ax in axes:
        ax.grid(True, alpha=.2)
        ax.legend(fontsize=7)
    save(fig, "stability_positivity")
    write_csv("stability_positivity.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    rows = data["transport"]["scenarios"]
    for boundary, marker in [("periodic", "o"), ("open", "s")]:
        selected = [(i, r) for i, r in enumerate(rows) if r["boundary"] == boundary]
        axes[0].semilogy([i for i, _ in selected], [max(r["mass_balance_relative"], 1e-17) for _, r in selected], marker, label=boundary)
    axes[0].axhline(1e-10, ls="--", color="#bf6231", label="Acceptance limit")
    axes[0].set(xlabel="Scenario index (wind / κ / method)", ylabel="Relative residual (zeros drawn at 1e−17)", title="32 boundary / wind / method cases")
    for method, points in data["transport"]["temporal"].items():
        axes[1].loglog([p["dt_s"] for p in points], [p["l2"] for p in points], "o-", label=NAMES[method])
    axes[1].set(xlabel="Time step (s)", ylabel="L2 error vs matrix exponential", title="First-order transport time accuracy")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=.2)
    save(fig, "transport_balance")
    write_csv("transport_balance.csv", rows)


def benchmark_figure():
    path = OUT / "benchmark.json"
    if not path.exists():
        raise RuntimeError("Run scripts.benchmark_numerics before generating the complete report.")
    data = json.loads(path.read_text())
    for case in data["cases_declared_before_measurement"]:
        for backend in ["numpy", "cpp"]:
            match = [r for r in data["results"] if r["case"]["name"] == case["name"] and r["backend"] == backend]
            if len(match) != 1 or match[0]["status"] != "complete":
                raise RuntimeError(f"Incomplete benchmark: {case['name']} / {backend}")
    for name, digest in data["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Stale benchmark source: {name}; rerun benchmark_numerics.")
    from backend.app.numerics.native import version
    current = version()
    if not current["available"] or current["source_hash"] != data["native_build"]["source_hash"]:
        raise RuntimeError("Native benchmark does not match the current build; rebuild and remeasure.")
    measured = [r for r in data["results"] if r.get("status") == "complete"]
    if not measured:
        # A status field is required, but preserve the benchmark's chosen label.
        measured = [r for r in data["results"] if r.get("median_total_ms") is not None]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for backend in ["numpy", "cpp"]:
        rows = [r for r in measured if r["backend"] == backend and r["case"]["output"] == "final_only"
                and r["case"]["duration_s"] == 1800]
        if rows:
            axes[0].loglog([r["case"]["n"] for r in rows], [r["median_total_ms"] for r in rows], "o-", label=backend)
    for backend in ["numpy", "cpp"]:
        rows = [r for r in measured if r["backend"] == backend]
        axes[1].plot(range(len(rows)), [r["median_total_ms"] for r in rows], "o-", label=backend)
        axes[1].set_xticks(range(len(rows)), [r["case"]["name"] for r in rows], rotation=25, ha="right", fontsize=8)
    axes[0].set(xlabel="Cells per axis", ylabel="Median full-call time (ms)", title="Identical FE · final frame only")
    axes[1].set(ylabel="Median full-call time (ms)", yscale="log", title="All measured workloads · including 61 frames")
    for ax in axes:
        ax.grid(True, which="both", alpha=.2)
        ax.legend()
    save(fig, "benchmark")
    return data


def study_figures(precision, routing):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, problem in zip(axes, ["diffusion", "transport"]):
        rows = [r for r in precision["rows"] if r["problem"] == problem]
        for method in dict.fromkeys(r["method"] for r in rows):
            for cache, style in [("cold", "o-"), ("warm", "s--")]:
                points = [r for r in rows if r["method"] == method and r["cache"] == cache]
                ax.loglog([r["l2"] for r in points], [r["median_total_ms"] for r in points], style,
                          markersize=4, label=f"{NAMES[method]} · {cache}")
        ax.set(xlabel="L2 error vs fixed-grid reference", ylabel="Median full-call time (ms)", title=f"{problem.title()} · 20 × 16 grid")
        ax.grid(True, which="both", alpha=.2)
        ax.legend(fontsize=7)
    save(fig, "work_precision")
    write_csv("work_precision.csv", [{**{k: v for k, v in row.items() if k != "samples"}, "sample_index": i,
                                      **sample} for row in precision["rows"] for i, sample in enumerate(row["samples"])])
    write_csv("tolerance_attainment.csv", precision["tolerance_attainment"])

    rows = [{"model": model["model"], **candidate} for model in routing["models"] for candidate in model["candidates"]]
    write_csv("routing_sensitivity.csv", rows)
    eligible = [r for r in rows if r["eligible"]]
    labels = [NAMES[r["method"]] for r in eligible]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, key, title, unit in zip(axes, ["field_l2", "edge_exposure_l2", "reference_regret_s"],
                                  ["Field error", "Edge exposure error", "Route regret in reference field"],
                                  ["RMS concentration", "RMS relative seconds", "Objective difference (s)"]):
        values = [r[key] for r in eligible]
        ax.bar(range(len(values)), values, color=["#087f8c"] * 3 + ["#bd6a29"] * 2)
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=8)
        ax.set(ylabel=unit, title=title)
        ax.grid(True, axis="y", alpha=.2)
        if key == "reference_regret_s" and max(abs(v) for v in values) < 1e-9:
            ax.set_ylim(-.01, .01)
            text = ("All candidates select the same path\nin this fixed scenario" if all(r["same_edge_path_as_reference"] for r in eligible)
                    else "Near-zero regret; some paths differ\nwithin tied numerical costs")
            ax.text(.5, .8, text, transform=ax.transAxes,
                    ha="center", fontsize=9)
    save(fig, "routing_sensitivity")


def conclusions(validation, precision, routing, benchmark):
    lines = ["# Measured findings", "",
             "These results are generated from the committed experiment scripts. They concern the recorded problems, machine and build; raw samples are preserved alongside the figures.", "",
             "## Accuracy, structure and counterexamples", "",
             f"All **{validation['summary']['order_checks']}** predeclared convergence/order and time-error-control checks passed. "
             f"Maximum normalized transport mass-balance residual across 32 boundary scenarios was **{validation['summary']['transport_max_relative_balance_error']:.3e}**.", "",
             "| Diffusion method | Final observed L2 time order |", "|---|---:|"]
    for method, rows in validation["temporal"]["methods"].items():
        lines.append(f"| {NAMES[method]} | {rows[-1]['order_l2']:.5f} |")
    cn = next(r for r in validation["stability"]["cases"] if r["label"] == "crank_nicolson")
    rannacher = next(r for r in validation["stability"]["cases"] if r["label"] == "crank_nicolson_rannacher")
    lines += ["", f"The stiff spike experiment produced CN minimum **{cn['minimum']:.5f}** despite decreasing variance energy. "
              f"With the tested Rannacher startup the minimum over the initial/output states was **{rannacher['minimum']:.5f}**. "
              "This demonstrates the difference between linear stability and positivity; it is not a proof of positivity for arbitrary Rannacher runs.", "",
              "IMEX remains first order and subject to the advection CFL. Upwind spatial convergence is approximately first order, including translated/decayed Fourier validation; its numerical diffusion is distinct from the physical κ. Open boundaries lose mass through modeled outflow while satisfying the discrete flux balance.", "",
              "## Implementation speed", "", "| Identical FE workload | NumPy median (ms) | C++ median (ms) | NumPy / C++ |",
              "|---|---:|---:|---:|"]
    for row in benchmark["comparisons"]:
        lines.append(f"| {row['case']} | {row['numpy_median_total_ms']:.4f} | {row['cpp_median_total_ms']:.4f} | {row['total_speedup_numpy_over_cpp']:.2f}× |")
    lines += ["", "The final-frame workloads show the benefit of a complete native loop and reusable buffers relative to the original vectorized NumPy FE implementation. "
              "Writing 61 frames reduces that advantage; a zero-duration call exposes native dispatch/build-validation overhead and can be slower. "
              "These timings exclude HTTP, JSON, browser rendering and the higher-level shared solver's additional normalized diagnostics. They do not measure an implicit-method speedup. See benchmark.json for exact timed scopes, raw samples, P90, memory and compiler flags.", "",
              "## Cost at the same error", "",
              "The table selects the fastest tested step satisfying each target, with all setup and solve work included. A dash means the tested steps did not attain the target; no extrapolated timing is substituted. The small fixed grid limits conclusions about large sparse systems.", "",
              "| Problem | LU cache | L2 target | Method | Fastest measured median (ms) |", "|---|---|---:|---|---:|"]
    for row in precision["tolerance_attainment"]:
        time_text = f"{row['median_total_ms']:.4f}" if row["attained"] else "—"
        lines.append(f"| {row['problem']} | {row['cache']} | {row['l2_tolerance']:.0e} | {NAMES[row['method']]} | {time_text} |")
    lines += ["", "BE's stability permits larger steps but does not remove its first-order truncation error. It is useful when damping and robustness matter and the chosen tolerance permits that error. "
              "For this smooth-mode study CN can reach tight errors with fewer steps; cold LU costs can dominate at small size. Compare the recorded target-specific timings instead of assuming every implicit solve is faster.", "",
              "## Effect on route decisions", "",
              "All candidates use the same graph, endpoints, source, frozen time, preference and fine road-sampling points. Candidate routes are re-evaluated in the finest reference field. The reference is numerical, with an additional refinement check; it is not observed truth.", "",
              "| Model | Reference refinement | Relative change in optimum cost | Candidate reference regrets (s) |", "|---|---|---:|---|"]
    for model in routing["models"]:
        ref = model["reference_refinement"]
        regrets = ", ".join(f"{NAMES[r['method']]}: {r['reference_regret_s']:.3e}" for r in model["candidates"] if r["eligible"])
        lines.append(f"| {model['model']} | {ref['coarse_n']}² → {ref['fine_n']}² | {ref['cost_relative_change']:.3e} | {regrets} |")
    same_paths = all(r["same_edge_path_as_reference"] for m in routing["models"] for r in m["candidates"] if r["eligible"])
    path_finding = ("In this scenario, measurable field and edge-exposure errors coexist with unchanged selected paths. " if same_paths
                    else "Some candidate paths differ from the reference path; the reported regret measures their actual cost consequence. ")
    lines += ["", path_finding +
              "That is a limited observation, not evidence that PDE accuracy never affects route choice. NetworkX checks agree independently for each frozen graph; graph correctness and PDE sensitivity answer different questions. "
              "Transport reference uncertainty is larger because the first-order upwind discretization converges more slowly.", "",
              "## Limits and reproducibility", "",
              "The public app uses a prescribed constant wind, synthetic relative concentration, a fixed rectangle and frozen-field walking exposure. "
              "There are no meteorological observations, calibration, buildings, reactions, traffic-flow PDEs or time-dependent route optimization. "
              "The native backend currently accelerates zero-flux explicit diffusion only; sparse implicit work uses SciPy/SuperLU. "
              "Interactive budgets limit each field array to 64 MiB, retained fields to 256 MiB, and retained LU arrays separately to 256 MiB; temporary allocations and the interpreter are additional. "
              "Larger experiments belong in scripts. Full commands and mathematical conventions are in [methods.md](methods.md), environment/source hashes in [summary.json](summary.json).", ""]
    (OUT / "conclusions.md").write_text("\n".join(lines))


def provenance():
    paths = sorted([*ROOT.glob("backend/app/numerics/*.py"), *ROOT.glob("cpp/src/*"),
                    *[ROOT / name for name in ["backend/app/diffusion.py", "backend/app/routing.py",
                    "cpp/CMakeLists.txt", "cpp/pyproject.toml", "cpp/source_hash.py", "requirements.lock.txt",
                    "scripts/run_numerical_experiments.py", "scripts/numerical_studies.py"]]])
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()},
            "environment": {"platform": platform.platform(), "python": platform.python_version(),
                            "numpy": np.__version__, "scipy": scipy.__version__}}


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Smaller study grids/repetitions, explicitly recorded")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "#f7f7f2", "axes.facecolor": "#f7f7f2",
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    validation = run_validation(quick=args.quick)
    write_json("validation.json", validation)
    validation_figures(validation)
    if not all([validation["summary"]["all_order_checks_pass"], validation["summary"]["transport_balance_pass"],
                validation["summary"]["cn_negative_example_observed"]]):
        raise RuntimeError("Numerical validation failed; inspect saved evidence.")
    from scripts.numerical_studies import run_work_precision, run_routing_sensitivity
    precision = run_work_precision(quick=args.quick)
    routing = run_routing_sensitivity(quick=args.quick)
    write_json("work_precision.json", precision)
    write_json("routing_sensitivity.json", routing)
    study_figures(precision, routing)
    benchmark = benchmark_figure()
    summary = {"schema_version": 2, "quick": args.quick, "validation": validation["summary"],
               "provenance": provenance(), "work_precision": {"measured_cases": len(precision["rows"]),
                   "raw_samples": sum(len(row["samples"]) for row in precision["rows"]),
                   "tolerance_attainment": precision["tolerance_attainment"]},
               "routing_sensitivity": {"models": [{"model": m["model"], "reference_refinement": m["reference_refinement"],
                  "max_reference_regret_s": max((r["reference_regret_s"] for r in m["candidates"] if r["eligible"]), default=None)} for m in routing["models"]]},
               "benchmark_comparisons": benchmark.get("comparisons", [])}
    write_json("summary.json", summary)
    conclusions(validation, precision, routing, benchmark)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
