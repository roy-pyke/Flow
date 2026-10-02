"""Reproduce manufactured PDE-to-decision counterexamples without V2 outputs.

Run: python -m scripts.run_decision_experiments --config CONFIG --output NEW_DIR

The near-critical lambda policies are explicit stress-test construction, not
random samples or evidence about how frequently practical decisions fail.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from numbers import Real

import numpy as np

from backend.app.diffusion import Grid
from backend.app.numerics.solver import clear_factor_cache, solve
from backend.app.observations import build_edge_observer
from backend.app.research.decision import (candidate_bounds, canonical_hash, corridor_network,
    enumerate_simple_paths, integral_error_bound, minimizers, path_costs, static_graph_interval_bound)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/research/decision_study.json"


def _real(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number, not a boolean.")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive.")
    return float(value)


def validate_config(config: dict) -> dict:
    required = {"schema_version", "title", "domain_bounds_m", "kappa_m2_s", "speed_mps",
                "mean_concentration", "mode_amplitude", "mode_y", "corridors", "method",
                "tie_tolerance_s", "cases"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("Decision configuration must contain exactly the documented top-level keys.")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Only decision schema version 1 is supported.")
    if not isinstance(config["title"], str) or not config["title"].strip():
        raise ValueError("Study title must be nonempty text.")
    if config["method"] != "backward_euler":
        raise ValueError("This manufactured modal protocol currently supports backward_euler only.")
    if type(config["mode_y"]) is not int or config["mode_y"] < 1:
        raise ValueError("mode_y must be a positive integer.")
    for key in ("kappa_m2_s", "speed_mps", "mean_concentration", "tie_tolerance_s"):
        _real(config[key], key, positive=True)
    amplitude = _real(config["mode_amplitude"], "mode_amplitude", positive=True)
    if not amplitude < config["mean_concentration"]:
        raise ValueError("The positive amplitude must be smaller than the mean, ensuring a positive field.")
    bounds = config["domain_bounds_m"]
    if not isinstance(bounds, list) or len(bounds) != 4:
        raise ValueError("domain_bounds_m requires four coordinates.")
    for value in bounds:
        _real(value, "domain_bounds_m coordinate")
    Grid(tuple(bounds), 2, 2)
    corridor = config["corridors"]
    if not isinstance(corridor, dict) or set(corridor) != {"upper_fraction", "lower_fraction"}:
        raise ValueError("Corridor configuration requires upper_fraction and lower_fraction.")
    for value in corridor.values():
        _real(value, "corridor fraction")
    corridor_network(bounds, **corridor)
    if not isinstance(config["cases"], list) or not config["cases"]:
        raise ValueError("At least one study case is required.")
    seen = set()
    for case in config["cases"]:
        base = {"id", "nx", "ny", "snapshot_s", "dt_s", "lambda_policy"}
        if not isinstance(case, dict) or not base <= set(case):
            raise ValueError("Each case requires identity, grid, time step, snapshot and lambda policy.")
        policy = case["lambda_policy"]
        extra = {"fixed": {"lambda_weight"}, "reference_tie": set(),
                 "reference_near_tie": {"relative_offset"}, "threshold_midpoint": set()}
        if policy not in extra or set(case) != base | extra[policy]:
            raise ValueError("Case fields do not match its lambda policy.")
        if not isinstance(case["id"], str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", case["id"]) or case["id"] in seen:
            raise ValueError("Case ids must be unique safe lowercase identifiers.")
        seen.add(case["id"])
        for key in ("nx", "ny"):
            if type(case[key]) is not int or not 2 <= case[key] <= 256:
                raise ValueError("Grid dimensions must be integers between 2 and 256 for this protocol.")
        if config["mode_y"] >= case["ny"]:
            raise ValueError("The Fourier mode must be resolved by every grid.")
        for key in ("snapshot_s", "dt_s"):
            _real(case[key], key, positive=True)
        if case["snapshot_s"] / case["dt_s"] > 100000:
            raise ValueError("Case exceeds the protocol's 100000-step budget.")
        if policy == "fixed" and _real(case["lambda_weight"], "lambda_weight") < 0:
            raise ValueError("lambda_weight must be finite and nonnegative.")
        if policy == "reference_near_tie" and not 0 < _real(case["relative_offset"], "relative_offset") < .01:
            raise ValueError("Near-tie relative_offset must be positive and less than .01.")
    return config


def exact_edge_exposures(network, *, wave_number: float, amplitude: float,
                         mean: float, ymin: float, speed_mps: float) -> np.ndarray:
    """Analytic integral of mean + amplitude*cos(k*(y-ymin)) on polylines."""
    result = []
    for edge in network.edges:
        points = np.asarray(edge["coordinates"])
        a, b = points[:-1], points[1:]
        length = np.linalg.norm(b - a, axis=1)
        phase = wave_number * ((a[:, 1] + b[:, 1]) / 2 - ymin)
        integral = mean + amplitude * np.cos(phase) * np.sinc(wave_number * (b[:, 1] - a[:, 1]) / (2 * np.pi))
        result.append(float(np.dot(length, integral) / speed_mps))
    return np.asarray(result)


def analytic_quadrature_bounds(network, grid: Grid, *, wave_number: float,
                               amplitude: float, speed_mps: float) -> np.ndarray:
    """Composite trapezoid error of the smooth exact field on each segment."""
    errors = []
    for edge in network.edges:
        vector = np.diff(np.asarray(edge["coordinates"]), axis=0)
        lengths = np.linalg.norm(vector, axis=1)
        valid = lengths > 0
        vector, lengths = vector[valid], lengths[valid]
        steps = lengths / np.maximum(1, np.ceil(lengths / (grid.h / 2)))
        directional_curvature = abs(amplitude) * wave_number**2 * (vector[:, 1] / lengths)**2
        errors.append(float(np.sum(lengths / speed_mps * steps**2 * directional_curvature / 12)))
    return np.asarray(errors)


def _case(config: dict, spec: dict, network, paths) -> tuple[dict, dict[str, np.ndarray]]:
    begin = perf_counter()
    grid = Grid(tuple(config["domain_bounds_m"]), spec["nx"], spec["ny"])
    speed, kappa, mean = config["speed_mps"], config["kappa_m2_s"], config["mean_concentration"]
    wave_number = config["mode_y"] * np.pi / (grid.bounds[3] - grid.bounds[1])
    cosine = np.broadcast_to(np.cos(wave_number * (grid.y[:, None] - grid.bounds[1])), (grid.ny, grid.nx))
    initial = mean + config["mode_amplitude"] * cosine
    initial_copy = initial.copy()
    clear_factor_cache()
    frames, diagnostics = solve(grid, initial, kappa, [spec["snapshot_s"]],
                                method=config["method"], backend="scipy", dt=spec["dt_s"], boundary="zero_flux")
    field = frames[-1]
    if not np.array_equal(initial, initial_copy):
        raise AssertionError("Production solver mutated the supplied initial field.")
    field_copy = field.copy()
    amplitude = config["mode_amplitude"] * np.exp(-kappa * wave_number**2 * spec["snapshot_s"])
    reference = mean + amplitude * cosine
    eigenvalue = -4 * kappa / grid.dy**2 * np.sin(wave_number * grid.dy / 2)**2
    semidiscrete = mean + config["mode_amplitude"] * np.exp(eigenvalue * spec["snapshot_s"]) * cosine
    observer = build_edge_observer(network, grid, boundary="zero_flux", speed_mps=speed, quadrature="trapezoid")
    numerical_exposure = observer.apply(field)
    if not np.array_equal(field, field_copy):
        raise AssertionError("Path observation mutated the computed field.")
    reference_exposure = exact_edge_exposures(network, wave_number=wave_number, amplitude=amplitude,
                                              mean=mean, ymin=grid.bounds[1], speed_mps=speed)
    edge_times = np.array([edge["length_m"] / speed for edge in network.edges])
    travel_times = path_costs(paths, edge_times)
    reference_route_exposure = path_costs(paths, reference_exposure)
    approximate_route_exposure = path_costs(paths, numerical_exposure)
    if len(paths.paths) != 2:
        raise ValueError("The threshold protocol requires exactly the two declared corridor paths.")
    time_difference = travel_times[0] - travel_times[1]
    reference_exposure_advantage = reference_route_exposure[1] - reference_route_exposure[0]
    approximate_exposure_advantage = approximate_route_exposure[1] - approximate_route_exposure[0]
    if min(time_difference, reference_exposure_advantage, approximate_exposure_advantage) <= 0:
        raise ValueError("The declared threshold construction requires a longer but cleaner first corridor.")
    exact_threshold = float(time_difference / reference_exposure_advantage)
    approximate_threshold = float(time_difference / approximate_exposure_advantage)
    policy = spec["lambda_policy"]
    weight = {"fixed": lambda: spec["lambda_weight"], "reference_tie": lambda: exact_threshold,
              "reference_near_tie": lambda: exact_threshold * (1 + spec["relative_offset"]),
              "threshold_midpoint": lambda: (exact_threshold + approximate_threshold) / 2}[policy]()
    approximate_cost = travel_times + weight * approximate_route_exposure
    reference_cost = travel_times + weight * reference_route_exposure
    selected = int(np.argmin(approximate_cost))
    optimal = minimizers(reference_cost, atol=config["tie_tolerance_s"])
    regret = float(max(0, reference_cost[selected] - reference_cost.min()))
    linf = float(np.max(np.abs(field - reference)))
    curvature = float(abs(amplitude) * wave_number**2)
    interpolation_sup_bound = curvature * grid.dy**2 / 8
    quadrature = analytic_quadrature_bounds(network, grid, wave_number=wave_number, amplitude=amplitude, speed_mps=speed)
    exposure_bounds = np.asarray(integral_error_bound(edge_times, linf + interpolation_sup_bound)["exposure"]) + quadrature
    cost_bounds = weight * path_costs(paths, exposure_bounds)
    candidate = candidate_bounds(approximate_cost, cost_bounds, selected, evidence="analytic_manufactured")
    graph = static_graph_interval_bound(network, "s", "t",
        edge_times + weight * np.maximum(0, numerical_exposure - exposure_bounds),
        edge_times + weight * (numerical_exposure + exposure_bounds), paths.paths[selected],
        evidence="analytic_manufactured")
    observed_error = np.abs(numerical_exposure - reference_exposure)
    if np.any(observed_error > exposure_bounds + 1e-9) or regret > graph["regret_bound"] + 1e-9:
        raise AssertionError("Manufactured observation or regret bound failed.")
    row = {"id": spec["id"], "grid": grid.to_dict(), "snapshot_s": spec["snapshot_s"],
        "method": config["method"], "dt_s": spec["dt_s"], "lambda_weight": float(weight),
        "lambda_protocol": {"policy": policy, "reference_switch_lambda": exact_threshold,
            "numerical_switch_lambda": approximate_threshold,
            "declared_construction": "All listed cases are reported. Threshold midpoint deliberately targets the disagreement interval; no frequency or held-out validation claim."},
        "reference_minimizers": optimal, "selected": selected,
        "approximate_minimizers": minimizers(approximate_cost, atol=config["tie_tolerance_s"]),
        "selected_is_reference_optimal_within_tolerance": selected in optimal,
        "reference_cost_gap_s": float(abs(reference_cost[0] - reference_cost[1])),
        "reference_regret_s": regret,
        "reference_relative_regret": regret / float(reference_cost.min()),
        "reference_tie_tolerance_s": config["tie_tolerance_s"],
        "field_error": {"nodal_linf": linf, "nodal_rms": float(np.sqrt(np.mean((field-reference)**2))),
            "temporal_vs_exact_semidiscrete_linf": float(np.max(abs(field-semidiscrete))),
            "semidiscrete_vs_continuous_nodal_linf": float(np.max(abs(semidiscrete-reference))),
            "exact_field_interpolation_sup_bound": interpolation_sup_bound,
            "reconstructed_numerical_vs_continuous_sup_bound": linf + interpolation_sup_bound},
        "paths": [{"index": i, "edge_indices": list(path),
            "edge_identities": [[network.edges[e]["u"], network.edges[e]["v"], network.edges[e]["key"]] for e in path],
            "travel_time_s": float(travel_times[i]), "reference_exposure": float(reference_route_exposure[i]),
            "approximate_exposure": float(approximate_route_exposure[i]),
            "reference_cost_s": float(reference_cost[i]), "approximate_cost_s": float(approximate_cost[i]),
            "objective_error_bound_s": float(cost_bounds[i])} for i, path in enumerate(paths.paths)],
        "edge_error_bounds": {"numerical_exposure": numerical_exposure.tolist(), "exact_exposure": reference_exposure.tolist(),
            "absolute_observed_error": observed_error.tolist(), "analytic_quadrature_bound": quadrature.tolist(),
            "combined_exposure_bound": exposure_bounds.tolist()},
        "candidate_certificate": candidate, "static_graph_certificate": graph,
        "observer": observer.metadata, "solver_diagnostics": diagnostics,
        "input_and_output_fields_unmodified": True, "total_case_ms": (perf_counter() - begin) * 1000}
    raw = {"initial": initial, "numerical": field, "continuous_exact_nodes": reference,
           "semidiscrete_exact_nodes": semidiscrete, "numerical_edge_exposure": numerical_exposure,
           "continuous_exact_edge_exposure": reference_exposure, "edge_exposure_bound": exposure_bounds}
    return row, raw


def run_study(config: dict) -> tuple[dict, dict[str, np.ndarray], dict]:
    validate_config(config)
    begin = perf_counter()
    network, graph_data = corridor_network(config["domain_bounds_m"], **config["corridors"])
    paths = enumerate_simple_paths(network, "s", "t")
    cases, raw = [], {}
    for spec in config["cases"]:
        case, arrays = _case(config, spec, network, paths)
        cases.append(case)
        raw.update({f"{spec['id']}__{key}": value for key, value in arrays.items()})
    report = {"schema_version": 1, "title": config["title"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": canonical_hash(config),
        "model": {"pde": "c_t = kappa * Laplacian(c)", "boundary": "homogeneous_Neumann_zero_flux",
            "initial": "mean + amplitude*cos(mode_y*pi*(y-ymin)/Ly)",
            "exact": "mean + amplitude*exp(-kappa*(mode_y*pi/Ly)^2*t)*cos(mode_y*pi*(y-ymin)/Ly)",
            "concentration_units": "dimensionless_synthetic", "distance_units": "metres", "time_units": "seconds",
            "objective": "J(P)=travel_time_s(P)+lambda*integral_P c(snapshot,x) ds/speed",
            "decision_type": "static_frozen_field_at_snapshot; no waiting, time-varying exposure or dose model",
            "feasible_set": "the two simple directed corridor paths on the saved common graph"},
        "reference_hierarchy": {"continuous": "closed-form cosine diffusion and analytic polyline integrals",
            "semidiscrete": "exact discrete cosine eigenmode exp(-4*kappa/dy^2*sin(k*dy/2)^2*t)",
            "time_reference": "exact semidiscrete evolution, independent of time stepping",
            "spatial_refinement_reference": "not needed for this manufactured mode; no general fine-grid reference implementation is claimed",
            "arithmetic": "float64 evaluation; analytic expressions are exact mathematically, not interval-enclosed machine computations"},
        "bound_derivation": [
            "Positive bilinear reconstruction implies nodal-error amplification at most one.",
            "For this y-only cosine and reflecting constant extension, interpolation error <= dy^2*sup|c_yy|/8, including half cells at the walls.",
            "Positive trapezoid weights sum to edge travel time, so reconstruction contribution <= edge_time*(nodal_linf+interpolation_bound).",
            "Smooth analytic-field quadrature remainder on each straight segment <= (length/speed)*(segment_step^2/12)*sup|d2c/ds2|.",
            "Add these edge exposure bounds, multiply by lambda, and apply finite-set or nonnegative lower-graph inequalities."],
        "limitations": ["Manufactured synthetic field and local Cartesian graph; not observed air quality or health guidance.",
            "Near-critical lambda is deliberately constructed using both solutions; this is a counterexample suite, not an unbiased failure-rate estimate.",
            "Full nodal error is available only because the manufactured reference is known; this is not an operational estimator for unknown solutions.",
            "No interval arithmetic, model/input uncertainty, dynamic routing, adjoint estimator or general large-graph candidate coverage is certified.",
            "Per-case timings are diagnostic single runs, not a controlled comparative benchmark."],
        "oracle": paths.to_dict(network), "cases": cases, "elapsed_s": perf_counter() - begin}
    return report, raw, graph_data


def _json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path: Path, report: dict) -> None:
    rows = []
    for case in report["cases"]:
        for route in case["paths"]:
            rows.append({"case": case["id"], "lambda": case["lambda_weight"],
                "grid_nx": case["grid"]["nx"], "grid_ny": case["grid"]["ny"],
                "dt_s": case["dt_s"], "snapshot_s": case["snapshot_s"],
                "selected": case["selected"], "reference_minimizers": json.dumps(case["reference_minimizers"]),
                "nodal_linf": case["field_error"]["nodal_linf"], "reference_regret_s": case["reference_regret_s"],
                "candidate_regret_bound_s": case["candidate_certificate"]["interval_regret_bound"],
                "graph_regret_bound_s": case["static_graph_certificate"]["regret_bound"],
                **{k: json.dumps(v) if isinstance(v, list) else v for k, v in route.items()}})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, report: dict, raw: dict, network_data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cases = report["cases"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    chosen = cases[-1]
    field = raw[f"{chosen['id']}__numerical"]
    extent = chosen["grid"]["bounds"]
    img = axes[0].imshow(field, origin="lower", extent=[extent[0], extent[2], extent[1], extent[3]],
                         cmap="viridis", aspect="equal")
    for i, indices in enumerate(report["oracle"]["paths"]):
        points = np.concatenate([np.asarray(network_data["edges"][j]["coordinates"]) for j in indices["edge_indices"]])
        axes[0].plot(points[:, 0], points[:, 1], "o-", color=["#ff734f", "#f7dc72"][i], label=f"Corridor {i}", lw=2)
    axes[0].set(title=f"Common graph; {chosen['id']} field", xlabel="x (m)", ylabel="y (m)")
    axes[0].legend(fontsize=8)
    fig.colorbar(img, ax=axes[0], label="Synthetic concentration", shrink=.8)
    labels = [case["id"].replace("_", "\n") for case in cases]
    position = np.arange(len(cases))
    axes[1].bar(position, [c["field_error"]["nodal_linf"] for c in cases], color="#4169a2")
    axes[1].set_yscale("log")
    axes[1].set(title="Field error alone does not decide route quality", ylabel="Nodal Linf error", xticks=position, xticklabels=labels)
    axes[2].bar(position, [c["reference_regret_s"] for c in cases], color="#c96548", label="Actual reference regret")
    axes[2].plot(position, [c["static_graph_certificate"]["regret_bound"] for c in cases], "ko", ms=4, label="Conditional graph upper bound")
    axes[2].set(title="Same objective, evaluated against analytic field", ylabel="Objective regret (s)", xticks=position, xticklabels=labels)
    axes[2].legend(fontsize=8)
    for ax in axes[1:]:
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _markdown(report: dict) -> str:
    lines = ["# 扩散误差与路径决策：可复现反例实验", "",
        "这是实际调用生产 PDE 求解器和稀疏观测算子的制造解实验。所有案例使用相同的有向双走廊图，参考成本使用连续解析解沿完整折线路径的解析积分。", "",
        "| 案例 | 网格 | 场节点 Linf 误差 | λ | 选择 / 参考最优 | 参考 regret (s) | 图 regret 上界 (s) |",
        "|---|---:|---:|---:|---|---:|---:|"]
    for c in report["cases"]:
        lines.append(f"| {c['id']} | {c['grid']['nx']}×{c['grid']['ny']} | {c['field_error']['nodal_linf']:.6g} | {c['lambda_weight']:.10g} | {c['selected']} / {c['reference_minimizers']} | {c['reference_regret_s']:.8g} | {c['static_graph_certificate']['regret_bound']:.8g} |")
    lines += ["", "路径 0 为较长的上侧走廊，路径 1 为较短的下侧走廊。regret 是所选路径在参考目标上的成本减去该图参考最优成本；不同路径若同成本，则不是错误决策。", "",
        "**构造协议。** fixed 使用预先给定的 λ；reference_tie 使用解析参考的交点；reference_near_tie 使用交点乘以配置中的偏移；threshold_midpoint 使用数值和解析两个交点的中点，主动构造换路区间内的样本。所有配置案例均输出，不据此声称现实失败率。", "",
        "**误差分解。** 分别保存时间离散相对精确半离散解的误差、半离散相对连续解的节点误差、连续解到双线性重建的解析上界，以及平滑解析场的梯形求积上界。节点 Linf 不能直接视为连续线积分界；这里显式加入重建和求积项。", "",
        "**保证范围。** 对每个候选都有有效误差界且近似搜索 gap ≤ η 时，参考 regret ≤ 2ε+η。异质区间给出更细的上界；图证书用所选路径上界减去非负下界图的最短成本。数值浮点值没有做区间算术包围，因此输出是附带前提的解析界核验，不是机器严格证明。", "",
        "**边界。** 冻结场、合成浓度、静态图、固定速度。尚不覆盖真实污染推断、时变旅程、模型不确定性、动态搜索或未知真解下的自适应估计器。时间数据是单次诊断，不作为性能优越性证据。", "",
        "`report.json` 保存完整模型、路径身份、候选集 hash、边误差和所有前提；`paths.csv` 可逐行核对；`raw_fields.npz` 保存未经修改的初值、数值场和两层参考场；`manifest.json` 保存产物与代码 hash；`source_snapshot/` 保存本次使用的 Python 源码及依赖锁文件。", "",
        "```bash", "python -m scripts.run_decision_experiments --config config.json --output NEW_OUTPUT_DIRECTORY", "```", "",
        "在归档的 `source_snapshot/` 目录中执行以上命令时，配置路径使用 `../config.json`。输出目录必须不存在，避免覆盖已完成证据。", ""]
    return "\n".join(lines)


def write_bundle(config: dict, output: Path) -> dict:
    """Write a fresh bundle atomically; never overwrite existing evidence."""
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        report, raw, graph = run_study(config)
        import scipy
        source_paths = sorted((ROOT / "backend").rglob("*.py")) + [Path(__file__).resolve(), ROOT / "scripts/__init__.py", ROOT / "requirements.lock.txt"]
        sources = {}
        for source in source_paths:
            relative = source.relative_to(ROOT)
            destination = staging / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            sources[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False)
        report["runtime"] = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "machine": platform.machine(), "git_revision": git.stdout.strip() or None,
            "git_dirty": bool(dirty.stdout.strip()), "source_sha256": sources,
            "thread_environment": {key: os.environ.get(key) for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]}}
        _json(staging / "config.json", config)
        _json(staging / "network.json", graph)
        _json(staging / "report.json", report)
        np.savez_compressed(staging / "raw_fields.npz", **raw)
        _csv(staging / "paths.csv", report)
        _plot(staging / "decision_evidence.png", report, raw, graph)
        (staging / "REPORT.zh-CN.md").write_text(_markdown(report), encoding="utf-8")
        files = {str(path.relative_to(staging)): {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
                 for path in sorted(staging.rglob("*")) if path.is_file()}
        _json(staging / "manifest.json", {"schema_version": 1, "config_sha256": canonical_hash(config),
            "created_at_utc": report["created_at_utc"], "files": files,
            "manifest_scope": "All saved artifacts except this manifest; hashes verify integrity, not scientific truth."})
        staging.rename(output)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True, help="A new evidence directory; existing directories are rejected.")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = write_bundle(config, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "cases": len(report["cases"]),
                      "config_sha256": report["config_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
