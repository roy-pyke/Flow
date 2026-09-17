"""Reproduce numerical validation, route sweeps, sensitivity, and algorithm checks."""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.routing import shortest_path
from backend.app.validation import validate_numerics

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/"reports"


def write_numerical_evidence(validation):
    """Keep the readable report and convergence plot tied to this run's data."""
    rows=validation["convergence"]
    sizes=np.array([row["n"] for row in rows],dtype=float)
    l2=np.array([row["l2_error"] for row in rows])
    linf=np.array([row["linf_error"] for row in rows])
    reference=l2[0]*(sizes[0]/sizes)**2
    with plt.rc_context({"figure.facecolor":"#f7f7f2","axes.facecolor":"#f7f7f2",
                         "axes.spines.top":False,"axes.spines.right":False}):
        fig,ax=plt.subplots(figsize=(8,5))
        ax.loglog(sizes,l2,"o-",color="#087f8c",label="L2 error")
        ax.loglog(sizes,linf,"s-",color="#bd6a29",label="L∞ error")
        ax.loglog(sizes,reference,"--",color="#667085",label="Second-order reference (N⁻²)")
        ax.set_xticks(sizes,labels=[str(int(size)) for size in sizes])
        ax.set(xlabel="Cells per axis, N (N × N grid)",ylabel="Concentration error",
               title="Zero-flux cosine solution · joint refinement")
        ax.grid(True,which="both",alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(REPORTS/"convergence.png",dpi=180)
        plt.close(fig)
    text="# Numerical validation\n\n"
    text+=f"All required checks: **{'PASS' if validation['passed'] else 'FAIL'}**.\n\n"
    text+="| Check | Result |\n|---|---|\n"
    for name,passed in validation["checks"].items():
        text+=f"| {name} | {'PASS' if passed else 'FAIL'} |\n"
    conservation=validation["conservation"]
    text+=f"\nMaximum relative mass drift: `{conservation['relative_mass_drift']:.3e}`. "
    text+=f"Constant-state maximum error: `{validation['constant_max_error']:.3e}`. "
    text+=f"Minimum concentration: `{conservation['min_concentration']:.3e}`. No negative-value clipping is used.\n\n"
    text+=f"Analytical solution: `{validation['analytic_solution']}` on the unit square, zero-flux walls, "
    text+=f"κ = 1, final time {validation['analytic_final_time']:g}.\n\n"
    text+="| Grid | L2 error | L∞ error | Observed joint order | Solve (ms) |\n|---|---|---|---|---|\n"
    for row in rows:
        order="—" if row["observed_order"] is None else f"{row['observed_order']:.5f}"
        text+=f"| {row['n']}² | {row['l2_error']:.7e} | {row['linf_error']:.7e} | {order} | {row['solve_ms']:.2f} |\n"
    text+=f"\n{validation['convergence_note']}\n\n![Cosine solution convergence](convergence.png)\n\n"
    text+="The spatial flux across each interior face is applied with opposite signs to its two neighboring cells. "
    text+="Exterior face fluxes are zero. Stability uses dt ≤ 0.9 / [2κ(1/dx² + 1/dy²)]. "
    text+="Concentration has no per-frame renormalization.\n\n"
    text+="Reproduce with `.venv/bin/python -m scripts.run_experiments`; raw diagnostics are in [validation.json](validation.json).\n"
    (REPORTS/"validation.md").write_text(text,encoding="utf-8")


def run():
    REPORTS.mkdir(exist_ok=True)
    validation=validate_numerics()
    main.write_json(REPORTS/"validation.json",validation)
    write_numerical_evidence(validation)
    assert validation["passed"],validation["checks"]
    client=TestClient(main.app)
    config=client.get("/api/config").json()
    defaults=config["defaults"]
    sensitivity=[]
    default_request=None
    for kappa in [5,20,50]:
        response=client.post("/api/simulations",json={"source":defaults["source"],"kappa":kappa})
        response.raise_for_status()
        sim=response.json()
        for frame in [0,20,60]:
            request={"simulation_id":sim["id"],"frame":frame,"start":defaults["start"],"end":defaults["end"],"lambda_weight":5}
            response=client.post("/api/routes",json=request)
            response.raise_for_status()
            result=response.json()
            sensitivity.append({"kappa":kappa,"time_s":sim["times"][frame],"simulation_ms":sim["elapsed_ms"],
                                "route_ms":result["timings"]["total_ms"],"integration_ms":result["timings"]["integration_ms"],
                                "search_ms":result["timings"]["search_ms"],"shortest_distance_m":result["shortest"]["distance_m"],
                                "shortest_exposure":result["shortest"]["exposure"],"weighted_distance_m":result["weighted"]["distance_m"],
                                "weighted_exposure":result["weighted"]["exposure"]})
            if kappa==20 and frame==20: default_request=request
    response=client.post("/api/experiments",json=default_request)
    response.raise_for_status()
    experiment=response.json()
    main.write_json(REPORTS/"example_experiment.json",experiment)
    entry=main.get_simulation(default_request["simulation_id"])
    network=main.dataset()["network"]
    exposures,_,_=main.get_exposures(entry,20)
    a=network.nearest_node(*main.point_xy(defaults["start"]))
    b=network.nearest_node(*main.point_xy(defaults["end"]))
    benchmark=[]
    for lam in [0,0.5,1,2,5,10]:
        for algorithm in ["dijkstra","astar"]:
            result=shortest_path(network,a,b,exposures,lambda_weight=lam,algorithm=algorithm)
            benchmark.append({"lambda_weight":lam,"algorithm":algorithm,"cost":result["objective"],"search_ms":result["search_ms"],"visited":result["visited_nodes"]})
    # Build a reference directed multigraph from the same geometry and edge exposures.
    raw=json.loads((ROOT/"data/demo/network.json").read_text())
    graph=nx.MultiDiGraph()
    for node in raw["nodes"]: graph.add_node(str(node["id"]))
    checks=[]
    for lam in [0,0.5,1,2,5,10]:
        graph.clear_edges()
        for i,edge in enumerate(network.edges):
            xy=np.asarray(edge["coordinates"])
            length=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum())
            graph.add_edge(edge["u"],edge["v"],key=edge["key"],weight=length/1.4+lam*exposures[i])
        reference=float(nx.shortest_path_length(graph,a,b,weight="weight"))
        own=[r["cost"] for r in benchmark if r["lambda_weight"]==lam]
        error=float(max(abs(value-reference) for value in own))
        checks.append({"lambda_weight":lam,"networkx_cost":reference,"max_cost_error":error,"passed":error<1e-7})
    assert all(c["passed"] for c in checks),checks
    analysis={"data_version":config["data_version"],"parameters":default_request,"defaults":defaults,
              "machine":{"system":platform.platform(),"machine":platform.machine(),"python":platform.python_version(),"processor":platform.processor()},
              "tradeoffs":experiment["rows"],"sensitivity":sensitivity,"benchmark":benchmark,"networkx_checks":checks,
              "performance_targets":{"simulation_under_5000_ms":max(r["simulation_ms"] for r in sensitivity)<5000,
                                     "route_under_500_ms":max(r["route_ms"] for r in sensitivity)<500},
              "interpretation":"Fixed OSM snapshot and endpoints; static fields. Sampled trade-offs, not a full Pareto frontier. Timings are measured on this machine; integration includes its first call."}
    main.write_json(REPORTS/"analysis.json",analysis)
    for name,rows in [("tradeoffs",experiment["rows"]),("sensitivity",sensitivity),("benchmark",benchmark)]:
        with (REPORTS/f"{name}.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({"figure.facecolor":"#f7f7f2","axes.facecolor":"#f7f7f2","axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(figsize=(8,5))
    rows=experiment["rows"]
    ax.plot([r["time_s"]/60 for r in rows],[r["exposure"] for r in rows],"o-",color="#087f8c")
    for r in rows: ax.annotate(f'λ={r["lambda_weight"]}',(r["time_s"]/60,r["exposure"]),xytext=(5,5),textcoords="offset points")
    ax.set(xlabel="Walking time (minutes)",ylabel="Frozen-field model exposure (s)",title="Mission · sampled route trade-offs")
    fig.tight_layout();fig.savefig(REPORTS/"tradeoffs.png",dpi=180);plt.close(fig)
    text="# Reproducible analysis\n\n"
    text+=f"Data SHA-256: `{config['data_version']}`. Environment: `{platform.platform()}`, Python {platform.python_version()}.\n\n"
    text+="All 6 lambda settings agree with NetworkX for both custom algorithms (absolute cost error < 1e-7).\n\n"
    text+="| κ (m²/s) | Frozen time (min) | Shortest exposure | Weighted exposure | Solve (ms) | Route total (ms) |\n|---|---|---|---|---|---|\n"
    for row in sensitivity:
        text+=f"| {row['kappa']} | {row['time_s']/60:g} | {row['shortest_exposure']:.3f} | {row['weighted_exposure']:.3f} | {row['simulation_ms']:.2f} | {row['route_ms']:.2f} |\n"
    text+="\n![Route trade-offs](tradeoffs.png)\n\nThe Gaussian broadens with diffusion; effects depend on source placement, coefficient and frozen time. Exposure is dimensionless relative concentration integrated over walking time, with no calibration to pollutants. No wind, buildings, terrain or time-evolving exposure along a walk is modeled. The reflecting rectangular boundary is a modeling assumption.\n\n"
    text+="Measurements include solver time, edge integration, and separate path-search timing in the JSON/CSV files. Browser transport/rendering is not included. Reproduce with `.venv/bin/python -m scripts.run_experiments`.\n"
    (REPORTS/"analysis.md").write_text(text)
    print(json.dumps({"validation":validation,"performance_targets":analysis["performance_targets"],"networkx_checks":checks},indent=2))


if __name__=="__main__":run()
