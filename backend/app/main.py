"""Loopback-only Flow API, persisted experiments, and local application assets."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import subprocess
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal
from uuid import uuid4

import duckdb
import numpy as np
import pyarrow as pa
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pyproj import Transformer
from pyproj.exceptions import ProjError

from .diffusion import Grid, Simulation, gaussian_initial
from .numerics import solve
from .numerics import native
from .routing import RoadNetwork, shortest_path

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "data" / "demo"
EXPERIMENTS = ROOT / "data" / "experiments"
app = FastAPI(title="Flow · Diffusion & Route Lab", docs_url=None, redoc_url=None)
LOCK = threading.RLock()
SIMULATIONS: OrderedDict[str, dict] = OrderedDict()
EXPOSURES: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
SIMULATION_BUDGET_BYTES = 256 * 1024 ** 2
EXPOSURE_BUDGET_BYTES = 32 * 1024 ** 2
METHOD_BACKENDS = {
    "explicit_euler": ["numpy", "cpp", "auto"],
    "backward_euler": ["scipy"],
    "crank_nicolson": ["scipy"],
    "advection_explicit": ["numpy"],
    "imex_euler": ["numpy_scipy"],
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)


class SimulationRequest(StrictModel):
    source: tuple[float, float]
    kappa: float = Field(default=20.0, ge=0, le=50)
    model: Literal["diffusion", "advection_diffusion"] = "diffusion"
    method: Literal["explicit_euler", "backward_euler", "crank_nicolson", "advection_explicit", "imex_euler"] = "explicit_euler"
    backend: Literal["numpy", "cpp", "scipy", "numpy_scipy", "auto"] = "numpy"
    nx: int = Field(default=160, ge=8, le=320)
    ny: int = Field(default=160, ge=8, le=320)
    dt: float | None = Field(default=None, gt=0)
    boundary: Literal["zero_flux", "periodic", "open"] = "zero_flux"
    velocity: tuple[float, float] = (0.0, 0.0)
    startup: Literal["none", "rannacher"] = "none"
    duration_s: float = Field(default=1800.0, ge=0, le=3600)
    frame_interval_s: float = Field(default=30.0, ge=1, le=3600)

    @model_validator(mode="after")
    def numerical_contract(self):
        if self.backend not in METHOD_BACKENDS[self.method]:
            raise ValueError("Unsupported method/backend combination.")
        diffusion = self.method in {"explicit_euler", "backward_euler", "crank_nicolson"}
        if (self.model == "diffusion") != diffusion:
            raise ValueError("Choose a method belonging to the selected model.")
        if any(abs(v) > 10 for v in self.velocity):
            raise ValueError("Wind components must be between -10 and 10 m/s.")
        if diffusion and (self.kappa <= 0 or any(self.velocity) or self.boundary == "open"):
            raise ValueError("Pure diffusion needs positive kappa, zero wind, and zero-flux or periodic boundaries.")
        if not diffusion and self.boundary not in {"open", "periodic"}:
            raise ValueError("Transport requires open or periodic boundaries.")
        if self.startup == "rannacher" and self.method != "crank_nicolson":
            raise ValueError("Rannacher startup is only available with Crank–Nicolson.")
        if self.backend == "cpp" and self.boundary != "zero_flux":
            raise ValueError("The native Euler kernel currently supports zero-flux walls.")
        frames = int(np.ceil(self.duration_s / self.frame_interval_s)) + 1
        if frames > 121 or frames * self.nx * self.ny * 8 > 64 * 1024 ** 2:
            raise ValueError("Interactive output exceeds the frame/memory budget; use fewer frames or a smaller grid.")
        return self


class RouteRequest(StrictModel):
    simulation_id: str = Field(pattern=r"^[a-f0-9]{20}$")
    frame: int = Field(default=20, ge=0)
    start: tuple[float, float]
    end: tuple[float, float]
    lambda_weight: float = Field(default=5, ge=0, le=100)
    algorithm: Literal["astar", "dijkstra"] = "astar"


def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


@lru_cache(maxsize=1)
def dataset():
    if not (DEMO / "network.json").exists():
        raise HTTPException(503, "Bundled road network missing. Run python scripts/prepare_network.py.")
    raw = read_json(DEMO / "network.json")
    network = RoadNetwork.from_json(raw)
    config = read_json(ROOT / "configs" / "region.json")
    bounds = raw["bounds"]
    crs = raw.get("crs", config["crs"])
    to_meters = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_geo = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    x0, y0, x1, y1 = bounds
    corners = [list(to_geo.transform(x, y)) for x, y in [(x0,y1),(x1,y1),(x1,y0),(x0,y0)]]
    geo_bounds = [min(p[0] for p in corners),min(p[1] for p in corners),max(p[0] for p in corners),max(p[1] for p in corners)]
    version = hashlib.sha256((DEMO / "network.json").read_bytes()).hexdigest()
    return {"network":network,"config":config,"bounds":bounds,"to_meters":to_meters,"corners":corners,"geo_bounds":geo_bounds,"version":version}


def point_xy(point):
    lon, lat = point
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise HTTPException(422, "Coordinates must be [longitude, latitude].")
    d = dataset()
    try:
        x, y = d["to_meters"].transform(lon, lat, errcheck=True)
    except ProjError as exc:
        raise HTTPException(422, "Select a point inside the study area.") from exc
    x0, y0, x1, y1 = d["bounds"]
    if not (x0 <= x <= x1 and y0 <= y <= y1):
        raise HTTPException(422, "Select a point inside the 4 × 4 km study area.")
    return x, y


@lru_cache(maxsize=1)
def code_version():
    try:
        commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
        return commit + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "source-archive"


def numerical_version():
    paths = [ROOT / "backend" / "app" / name for name in ["diffusion.py", "routing.py"]]
    paths += sorted((ROOT / "backend" / "app" / "numerics").glob("*.py"))
    paths += sorted((ROOT / "cpp" / "src").glob("*"))
    paths += [p for p in [ROOT / "cpp" / "CMakeLists.txt", ROOT / "cpp" / "pyproject.toml"] if p.exists()]
    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def simulation_summary(entry):
    sim = entry["sim"]
    return {"id":entry["id"],"times":sim.times_s.tolist(),"bounds":list(sim.grid.bounds),
            "geographic_corners":dataset()["corners"],"mass_drift":sim.diagnostics["relative_mass_drift"],
            "elapsed_ms":entry["elapsed_ms"],"parameters":entry["parameters"],"diagnostics":sim.diagnostics,"data_version":dataset()["version"]}


def make_simulation(request: SimulationRequest):
    xy = point_xy(request.source)
    params = request.model_dump(mode="json")
    actual_backend = request.backend
    native_info = native.status()
    if actual_backend == "auto":
        actual_backend = "cpp" if native_info["available"] and request.boundary == "zero_flux" else "numpy"
    if actual_backend == "cpp" and not native_info["available"]:
        raise HTTPException(422, "C++ backend unavailable. Build the native extension or select NumPy.")
    params.update(backend=actual_backend, requested_backend=request.backend,
                  sigma_m=250.0, schema_version=2, initial_condition="gaussian",
                  timestep_policy="automatic" if request.dt is None else "fixed_maximum",
                  data_version=dataset()["version"], numerical_version=numerical_version(),
                  native_build=native_info if actual_backend == "cpp" else None)
    sid = hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()[:20]
    with LOCK:
        if sid in SIMULATIONS:
            SIMULATIONS.move_to_end(sid)
            return SIMULATIONS[sid]
        started = time.perf_counter()
        grid = Grid(tuple(dataset()["bounds"]), request.nx, request.ny)
        times = np.append(np.arange(0, request.duration_s, request.frame_interval_s, dtype=np.float64), request.duration_s)
        advection_rate = abs(request.velocity[0]) / grid.dx + abs(request.velocity[1]) / grid.dy
        diffusion_rate = 2 * request.kappa * (grid.dx ** -2 + grid.dy ** -2)
        if request.method == "explicit_euler":
            automatic_dt = 0.9 / diffusion_rate
        elif request.method == "advection_explicit":
            automatic_dt = 0.9 / (advection_rate + diffusion_rate) if advection_rate + diffusion_rate else 30.0
        elif request.method == "imex_euler":
            automatic_dt = min(30.0, 0.9 / advection_rate) if advection_rate else 30.0
        else:
            automatic_dt = 30.0
        chosen = request.dt if request.dt is not None else automatic_dt
        # Compare before dividing: subnormal positive dt must yield a clear
        # budget error rather than an infinity-to-integer conversion failure.
        step_budget = 500_000_000 // (request.nx * request.ny)
        if request.duration_s > max(0, step_budget - len(times)) * chosen:
            raise HTTPException(422, "Interactive computation budget exceeded. Increase the step, reduce the grid/duration, or run the batch scripts.")
        try:
            frames, diagnostics = solve(grid, gaussian_initial(grid, xy), request.kappa, times,
                method=request.method, backend=actual_backend, dt=request.dt,
                boundary=request.boundary, velocity=request.velocity, startup=request.startup)
        except (ValueError, RuntimeError, ImportError, FloatingPointError) as exc:
            raise HTTPException(422, str(exc)) from exc
        sim = Simulation(grid, times, frames, params, diagnostics)
        elapsed = (time.perf_counter()-started)*1000
        entry = {"id":sid,"sim":sim,"parameters":params,"elapsed_ms":elapsed}
        SIMULATIONS[sid] = entry
        while len(SIMULATIONS)>8 or sum(item["sim"].frames.nbytes for item in SIMULATIONS.values()) > SIMULATION_BUDGET_BYTES:
            old, _ = SIMULATIONS.popitem(last=False)
            for key in list(EXPOSURES):
                if key[0] == old: del EXPOSURES[key]
        EXPERIMENTS.mkdir(parents=True,exist_ok=True)
        write_json(EXPERIMENTS / f"simulation-{sid}.json", {**simulation_summary(entry),"created_at":datetime.now(timezone.utc).isoformat(),"git_commit":code_version()})
        return entry


def get_simulation(sid):
    with LOCK:
        if sid in SIMULATIONS:
            SIMULATIONS.move_to_end(sid)
            return SIMULATIONS[sid]
        saved = read_json(EXPERIMENTS / f"simulation-{sid}.json")
        if saved and saved["data_version"] == dataset()["version"] and saved["parameters"].get("schema_version") == 2:
            values = {key: saved["parameters"][key] for key in SimulationRequest.model_fields if key in saved["parameters"]}
            values["backend"] = saved["parameters"].get("requested_backend", values["backend"])
            entry = make_simulation(SimulationRequest(**values))
            if entry["id"] == sid: return entry
    raise HTTPException(404,"Simulation not found or data/code changed. Run the simulation again.")


def get_exposures(entry, frame):
    if frame >= len(entry["sim"].times_s): raise HTTPException(422,"Frame is outside this simulation.")
    if float(entry["sim"].frames[frame].min()) < 0:
        raise HTTPException(422, "This field has negative concentrations. It remains available for diagnostics, but routing requires a nonnegative field; use a smaller step or a positivity-preserving method.")
    key = (entry["id"],frame)
    started = time.perf_counter()
    with LOCK:
        cached = key in EXPOSURES
        if not cached:
            EXPOSURES[key] = dataset()["network"].edge_exposures(entry["sim"].frames[frame], entry["sim"].grid)
            while len(EXPOSURES)>32 or sum(value.nbytes for value in EXPOSURES.values()) > EXPOSURE_BUDGET_BYTES:
                EXPOSURES.popitem(last=False)
        EXPOSURES.move_to_end(key)
        return EXPOSURES[key], (time.perf_counter()-started)*1000, cached


def export_rows(result):
    params = result["parameters"]
    sim = result["simulation"]["parameters"]
    provenance = {
        "experiment_id": result["id"], "simulation_id": params["simulation_id"],
        "frame": params["frame"], "frozen_time_s": result["simulation"]["times"][params["frame"]],
        "start_lon": params["start"][0], "start_lat": params["start"][1],
        "end_lon": params["end"][0], "end_lat": params["end"][1],
        "source_lon": sim["source"][0], "source_lat": sim["source"][1],
        "kappa": sim["kappa"], "algorithm": params["algorithm"],
        "model": sim.get("model", "diffusion"), "method": sim.get("method", "explicit_euler"),
        "backend": sim.get("backend", "numpy"), "boundary": sim.get("boundary", "zero_flux"),
        "grid_nx": sim.get("nx",160), "grid_ny": sim.get("ny",160),
        "requested_dt_s": sim.get("dt"), "startup": sim.get("startup","none"),
        "velocity_x_mps": sim.get("velocity",[0,0])[0], "velocity_y_mps": sim.get("velocity",[0,0])[1],
        "data_version": result["data_version"], "numerical_version": result["numerical_version"],
        "git_commit": result["git_commit"],
    }
    return [{**row, **provenance} for row in result["rows"]]


def route_json(route):
    return {"geojson":{"type":"Feature","properties":{},"geometry":route["geometry"]},
            "distance_m":route["distance_m"],"time_s":route["time_s"],"exposure":route["exposure"],
            "cost":route["objective"],"elapsed_ms":route["search_ms"],"visited":route["visited_nodes"]}


def compare(request):
    started = time.perf_counter()
    entry = get_simulation(request.simulation_id)
    network = dataset()["network"]
    try:
        a = network.nearest_node(*point_xy(request.start),max_distance_m=250)
        b = network.nearest_node(*point_xy(request.end),max_distance_m=250)
        exposures, integration_ms, cached = get_exposures(entry,request.frame)
        shortest = shortest_path(network,a,b,exposures,lambda_weight=0,algorithm=request.algorithm)
        weighted = shortest_path(network,a,b,exposures,lambda_weight=request.lambda_weight,algorithm=request.algorithm)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
    return {"shortest":route_json(shortest),"weighted":route_json(weighted),
            "snapped_nodes":{"start":a,"end":b},"time_s":float(entry["sim"].times_s[request.frame]),
            "timings":{"integration_ms":integration_ms,"integration_cached":cached,"search_ms":shortest["search_ms"]+weighted["search_ms"],"total_ms":(time.perf_counter()-started)*1000},
            "parameters":request.model_dump(),"data_version":dataset()["version"],
            "assumption":"Frozen relative concentration; exposure is a model integral in seconds, not a health prediction."}


@app.get("/api/health")
def health():
    return {"status":"ok","data_ready":(DEMO / "network.json").is_file()}


@app.get("/api/config")
def configuration():
    d = dataset()
    return {"region":{**d["config"],"bounds":d["bounds"],"geographic_bounds":d["geo_bounds"]},
            "defaults":{"source":[-122.4149,37.7599],"start":[-122.428,37.7515],"end":[-122.403,37.7675]},
            "data_version":d["version"],"basemap_url":"/data/demo/roads.pmtiles",
            "numerics":{"methods":METHOD_BACKENDS,"native":native.status(),"grids":[80,160,320],
                        "schema_version":2,"max_output_bytes":64*1024**2,"cache_budget_bytes":SIMULATION_BUDGET_BYTES}}


@app.post("/api/simulations")
def new_simulation(request: SimulationRequest):
    return simulation_summary(make_simulation(request))


@app.get("/api/simulations/{sid}/frames/{frame}")
def simulation_frame(sid: str, frame: int, reference_id: str | None = None):
    if not re.fullmatch(r"[a-f0-9]{20}",sid): raise HTTPException(404,"Simulation not found.")
    entry = get_simulation(sid)
    if not 0<=frame<len(entry["sim"].times_s): raise HTTPException(422,"Frame is outside this simulation.")
    values = entry["sim"].frames[frame]
    result = {"time_s":float(entry["sim"].times_s[frame]),"values":values.tolist(),"min":float(values.min()),"max":float(values.max())}
    if reference_id:
        if not re.fullmatch(r"[a-f0-9]{20}",reference_id): raise HTTPException(404,"Reference simulation not found.")
        reference = get_simulation(reference_id)["sim"]
        matches = np.flatnonzero(np.isclose(reference.times_s, result["time_s"], rtol=0, atol=1e-9))
        if reference.grid != entry["sim"].grid or len(matches) != 1:
            raise HTTPException(422,"Comparison requires the same grid, domain, and output time.")
        difference = values - reference.frames[int(matches[0])]
        result.update(difference=difference.tolist(),reference_id=reference_id,
                      reference_time_s=float(reference.times_s[int(matches[0])]),
                      difference_min=float(difference.min()),difference_max=float(difference.max()),
                      difference_l2=float(np.sqrt(np.mean(difference**2))),difference_linf=float(np.max(np.abs(difference))))
    return result


@app.post("/api/routes")
def routes(request: RouteRequest):
    return compare(request)


@app.post("/api/experiments")
def experiment(request: RouteRequest):
    rows=[]
    for lam in [0,0.5,1,2,5,10]:
        result=compare(request.model_copy(update={"lambda_weight":lam}))
        rows.append({"lambda_weight":lam,**{k:v for k,v in result["weighted"].items() if k!="geojson"}})
    eid=uuid4().hex
    result={"id":eid,"created_at":datetime.now(timezone.utc).isoformat(),"parameters":request.model_dump(),
            "simulation":simulation_summary(get_simulation(request.simulation_id)),"rows":rows,
            "git_commit":code_version(),"numerical_version":numerical_version(),"data_version":dataset()["version"],
            "interpretation":"Sampled trade-offs; not a complete Pareto frontier."}
    write_json(EXPERIMENTS / f"{eid}.json",result)
    connection=duckdb.connect()
    try:
        connection.register("experiment_rows", pa.Table.from_pylist(export_rows(result)))
        connection.execute("COPY experiment_rows TO ? (FORMAT PARQUET)",[str(EXPERIMENTS/f"{eid}.parquet")])
    finally: connection.close()
    return result


@app.get("/api/experiments")
def list_experiments():
    return [{"id":p.stem,"created_at":data["created_at"],"parameters":data["parameters"]}
            for p in sorted(EXPERIMENTS.glob("*.json"),reverse=True) if not p.name.startswith("simulation-")
            for data in [read_json(p)]]


@app.get("/api/experiments/{eid}/export")
def export_experiment(eid: str, format: Literal["json","csv"]="json"):
    if not re.fullmatch(r"[a-f0-9]{32}",eid): raise HTTPException(404,"Experiment not found.")
    result=read_json(EXPERIMENTS/f"{eid}.json")
    if result is None: raise HTTPException(404,"Experiment not found.")
    if format=="json":
        body=json.dumps(result,indent=2,allow_nan=False)
        media="application/json"
    else:
        output=io.StringIO()
        rows = export_rows(result)
        writer=csv.DictWriter(output,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
        body=output.getvalue(); media="text/csv"
    return Response(body,media_type=media,headers={"Content-Disposition":f'attachment; filename="flow-{eid}.{format}"'})


@app.get("/api/reports")
def reports():
    result = {name:read_json(ROOT/"reports"/f"{name}.json") for name in ["data_quality","validation","analysis"]}
    result["numerics"] = read_json(ROOT/"reports"/"numerics"/"summary.json")
    return result


app.mount("/data/demo",StaticFiles(directory=DEMO,check_dir=False),name="data")
app.mount("/reports",StaticFiles(directory=ROOT/"reports",check_dir=False),name="reports")
app.mount("/",StaticFiles(directory=ROOT/"frontend"/"dist",html=True,check_dir=False),name="frontend")
