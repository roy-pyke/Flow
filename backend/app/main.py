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
from pydantic import BaseModel, ConfigDict, Field
from pyproj import Transformer
from pyproj.exceptions import ProjError

from .diffusion import Grid, simulate
from .routing import RoadNetwork, shortest_path

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "data" / "demo"
EXPERIMENTS = ROOT / "data" / "experiments"
app = FastAPI(title="Flow · Diffusion & Route Lab", docs_url=None, redoc_url=None)
LOCK = threading.RLock()
SIMULATIONS: OrderedDict[str, dict] = OrderedDict()
EXPOSURES: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)


class SimulationRequest(StrictModel):
    source: tuple[float, float]
    kappa: float = Field(default=20.0, ge=5, le=50)


class RouteRequest(StrictModel):
    simulation_id: str = Field(pattern=r"^[a-f0-9]{20}$")
    frame: int = Field(default=20, ge=0, le=60)
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
        raise HTTPException(503, "Bundled road network missing. Run python -m scripts.prepare_network.")
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
    return hashlib.sha256(b"".join((ROOT / "backend" / "app" / name).read_bytes() for name in ["diffusion.py", "routing.py"])).hexdigest()


def simulation_summary(entry):
    sim = entry["sim"]
    return {"id":entry["id"],"times":sim.times_s.tolist(),"bounds":list(sim.grid.bounds),
            "geographic_corners":dataset()["corners"],"mass_drift":float(np.max(np.abs(sim.frames.sum(axis=(1,2)) / sim.frames[0].sum() - 1))),
            "elapsed_ms":entry["elapsed_ms"],"parameters":entry["parameters"],"diagnostics":sim.diagnostics,"data_version":dataset()["version"]}


def make_simulation(request: SimulationRequest):
    xy = point_xy(request.source)
    params = {"source":list(request.source),"kappa":request.kappa,"duration_s":1800,"frame_interval_s":30,
              "sigma_m":250,"nx":160,"ny":160,"data_version":dataset()["version"],"numerical_version":numerical_version()}
    sid = hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()[:20]
    with LOCK:
        if sid in SIMULATIONS:
            SIMULATIONS.move_to_end(sid)
            return SIMULATIONS[sid]
        started = time.perf_counter()
        sim = simulate(Grid(tuple(dataset()["bounds"])), source=xy,kappa=request.kappa)
        elapsed = (time.perf_counter()-started)*1000
        entry = {"id":sid,"sim":sim,"parameters":params,"elapsed_ms":elapsed}
        SIMULATIONS[sid] = entry
        if len(SIMULATIONS)>8:
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
        if saved and saved["data_version"] == dataset()["version"]:
            entry = make_simulation(SimulationRequest(**{key:saved["parameters"][key] for key in ("source","kappa")}))
            if entry["id"] == sid: return entry
    raise HTTPException(404,"Simulation not found or data/code changed. Run the simulation again.")


def get_exposures(entry, frame):
    if frame >= len(entry["sim"].times_s): raise HTTPException(422,"Frame is outside this simulation.")
    key = (entry["id"],frame)
    started = time.perf_counter()
    with LOCK:
        cached = key in EXPOSURES
        if not cached:
            EXPOSURES[key] = dataset()["network"].edge_exposures(entry["sim"].frames[frame], entry["sim"].grid)
            if len(EXPOSURES)>32: EXPOSURES.popitem(last=False)
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
            "data_version":d["version"],"basemap_url":"/data/demo/roads.pmtiles"}


@app.post("/api/simulations")
def new_simulation(request: SimulationRequest):
    return simulation_summary(make_simulation(request))


@app.get("/api/simulations/{sid}/frames/{frame}")
def simulation_frame(sid: str, frame: int):
    if not re.fullmatch(r"[a-f0-9]{20}",sid): raise HTTPException(404,"Simulation not found.")
    entry = get_simulation(sid)
    if not 0<=frame<len(entry["sim"].times_s): raise HTTPException(422,"Frame is outside this simulation.")
    values = entry["sim"].frames[frame]
    return {"time_s":float(entry["sim"].times_s[frame]),"values":values.tolist(),"min":float(values.min()),"max":float(values.max())}


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
    return {name:read_json(ROOT/"reports"/f"{name}.json") for name in ["data_quality","validation","analysis"]}


app.mount("/data/demo",StaticFiles(directory=DEMO,check_dir=False),name="data")
app.mount("/reports",StaticFiles(directory=ROOT/"reports",check_dir=False),name="reports")
app.mount("/",StaticFiles(directory=ROOT/"frontend"/"dist",html=True,check_dir=False),name="frontend")
