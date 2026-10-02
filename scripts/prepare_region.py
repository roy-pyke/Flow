"""Build a validated metric analysis rectangle and geographic OSM query polygon."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

from pyproj import CRS, Transformer
from shapely.geometry import box, mapping
from shapely.ops import transform

ROOT = Path(__file__).resolve().parents[1]


def _finite_number(config: dict, key: str, default=None) -> float:
    value = config.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    return value


def prepare_region(config_path: Path | str = ROOT / "configs/region.json") -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Region configuration must be an object")
    if not isinstance(config.get("region_id"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", config["region_id"]):
        raise ValueError("region_id must contain only letters, digits, underscores and hyphens")
    for key in ("width_m", "height_m"):
        config[key] = _finite_number(config, key)
        if not 0 < config[key] <= 100_000:
            raise ValueError(f"{key} must be positive and at most 100 km")
    config["download_buffer_m"] = _finite_number(config, "download_buffer_m", 0)
    if not 0 <= config["download_buffer_m"] <= 100_000:
        raise ValueError("download_buffer_m must be nonnegative and at most 100 km")
    config["walk_speed_mps"] = _finite_number(config, "walk_speed_mps")
    if config["walk_speed_mps"] <= 0:
        raise ValueError("walk_speed_mps must be positive")
    config["center_lon"] = _finite_number(config, "center_lon")
    config["center_lat"] = _finite_number(config, "center_lat")
    if not -180 <= config["center_lon"] <= 180 or not -90 < config["center_lat"] < 90:
        raise ValueError("Region center must be a valid geographic longitude/latitude")
    config.setdefault("network_type", "walk")
    if config["network_type"] != "walk":
        raise ValueError("Only the walk network type is currently supported")
    crs = CRS.from_user_input(config["crs"])
    if not crs.is_projected or len(crs.axis_info) < 2 or any(not math.isclose(axis.unit_conversion_factor, 1.0) for axis in crs.axis_info[:2]):
        raise ValueError("Analysis CRS must be projected with metre units on both axes")
    to_meters = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_geographic = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    x, y = to_meters.transform(config["center_lon"], config["center_lat"], errcheck=True)
    if not all(math.isfinite(value) for value in (x, y)):
        raise ValueError("Projected region center must be finite")
    w, h = config["width_m"] / 2, config["height_m"] / 2
    polygon = box(x - w, y - h, x + w, y + h)
    geographic = transform(to_geographic.transform, polygon)
    buffered = transform(to_geographic.transform, polygon.buffer(config["download_buffer_m"], join_style="mitre"))
    if not all(geom.is_valid and not geom.is_empty and all(math.isfinite(v) for xy in geom.exterior.coords for v in xy) for geom in (geographic, buffered)):
        raise ValueError("Region cannot be represented by a finite geographic polygon")
    return {**config, "center_m": [x, y], "bounds": list(polygon.bounds),
            "geographic_bounds": list(geographic.bounds), "geometry": mapping(geographic),
            "download_geometry": mapping(buffered)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/region.json")
    parser.add_argument("--output", type=Path, required=True, help="Standalone preview JSON (not a published dataset)")
    args = parser.parse_args()
    region = prepare_region(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(region, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"center_m": region["center_m"], "bounds": region["bounds"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
