"""Build a metric analysis rectangle and geographic OSM query polygon."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import box, mapping
from shapely.ops import transform

ROOT = Path(__file__).resolve().parents[1]


def prepare_region(config_path: Path | str = ROOT / "configs/region.json") -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    for key in ("width_m", "height_m"):
        if not 0 < float(config[key]) <= 100_000:
            raise ValueError(f"{key} must be positive and at most 100 km")
    if float(config.get("download_buffer_m", 0)) < 0:
        raise ValueError("download_buffer_m must be nonnegative")
    to_meters = Transformer.from_crs("EPSG:4326", config["crs"], always_xy=True)
    to_geographic = Transformer.from_crs(config["crs"], "EPSG:4326", always_xy=True)
    x, y = to_meters.transform(config["center_lon"], config["center_lat"], errcheck=True)
    w, h = config["width_m"] / 2, config["height_m"] / 2
    polygon = box(x - w, y - h, x + w, y + h)
    geographic = transform(
        to_geographic.transform, 
        polygon
    )
    buffered = transform(
        to_geographic.transform, 
        polygon.buffer(
            config.get("download_buffer_m", 0), 
            join_style="mitre"
        )
    )
    return {**config, 
            "center_m": [x, y], 
            "bounds": list(polygon.bounds),
            "geographic_bounds": list(geographic.bounds), 
            "geometry": mapping(geographic),
            "download_geometry": mapping(buffered)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/region.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/demo/region.json")
    args = parser.parse_args()
    region = prepare_region(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(region, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"center_m": region["center_m"], "bounds": region["bounds"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
