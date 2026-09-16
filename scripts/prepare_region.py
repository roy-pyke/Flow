import json
from pathlib import Path

from pyproj import Transfer

project_root = Path(__file__).resolve().parent[1]
config_path = project_root / "configs" / "region.json"

with config_path.open(encoding = "UTF-8") as file:
    config = json.load(file)

to_meters = Transfer.from_crs(
    crs_from = "EPSG:4326",
    crs_to = config["crs"],
    always_xy = True,
)

center_x_m, center_y_m = to_meters.tranform(
    config["center_lon"],
    config["center_lat"],
    errcheck = True
)

print(f"center x = {center_x_m:.3f} m")
print(f"center y = {center_y_m:.3f} m")