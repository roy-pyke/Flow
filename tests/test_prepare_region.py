import pytest

from pyproj import Transformer
from shapely.geometry import shape
from scripts.prepare_region import prepare_region


def test_metric_rectangle_is_exactly_four_kilometers():
    region = prepare_region()
    xmin, ymin, xmax, ymax = region["bounds"]
    assert xmax - xmin == pytest.approx(4000)
    assert ymax - ymin == pytest.approx(4000)
    assert region["center_m"] == pytest.approx([551537.4753338977, 4179337.113915941])


def test_geographic_polygon_roundtrips_to_metric_bounds():
    region = prepare_region()
    project = Transformer.from_crs("EPSG:4326", region["crs"], always_xy=True)
    xmin, ymin, xmax, ymax = region["bounds"]
    for lon, lat in region["geometry"]["coordinates"][0]:
        x, y = project.transform(lon, lat)
        assert min(abs(x - xmin), abs(x - xmax)) < 1e-6
        assert min(abs(y - ymin), abs(y - ymax)) < 1e-6
    assert shape(region["download_geometry"]).contains(shape(region["geometry"]))
