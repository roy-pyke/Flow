import networkx as nx
import numpy as np
import pytest

from backend.app.diffusion import Grid
from backend.app.routing import RoadNetwork, shortest_path
from backend.app.validation import validate_routing


@pytest.fixture
def network():
    nodes = [{"id": name, "x": x, "y": y, "lon": x, "lat": y}
             for name, x, y in [("a", 10, 10), ("b", 90, 10), ("c", 10, 90), ("d", 90, 90), ("isolated", 50, 50)]]
    return RoadNetwork.from_json({"crs": "EPSG:4326", "bounds": [0, 0, 100, 100], "nodes": nodes,
                                 "edges": [{"u": "a", "v": "b", "key": 0, "coordinates": [[10, 10], [90, 10]]},
                                           {"u": "a", "v": "b", "key": 1, "coordinates": [[10, 10], [50, 40], [90, 10]]},
                                           {"u": "b", "v": "d", "key": 0, "coordinates": [[90, 10], [90, 90]]},
                                           {"u": "a", "v": "c", "key": 0, "coordinates": [[10, 10], [10, 90]]},
                                           {"u": "c", "v": "d", "key": 0, "coordinates": [[10, 90], [90, 90]]}]})


@pytest.mark.parametrize("weight", [0, 0.5, 1, 2, 5, 10])
@pytest.mark.parametrize("algorithm", ["dijkstra", "astar"])
def test_directed_parallel_edge_costs_match_networkx(network, weight, algorithm):
    exposures = np.array([100, 0, 0, 200, 0])
    graph = nx.MultiDiGraph()
    for index, edge in enumerate(network.edges):
        graph.add_edge(edge["u"], edge["v"], key=index, weight=edge["length_m"] / 1.4 + weight * exposures[index])
    route = shortest_path(network, "a", "d", exposures, weight, algorithm)
    assert route["objective"] == pytest.approx(nx.shortest_path_length(graph, "a", "d", weight="weight"))
    if weight == 0:
        assert route["distance_m"] == pytest.approx(160)
    else:
        assert route["edge_indices"] == [1, 2]
        assert route["exposure"] == 0


def test_full_polyline_trapezoid_constant_integral_and_spacing(network):
    grid = Grid((0, 0, 100, 100), 10, 10)
    dose = network.edge_exposures(np.full((10, 10), 0.37), grid)
    expected = [0.37 * edge["length_m"] / 1.4 for edge in network.edges]
    np.testing.assert_allclose(dose, expected, rtol=1e-14)
    points, _, edge_indices = network._sampling(5)
    for index in range(len(network.edges)):
        assert np.linalg.norm(np.diff(points[edge_indices == index], axis=0), axis=1).max() <= 5 + 1e-12
    # The curved parallel edge is 100m, not the 80m endpoint distance.
    assert network.edges[1]["length_m"] == pytest.approx(100)
    assert len(shortest_path(network, "a", "b", np.array([100, 0, 0, 0, 0]), 10)["geometry"]["coordinates"]) == 3


def test_zero_concentration_recovers_shortest_distance(network):
    dose = network.edge_exposures(np.zeros((10, 10)), Grid((0, 0, 100, 100), 10, 10))
    route = shortest_path(network, "a", "d", dose, 10)
    assert route["distance_m"] == pytest.approx(160)
    assert route["exposure"] == 0


def test_same_endpoint_unreachable_direction_and_outside_click(network):
    route = shortest_path(network, "a", "a")
    assert route["distance_m"] == route["time_s"] == route["exposure"] == route["objective"] == 0
    assert route["edge_indices"] == []
    assert len(route["geometry"]["coordinates"]) == 2
    for destination in ("isolated",):
        with pytest.raises(ValueError, match="No directed"):
            shortest_path(network, "a", destination)
    with pytest.raises(ValueError, match="No directed"):
        shortest_path(network, "d", "a")
    with pytest.raises(ValueError, match="outside"):
        network.nearest_node(-0.1, 10)
    assert network.nearest_node(11, 11) == "a"
    with pytest.raises(ValueError, match="close enough"):
        network.nearest_node(0, 0, max_distance_m=1)


def test_negative_penalties_and_exposures_rejected(network):
    with pytest.raises(ValueError, match="nonnegative"):
        shortest_path(network, "a", "d", lambda_weight=-1)
    with pytest.raises(ValueError, match="nonnegative"):
        shortest_path(network, "a", "d", exposures=-np.ones(5))


def test_report_validates_integral_and_costs(network):
    grid = Grid((0, 0, 100, 100), 10, 10)
    report = validate_routing(network, grid, np.ones((10, 10)), "a", "d")
    assert report["passed"], report
    assert len(report["comparisons"]) == 12
