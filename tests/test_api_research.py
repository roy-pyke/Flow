"""Boundary propagation and durable path identities in the existing API."""
import json

from fastapi.testclient import TestClient
import pytest

from backend.app import main


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "EXPERIMENTS", tmp_path)
    main.SIMULATIONS.clear()
    main.EXPOSURES.clear()
    with TestClient(main.app) as client:
        yield client


def create(client, boundary="zero_flux"):
    reply = client.post("/api/simulations", json={"source": [-122.4149, 37.7599],
        "nx": 16, "ny": 12, "duration_s": 30, "frame_interval_s": 30, "boundary": boundary})
    assert reply.status_code == 200, reply.text
    return {"simulation_id": reply.json()["id"], "frame": 1,
            "start": [-122.428, 37.7515], "end": [-122.403, 37.7675]}


def test_periodic_boundary_reaches_observer(client, monkeypatch):
    network = main.dataset()["network"]
    original = network.edge_exposures
    used = []
    def record(*args, **kwargs):
        used.append(kwargs.get("boundary"))
        return original(*args, **kwargs)
    monkeypatch.setattr(network, "edge_exposures", record)
    response = client.post("/api/routes", json=create(client, "periodic"))
    assert response.status_code == 200, response.text
    assert used == ["periodic"]


def test_sweep_keeps_paths_and_success_companion(client):
    response = client.post("/api/experiments", json=create(client))
    assert response.status_code == 200, response.text
    value = response.json()
    network = main.dataset()["network"]
    for row in value["rows"]:
        assert len(row["nodes"]) == len(row["edge_indices"])+1
        assert row["edge_ids"] == [{"u": network.edges[i]["u"], "v": network.edges[i]["v"],
                                     "key": str(network.edges[i]["key"]),
                                     "key_type": type(network.edges[i]["key"]).__name__}
                                    for i in row["edge_indices"]]
    assert (main.EXPERIMENTS / (value["id"]+".parquet")).is_file()
    saved = json.loads((main.EXPERIMENTS / (value["id"]+".json")).read_text())
    assert saved["rows"][0]["edge_ids"] == value["rows"][0]["edge_ids"]


def test_parquet_failure_never_publishes_experiment(client, monkeypatch):
    request = create(client)
    class FailedConnection:
        def register(self, *args): pass
        def execute(self, *args): raise OSError("disk write failed")
        def close(self): pass
    monkeypatch.setattr(main.duckdb, "connect", FailedConnection)
    with pytest.raises(OSError, match="disk write"):
        client.post("/api/experiments", json=request)
    assert client.get("/api/experiments").json() == []
    assert not list(main.EXPERIMENTS.glob("*.parquet"))
