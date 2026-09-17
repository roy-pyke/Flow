"""V2 contracts: model identity, persistence, comparison and resource limits."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import main


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "EXPERIMENTS", tmp_path)
    main.SIMULATIONS.clear()
    main.EXPOSURES.clear()
    yield TestClient(main.app)
    main.SIMULATIONS.clear()
    main.EXPOSURES.clear()


def request(**updates):
    return {"source": [-122.4149, 37.7599], "nx": 24, "ny": 20,
            "duration_s": 60, "frame_interval_s": 15, **updates}


def create(client, **updates):
    response = client.post("/api/simulations", json=request(**updates))
    assert response.status_code == 200, response.text
    return response.json()


def test_identity_persistence_and_method_diagnostics(client):
    explicit = create(client)
    implicit = create(client, method="backward_euler", backend="scipy", dt=10)
    crank = create(client, method="crank_nicolson", backend="scipy", dt=10, startup="rannacher")
    assert len({explicit["id"], implicit["id"], crank["id"]}) == 3
    assert implicit["parameters"]["schema_version"] == 2
    assert implicit["diagnostics"]["method"] == "backward_euler"
    assert implicit["diagnostics"]["linear_residual_max"] < 1e-10
    before = client.get(f'/api/simulations/{crank["id"]}/frames/3').json()
    main.SIMULATIONS.clear()
    after = client.get(f'/api/simulations/{crank["id"]}/frames/3').json()
    np.testing.assert_allclose(before["values"], after["values"], rtol=0, atol=0)
    assert main.SIMULATIONS[crank["id"]]["parameters"]["startup"] == "rannacher"


def test_open_boundary_balance_and_full_transport_parameters(client):
    sim = create(client, model="advection_diffusion", method="imex_euler", backend="numpy_scipy",
                 boundary="open", velocity=[2, -1], dt=10)
    assert sim["parameters"]["velocity"] == [2.0, -1.0]
    assert sim["diagnostics"]["relative_mass_balance_error"] < 1e-10
    main.SIMULATIONS.clear()
    assert client.get(f'/api/simulations/{sim["id"]}/frames/4').status_code == 200
    assert create(client, model="advection_diffusion", method="imex_euler", backend="numpy_scipy",
                  boundary="open", velocity=[-2, -1], dt=10)["id"] != sim["id"]


def test_same_time_signed_comparison_and_mismatched_grids(client):
    a = create(client)
    b = create(client, method="backward_euler", backend="scipy", dt=15)
    response = client.get(f'/api/simulations/{b["id"]}/frames/4', params={"reference_id": a["id"]})
    assert response.status_code == 200, response.text
    compared = response.json()
    reference = client.get(f'/api/simulations/{a["id"]}/frames/4').json()
    np.testing.assert_allclose(compared["difference"], np.array(compared["values"]) - np.array(reference["values"]))
    assert compared["reference_time_s"] == 60
    assert compared["difference_linf"] > 0
    other = create(client, nx=16)
    assert client.get(f'/api/simulations/{b["id"]}/frames/4', params={"reference_id": other["id"]}).status_code == 422


def test_dynamic_frame_index_and_negative_field_remains_diagnostic(client):
    sim = create(client, duration_s=90, frame_interval_s=1)
    assert len(sim["times"]) == 91
    assert client.get(f'/api/simulations/{sim["id"]}/frames/80').status_code == 200
    main.SIMULATIONS[sim["id"]]["sim"].frames[80, 0, 0] = -0.001
    frame = client.get(f'/api/simulations/{sim["id"]}/frames/80')
    assert frame.json()["min"] == -0.001
    response = client.post("/api/routes", json={"simulation_id": sim["id"], "frame": 80,
        "start": [-122.428, 37.7515], "end": [-122.403, 37.7675]})
    assert response.status_code == 422 and "negative" in response.json()["detail"]


@pytest.mark.parametrize("updates", [
    {"method": "crank_nicolson", "backend": "cpp"},
    {"model": "diffusion", "velocity": [1, 0]},
    {"model": "advection_diffusion", "method": "explicit_euler"},
    {"method": "backward_euler", "backend": "scipy", "startup": "rannacher"},
    {"dt": 1e-8}, {"duration_s": 3600, "frame_interval_s": 1},
    {"nx": 320, "ny": 320, "duration_s": 120, "frame_interval_s": 1},
    {"backend": "cpp", "boundary": "periodic"},
])
def test_invalid_combinations_and_work_budgets(client, updates):
    assert client.post("/api/simulations", json=request(**updates)).status_code == 422


def test_explicit_native_unavailable_is_not_silent_fallback(client, monkeypatch):
    monkeypatch.setattr(main.native, "status", lambda: {"available": False, "reason": "test missing"})
    assert client.post("/api/simulations", json=request(backend="cpp")).status_code == 422
    result = create(client, backend="auto")
    assert result["parameters"]["backend"] == "numpy"
    assert result["parameters"]["requested_backend"] == "auto"


def test_memory_budget_evicts_and_persistent_full_parameters_recompute(client, monkeypatch):
    monkeypatch.setattr(main, "SIMULATION_BUDGET_BYTES", 24 * 20 * 5 * 8 + 1)
    a = create(client)
    b = create(client, method="backward_euler", backend="scipy")
    assert a["id"] not in main.SIMULATIONS and b["id"] in main.SIMULATIONS
    assert client.get(f'/api/simulations/{a["id"]}/frames/4').status_code == 200
    assert len(main.SIMULATIONS) == 1


def test_export_keeps_numerical_provenance(client):
    sim = create(client, method="backward_euler", backend="scipy", dt=10)
    response = client.post("/api/experiments", json={"simulation_id": sim["id"], "frame": 4,
        "start": [-122.428, 37.7515], "end": [-122.403, 37.7675]})
    assert response.status_code == 200, response.text
    rows = main.export_rows(response.json())
    assert all(row["method"] == "backward_euler" and row["backend"] == "scipy" for row in rows)
    assert rows[0]["grid_nx"] == 24 and rows[0]["requested_dt_s"] == 10
def test_extreme_requested_step_is_rejected_without_overflow(client):
    response = client.post("/api/simulations", json={"source": [-122.4149, 37.7599], "dt": 1e-309})
    assert response.status_code == 422
    assert "budget" in response.json()["detail"]


def test_tiny_final_output_gap_does_not_control_every_implicit_step(client):
    response = client.post("/api/simulations", json={"source": [-122.4149, 37.7599],
        "method": "backward_euler", "backend": "scipy", "nx": 16, "ny": 16,
        "duration_s": 120.0001, "frame_interval_s": 30})
    assert response.status_code == 200
    assert response.json()["diagnostics"]["internal_steps"] == 5
