"""Exercise a complete simulation → route → persistent export on the bundled data."""
import csv
import io

import duckdb
import pytest
from fastapi.testclient import TestClient

from backend.app import main


@pytest.fixture()
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(main,"EXPERIMENTS",tmp_path)
    main.SIMULATIONS.clear()
    main.EXPOSURES.clear()
    return TestClient(main.app)


def scenario(client):
    config=client.get("/api/config")
    assert config.status_code==200
    defaults=config.json()["defaults"]
    result=client.post("/api/simulations",json={"source":defaults["source"],"kappa":20})
    assert result.status_code==200,result.text
    sim=result.json()
    assert len(sim["times"])==61
    assert sim["mass_drift"]<1e-10
    return {"simulation_id":sim["id"],"start":defaults["start"],"end":defaults["end"],"frame":20,"lambda_weight":5}


def test_end_to_end_persistence_and_exports(client):
    request=scenario(client)
    result=client.post("/api/routes",json=request)
    assert result.status_code==200,result.text
    routes=result.json()
    assert routes["weighted"]["exposure"]<=routes["shortest"]["exposure"]+1e-8
    assert routes["weighted"]["time_s"]>=routes["shortest"]["time_s"]-1e-8
    frame=client.get(f'/api/simulations/{request["simulation_id"]}/frames/20').json()
    assert len(frame["values"])==160
    exp=client.post("/api/experiments",json=request)
    assert exp.status_code==200,exp.text
    eid=exp.json()["id"]
    rows=exp.json()["rows"]
    assert [r["lambda_weight"] for r in rows]==[0,0.5,1,2,5,10]
    exposures=[r["exposure"] for r in rows]
    assert all(b<=a+1e-8 for a,b in zip(exposures,exposures[1:]))
    downloaded=client.get(f"/api/experiments/{eid}/export?format=csv")
    assert len(list(csv.DictReader(io.StringIO(downloaded.text))))==6
    assert client.get(f"/api/experiments/{eid}/export").json()["parameters"]==exp.json()["parameters"]
    with duckdb.connect() as conn:
        count=conn.execute("select count(*) from read_parquet(?)",[str(main.EXPERIMENTS/f"{eid}.parquet")]).fetchone()[0]
        assert count==6
    main.SIMULATIONS.clear()
    main.EXPOSURES.clear()
    assert client.post("/api/routes",json=request).status_code==200


def test_invalid_inputs_and_equal_points(client):
    request=scenario(client)
    assert client.post("/api/routes",json={**request,"start":[0,0]}).status_code==422
    assert client.post("/api/routes",json={**request,"frame":61}).status_code==422
    assert client.post("/api/routes",json={**request,"lambda_weight":-1}).status_code==422
    assert client.post("/api/simulations",json={"source":[0,0],"kappa":20}).status_code==422
    assert client.post("/api/simulations",json={"source":[-122.4149,37.7599],"kappa":51}).status_code==422
    assert client.get("/api/experiments/not-valid/export").status_code==404
    same=client.post("/api/routes",json={**request,"end":request["start"]})
    assert same.status_code==200,same.text
    assert same.json()["weighted"]["distance_m"]==0


def test_pmtiles_range_and_assets(client):
    response=client.get("/data/demo/roads.pmtiles",headers={"Range":"bytes=0-126"})
    assert response.status_code==206
    assert response.content.startswith(b"PMTiles")
    assert len(response.content)==127
    assert client.get("/data/demo/roads.geojson").json()["type"]=="FeatureCollection"


def test_default_simulation_restarts_and_projection_errors_are_client_errors(client):
    defaults=client.get("/api/config").json()["defaults"]
    response=client.post("/api/simulations",json={"source":defaults["source"]})
    assert response.status_code==200
    sid=response.json()["id"]
    main.SIMULATIONS.clear()
    assert client.get(f"/api/simulations/{sid}/frames/0").status_code==200
    assert client.post("/api/simulations",json={"source":[-33,0]}).status_code==422
