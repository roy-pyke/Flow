"""Audit GeoParquet with DuckDB Spatial and write reproducible quality reports."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import networkx as nx

from prepare_region import ROOT

EDGE_SQL = """SELECT count(*) AS directed_edges,
    sum(length_m) AS total_directed_length_m,
    sum(ST_Length(geometry)) AS geometry_directed_length_m,
    min(length_m) AS shortest_edge_m,
    max(length_m) AS longest_edge_m,
    count(*) FILTER (WHERE NOT ST_IsValid(geometry)) AS invalid_geometries,
    count(*) FILTER (WHERE ST_IsEmpty(geometry)) AS empty_geometries,
    count(*) FILTER (WHERE length_m <= 0) AS nonpositive_lengths,
    max(abs(length_m-ST_Length(geometry))) AS max_length_disagreement_m,
    count(DISTINCT component) AS weak_components FROM edges"""


def record(cursor) -> dict:
    return dict(zip([d[0] for d in cursor.description], cursor.fetchone()))


def report(install_spatial: bool = False) -> dict:
    demo, reports = ROOT / "data/demo", ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    data = json.loads((demo / "network.json").read_text())
    source = json.loads((demo / "source.json").read_text())
    # Install only during preparation, never hidden inside an API request.
    db = duckdb.connect(str(demo / "network.duckdb"))
    if install_spatial:
        db.execute("INSTALL spatial")
    db.execute("LOAD spatial")
    for table in ("nodes", "edges"):
        filename = str(demo / f"{table}.parquet").replace("'", "''")
        db.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_parquet('{filename}')")
    quality = record(db.execute(EDGE_SQL))
    quality["nodes"] = db.execute("SELECT count(*) FROM nodes").fetchone()[0]
    quality["component_sizes"] = [row[0] for row in db.execute("SELECT count(*) FROM nodes GROUP BY component ORDER BY count(*) DESC").fetchall()]
    quality["orphan_edge_endpoints"] = db.execute("SELECT count(*) FROM edges e LEFT JOIN nodes a ON e.u=a.id LEFT JOIN nodes b ON e.v=b.id WHERE a.id IS NULL OR b.id IS NULL").fetchone()[0]
    quality["duplicate_edge_ids"] = db.execute("SELECT count(*) FROM (SELECT u,v,key FROM edges GROUP BY ALL HAVING count(*)>1)").fetchone()[0]
    quality["parallel_edge_pairs"] = db.execute("SELECT count(*) FROM (SELECT u,v FROM edges GROUP BY ALL HAVING count(*)>1)").fetchone()[0]
    xmin, ymin, xmax, ymax = data["bounds"]
    quality["edges_outside_domain"] = db.execute("SELECT count(*) FROM edges WHERE NOT ST_Covers(ST_MakeEnvelope(?,?,?,?),geometry)", [xmin, ymin, xmax, ymax]).fetchone()[0]
    quality["self_loops"] = db.execute("SELECT count(*) FROM edges WHERE u=v").fetchone()[0]
    quality["unique_physical_polylines"] = len(json.loads((demo / "roads.geojson").read_text())["features"])
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(n["id"] for n in data["nodes"])
    graph.add_edges_from((e["u"], e["v"], e["key"]) for e in data["edges"])
    quality["strong_components"] = nx.number_strongly_connected_components(graph)
    quality["bidirectional_policy"] = source["walking_direction_policy"]
    db.close()
    passed = all(quality[key] == 0 for key in ("invalid_geometries", "empty_geometries", "nonpositive_lengths", "orphan_edge_endpoints", "duplicate_edge_ids", "edges_outside_domain")) and quality["max_length_disagreement_m"] < 1e-8
    report_data = {"generated_utc": datetime.now(timezone.utc).isoformat(), "status": "passed" if passed else "failed",
                   "region_id": data["region_id"], "crs": data["crs"], "bounds": data["bounds"],
                   "source_snapshot_sha256": data["source_snapshot_sha256"], "source": source,
                   "quality": quality, "cleaning": data["cleaning"], "duckdb_version": duckdb.__version__,
                   "sql": EDGE_SQL.strip(), "runtime_spatial_policy": "Spatial installed explicitly during data preparation; SQL runtime only LOADs its local extension.",
                   "limitations": ["OpenStreetMap completeness and accessibility tags are not independently surveyed.",
                       "All disconnected components are preserved; endpoints in different components are unreachable.",
                       "Walking graph is bidirectional by OSMnx policy; vehicle one-way restrictions do not apply.",
                       "Physical-polyline total differs from directed length because the latter includes both travel directions.",
                       "Street basemap intentionally omits buildings, terrain and wind; these are not part of the diffusion model."]}
    (reports / "data_quality.json").write_text(json.dumps(report_data, indent=2) + "\n")
    rows = [f"| {key} | {value} |" for key, value in quality.items() if key not in ("component_sizes", "bidirectional_policy")]
    markdown = "\n".join([
        "# OSM data quality report", "", f"Status: **{report_data['status']}**. Generated {report_data['generated_utc']}.", "",
        f"Mission, San Francisco; 4,000 × 4,000 m; {data['crs']}. OSM base timestamp: {', '.join(source['osm_timestamps'])}.",
        "", "| Metric | Value |", "| --- | --- |", *rows, "",
        f"Largest connected component: {quality['component_sizes'][0]:,} nodes ({quality['component_sizes'][0]/quality['nodes']:.2%}). All components are retained.",
        "", "## Geometry and provenance", "",
        "Every directed edge keeps its key and complete projected polyline, oriented from its source node to its destination. Length is recomputed from that polyline in meters. Edges leaving the exact analysis rectangle are removed so interpolation stays inside the model domain.",
        "", "Original Overpass JSON and the downloaded OSMnx GraphML are committed as compressed snapshots. `data/demo/source.json` records query geometry, timestamps, versions, licenses and SHA-256 checksums. `data/manifest.json` hashes every generated dataset.",
        "", "## SQL audit", "", "The nodes and edges are GeoParquet. `data/demo/network.duckdb` contains the same two materialized tables; DuckDB Spatial executed the following audit:", "", "```sql", EDGE_SQL.strip(), "```", "", "## Interpretation limits", "", *[f"- {s}" for s in report_data["limitations"]], "",
        "© [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/).", ""])
    (reports / "data_quality.md").write_text(markdown)
    manifest = {"schema_version": 1, "data_version": data["source_snapshot_sha256"][:16], "region_id": data["region_id"],
                "generated_utc": report_data["generated_utc"], "source": source, "crs": data["crs"], "bounds": data["bounds"],
                "artifacts": {str(p.relative_to(ROOT)): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "size_bytes": p.stat().st_size}
                              for p in sorted(demo.iterdir()) if p.is_file()},
                "rebuild_commands": ["python scripts/prepare_region.py", "python scripts/fetch_osm.py", "python scripts/prepare_network.py", "python scripts/prepare_basemap.py", "python scripts/report_data_quality.py --install-spatial"],
                "attribution": "© OpenStreetMap contributors, ODbL-1.0"}
    (ROOT / "data/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": report_data["status"], **quality}, indent=2))
    if not passed:
        raise RuntimeError("Dataset failed quality checks")
    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-spatial", action="store_true", help="Download DuckDB Spatial once while online")
    report(parser.parse_args().install_spatial)
