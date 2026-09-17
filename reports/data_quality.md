# OSM data quality report

Status: **passed**. Generated 2026-09-17T18:02:59.564960+00:00.

Mission, San Francisco; 4,000 × 4,000 m; EPSG:32610. OSM base timestamp: 2026-09-17T17:56:05Z.

| Metric | Value |
| --- | --- |
| directed_edges | 28498 |
| total_directed_length_m | 1169880.6036134665 |
| geometry_directed_length_m | 1169880.6036134665 |
| shortest_edge_m | 0.4799049059187627 |
| longest_edge_m | 529.9636860458895 |
| invalid_geometries | 0 |
| empty_geometries | 0 |
| nonpositive_lengths | 0 |
| max_length_disagreement_m | 5.684341886080802e-14 |
| weak_components | 44 |
| nodes | 9611 |
| orphan_edge_endpoints | 0 |
| duplicate_edge_ids | 0 |
| parallel_edge_pairs | 279 |
| edges_outside_domain | 0 |
| self_loops | 54 |
| unique_physical_polylines | 14247 |
| strong_components | 44 |

Largest connected component: 9,498 nodes (98.82%). All components are retained.

## Geometry and provenance

Every directed edge keeps its key and complete projected polyline, oriented from its source node to its destination. Length is recomputed from that polyline in meters. Edges leaving the exact analysis rectangle are removed so interpolation stays inside the model domain.

Original Overpass JSON and the downloaded OSMnx GraphML are committed as compressed snapshots. `data/demo/source.json` records query geometry, timestamps, versions, licenses and SHA-256 checksums. `data/manifest.json` hashes every generated dataset.

## SQL audit

The nodes and edges are GeoParquet. `data/demo/network.duckdb` contains the same two materialized tables; DuckDB Spatial executed the following audit:

```sql
SELECT count(*) AS directed_edges,
    sum(length_m) AS total_directed_length_m,
    sum(ST_Length(geometry)) AS geometry_directed_length_m,
    min(length_m) AS shortest_edge_m,
    max(length_m) AS longest_edge_m,
    count(*) FILTER (WHERE NOT ST_IsValid(geometry)) AS invalid_geometries,
    count(*) FILTER (WHERE ST_IsEmpty(geometry)) AS empty_geometries,
    count(*) FILTER (WHERE length_m <= 0) AS nonpositive_lengths,
    max(abs(length_m-ST_Length(geometry))) AS max_length_disagreement_m,
    count(DISTINCT component) AS weak_components FROM edges
```

## Interpretation limits

- OpenStreetMap completeness and accessibility tags are not independently surveyed.
- All disconnected components are preserved; endpoints in different components are unreachable.
- Walking graph is bidirectional by OSMnx policy; vehicle one-way restrictions do not apply.
- Physical-polyline total differs from directed length because the latter includes both travel directions.
- Street basemap intentionally omits buildings, terrain and wind; these are not part of the diffusion model.

© [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/).
