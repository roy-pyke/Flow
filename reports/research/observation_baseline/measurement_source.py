"""Matched V2 cached-sampling versus CSR observation and quadrature study.

Both paths apply the same bilinear reconstruction and trapezoid rule. Fields,
road parsing and setup are outside repeated apply timing and reported separately.
This measures a spatial observation workload, not PDE or whole-application speed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backend.app.diffusion import Grid, bilinear_interpolate
from backend.app.observations import build_edge_observer
from backend.app.research.experiments import provenance
from backend.app.routing import RoadNetwork

ROOT = Path(__file__).resolve().parents[1]


def timed(call, repeats):
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        call()
        samples.append((perf_counter()-start)*1000)
    return samples


def v2_zero_flux_interpolate(field: np.ndarray, grid: Grid, points: np.ndarray) -> np.ndarray:
    """Frozen V2 direct expression; avoid slowing the baseline with new stencils.

    This retains V2's validation and 1e-7 coordinate tolerance. Benchmark points
    also pass the new, stricter domain validator before comparisons or timings.
    Only zero-flux wall reconstruction is represented by this historical helper.
    """
    field = np.asarray(field, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if field.shape != (grid.ny, grid.nx) or not np.isfinite(field).all():
        raise ValueError("Field must be finite and have grid shape.")
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Points must be a finite array of shape (n,2).")
    xmin, ymin, xmax, ymax = grid.bounds
    tolerance = 1e-7
    if np.any(points[:, 0] < xmin - tolerance) or np.any(points[:, 0] > xmax + tolerance) or np.any(points[:, 1] < ymin - tolerance) or np.any(points[:, 1] > ymax + tolerance):
        raise ValueError("Sampling points lie outside the simulation region.")
    fx = np.clip((points[:, 0] - xmin) / grid.dx - 0.5, 0, grid.nx - 1)
    fy = np.clip((points[:, 1] - ymin) / grid.dy - 0.5, 0, grid.ny - 1)
    ix = np.minimum(np.floor(fx).astype(int), grid.nx - 2)
    iy = np.minimum(np.floor(fy).astype(int), grid.ny - 2)
    tx, ty = fx - ix, fy - iy
    return ((1 - tx) * (1 - ty) * field[iy, ix] + tx * (1 - ty) * field[iy, ix + 1]
            + (1 - tx) * ty * field[iy + 1, ix] + tx * ty * field[iy + 1, ix + 1])


def integral_one_plus_xy(polyline: np.ndarray) -> float:
    """Analytic arc-length integral of 1+x*y, independent of grid/reconstruction."""
    points = np.asarray(polyline, dtype=np.float64)
    starts, deltas = points[:-1], np.diff(points, axis=0)
    mean_xy = (starts[:, 0] * starts[:, 1]
               + (starts[:, 0] * deltas[:, 1] + starts[:, 1] * deltas[:, 0]) / 2
               + deltas[:, 0] * deltas[:, 1] / 3)
    return float(np.sum(np.linalg.norm(deltas, axis=1) * (1 + mean_xy)))


def benchmark(network_path: Path, output: Path, *, n: int = 160, batches=(1, 8, 32), repeats=7) -> dict:
    if output.exists():
        raise FileExistsError(f"Output exists: {output}")
    if not 8 <= n <= 640 or not batches or any(not 1 <= b <= 128 for b in batches) or not 3 <= repeats <= 50:
        raise ValueError("Use grid 8..640, batches 1..128 and repeats 3..50.")
    if max(batches)*n*n*8 > 128*1024**2:
        raise ValueError("Input fields exceed benchmark 128 MiB budget.")
    begin = perf_counter()
    network = RoadNetwork.from_json(network_path)
    network_load_ms = (perf_counter()-begin)*1000
    grid = Grid(network.bounds, n, n)
    speed = 1.4
    begin = perf_counter()
    points, weights, owners = network._sampling(grid.h/2)
    sampling_ms = (perf_counter()-begin)*1000
    sampling_retained = grid.h/2 in network._sampling_cache
    begin = perf_counter()
    observer = build_edge_observer(network, grid, speed_mps=speed)
    observer_build_wall_ms = (perf_counter()-begin)*1000
    # If the sample set exceeds the retention cap, H builds its own samples;
    # that construction is already inside its wall time and must not be added
    # twice to a standalone CSR cold start. Both methods start from a parsed
    # graph and prepared fields; their common load/generation costs are separate.
    csr_cold_setup_ms = observer_build_wall_ms + (sampling_ms if sampling_retained else 0)
    incremental_setup_ms = csr_cold_setup_ms - sampling_ms
    begin = perf_counter()
    xx, yy = np.meshgrid((grid.x-grid.bounds[0])/(grid.bounds[2]-grid.bounds[0]),
                         (grid.y-grid.bounds[1])/(grid.bounds[3]-grid.bounds[1]))
    fields = np.stack([1 + .25*np.cos(2*np.pi*xx+i*.1)*np.sin(2*np.pi*yy-i*.07)
                       for i in range(max(batches))])
    field_generation_ms = (perf_counter()-begin)*1000
    interpolation_difference = 0.0
    for field in fields:
        historical = v2_zero_flux_interpolate(field, grid, points)
        current = bilinear_interpolate(field, grid, points)
        np.testing.assert_allclose(historical, current, rtol=1e-14, atol=1e-14)
        interpolation_difference = max(interpolation_difference,
                                       float(np.max(np.abs(historical-current), initial=0)))

    def reference(batch):
        return np.stack([np.bincount(owners, weights=v2_zero_flux_interpolate(field, grid, points)*weights/speed,
                                    minlength=len(network.edges)) for field in batch])

    cases=[]
    for count in batches:
        batch=fields[:count]
        expected=reference(batch)
        actual=observer.apply_batch(batch)
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-11)
        for _ in range(2):
            reference(batch); observer.apply_batch(batch)
        reference_ms=[]; csr_ms=[]
        # Alternate order to reduce systematic warm-machine/order effects.
        for repeat in range(repeats):
            if repeat%2:
                csr_ms += timed(lambda: observer.apply_batch(batch), 1)
                reference_ms += timed(lambda: reference(batch), 1)
            else:
                reference_ms += timed(lambda: reference(batch), 1)
                csr_ms += timed(lambda: observer.apply_batch(batch), 1)
        base=float(np.median(reference_ms)); optimized=float(np.median(csr_ms))
        cases.append({"batch":count,"reference_ms":reference_ms,"csr_ms":csr_ms,
                      "reference_median_ms":base,"csr_median_ms":optimized,
                      "speed_ratio":base/optimized,"max_absolute_difference":float(np.max(np.abs(actual-expected), initial=0)),
                      "input_bytes":batch.nbytes,"output_bytes":actual.nbytes,
                      "reference_setup_plus_apply_ms":sampling_ms+base,
                      "csr_setup_plus_apply_ms":csr_cold_setup_ms+optimized,
                      "amortization_calls_estimate":max(0.0,incremental_setup_ms)/(base-optimized) if base>optimized else None})
    # An independent small polyline problem separates quadrature error from
    # field-reconstruction error. Analytic bilinear c=1+x*y is exact inside the
    # cell-centre rectangle; the chosen segments stay inside every tested grid.
    small=RoadNetwork({"bounds":[0,0,1,1],"nodes":[{"id":0,"x":.2,"y":.2},{"id":1,"x":.8,"y":.7}],
        "edges":[{"u":0,"v":1,"key":0,"coordinates":[[.2,.2],[.55,.75],[.8,.7]]}]})
    g=Grid((0,0,1,1),16,16); gx,gy=np.meshgrid(g.x,g.y); field=1+gx*gy
    exact=integral_one_plus_xy(small.edges[0]["coordinates"])
    gauss_reference=float(build_edge_observer(small,g,speed_mps=1,quadrature="gauss2_grid").apply(field)[0])
    np.testing.assert_allclose(gauss_reference, exact, rtol=3e-15, atol=3e-15)
    quadrature=[]
    for spacing in [.2,.1,.05,.025,.0125]:
        value=build_edge_observer(small,g,speed_mps=1,sample_spacing_m=spacing).apply(field)[0]
        quadrature.append({"spacing_m":spacing,"integral":float(value),"absolute_error":abs(float(value-exact))})
    reconstruction=[]
    start=np.array([.2,.2]); end=np.array([.8,.7]); length=float(np.linalg.norm(end-start))
    straight=RoadNetwork({"bounds":[0,0,1,1],"nodes":small.nodes.values(),"edges":[{"u":0,"v":1,"key":0,"coordinates":[start.tolist(),end.tolist()]}]})
    exact_quadratic=length*(1+(start@start+start@end+end@end)/3)
    for size in [8,16,32,64]:
        current=Grid((0,0,1,1),size,size); x,y=np.meshgrid(current.x,current.y)
        value=build_edge_observer(straight,current,speed_mps=1,quadrature="gauss2_grid").apply(1+x*x+y*y)[0]
        reconstruction.append({"n":size,"integral":float(value),"absolute_error":abs(float(value-exact_quadratic))})
    result={"schema_version":1,"protocol":{"network":str(network_path),"network_sha256":hashlib.sha256(network_path.read_bytes()).hexdigest(),
                "grid":grid.to_dict(),"directed_edges":len(network.edges),"repeats":repeats,"warmups":2,
                "output":"all edge exposures for every input field", "speed_mps":speed,
                "timed_scope":"matched repeated observations, including input validation and output allocation; excludes PDE, graph load, H/sampling construction and HTTP",
                "baseline":"frozen V2 direct zero-flux bilinear interpolation and NumPy bincount, cached points, Python loop over fields",
                "sign_policy":"known nonnegative generated fields; spatial operators are compared without routing sign checks",
                "setup_note":"cold observation costs start from a parsed graph and prepared fields; sum separately measured setup and median apply, not an independently repeated cold-start benchmark",
                "observer_sampling_note":"reused previously constructed samples" if sampling_retained else "sample set exceeded cache cap; H rebuilt samples inside its measured construction time",
                "amortization_note":"incremental cold setup divided by median per-call savings, clamped to zero; estimate from one setup measurement, not a guaranteed crossover",
                "thread_policy":"single Python process, no explicit concurrent workers; CSR uses the installed SciPy implementation"},
            "network_load_ms":network_load_ms,"field_generation_ms":field_generation_ms,
            "generated_batch_size":max(batches),"sampling_setup_ms":sampling_ms,
            "sampling_retained_for_H_build":sampling_retained,
            "sampling_array_bytes":sum(a.nbytes for a in (points,weights,owners)),
            "sampling_cache_limit_bytes":network._sampling_cache_max_bytes,
            "observer_build_wall_ms":observer_build_wall_ms,"csr_cold_setup_ms":csr_cold_setup_ms,
            "incremental_cold_setup_vs_sampling_ms":incremental_setup_ms,
            "legacy_vs_current_interpolation_max_absolute_difference":interpolation_difference,
            "observer":{k:v for k,v in observer.metadata.items() if k!="edge_ids"},
            "cases":cases,"quadrature_convergence":quadrature,"reconstruction_convergence":reconstruction,
            "quadrature_reference":{"field":"1+x*y","grid_n":16,"speed_mps":1,
                "polyline":small.edges[0]["coordinates"],"analytic_integral":exact,
                "gauss2_grid_integral":gauss_reference,"gauss_absolute_error":abs(gauss_reference-exact)},
            "reconstruction_reference":{"field":"1+x*x+y*y","speed_mps":1,
                "polyline":straight.edges[0]["coordinates"],"analytic_integral":float(exact_quadratic),
                "projection":"cell-centre point samples","quadrature":"gauss2_grid"},
            "provenance":provenance(),"benchmark_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limits":"Local spatial observation benchmark only. No PDE, dynamic routing or whole-application speedup claimed. Reconstruction study uses point samples, not FV cell averages."}
    output.mkdir(parents=True)
    (output/'benchmark.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    axes[0].plot([r['batch'] for r in cases],[r['reference_median_ms'] for r in cases],'o-',label='V2 sample + aggregate')
    axes[0].plot([r['batch'] for r in cases],[r['csr_median_ms'] for r in cases],'o-',label='CSR H batch')
    axes[0].set(xlabel='Fields per batch',ylabel='Median apply time (ms)',title=f'{len(network.edges):,} directed edges');axes[0].legend()
    axes[1].loglog([r['spacing_m'] for r in quadrature],[r['absolute_error'] for r in quadrature],'o-');axes[1].set(xlabel='Quadrature spacing (m)',ylabel='Absolute integral error',title='Fixed bilinear reconstruction')
    axes[2].loglog([1/r['n'] for r in reconstruction],[r['absolute_error'] for r in reconstruction],'o-');axes[2].set(xlabel='Grid spacing (m)',ylabel='Absolute integral error',title='Exact reconstruction integration')
    fig.tight_layout();fig.savefig(output/'benchmark.png',dpi=150);plt.close(fig)
    lines=['# Spatial observation performance and convergence','',result['limits'],'',
           '| Fields | V2 sample median ms | CSR median ms | Ratio | Max difference |','|---:|---:|---:|---:|---:|']
    lines += [f"| {r['batch']} | {r['reference_median_ms']:.4f} | {r['csr_median_ms']:.4f} | {r['speed_ratio']:.2f} | {r['max_absolute_difference']:.3e} |" for r in cases]
    lines += ['',f"Common graph load: {network_load_ms:.3f} ms. Generation of {max(batches)} fields: {field_generation_ms:.3f} ms.",
              f"Sampling setup: {sampling_ms:.3f} ms. Measured H build: {observer_build_wall_ms:.3f} ms. CSR cold setup: {csr_cold_setup_ms:.3f} ms; CSR storage: {observer.metadata['csr_bytes']} bytes.",
              result['protocol']['observer_sampling_note']+'. '+result['protocol']['setup_note']+'.',
              'Setup is not included in apply ratios. Raw repetitions, setup-inclusive times and amortization estimates are in benchmark.json.',
              f"Frozen V2 and current interpolation agree to {interpolation_difference:.3e}; Gauss reference agrees with the independent analytic polyline integral to {abs(gauss_reference-exact):.3e}.",
              '', '![Measured observations and separated errors](benchmark.png)','']
    (output/'README.md').write_text('\n'.join(lines))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network',type=Path,default=ROOT/'data/demo/network.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--grid',type=int,default=160)
    parser.add_argument('--repeats',type=int,default=7)
    args=parser.parse_args()
    result=benchmark(args.network,args.output,n=args.grid,repeats=args.repeats)
    print(json.dumps({"output":str(args.output),"cases":result['cases']},indent=2))


if __name__=='__main__':main()
