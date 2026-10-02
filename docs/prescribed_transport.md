# Prescribed winds, sources and boundary mass balance

This research interface solves

```text
∂c/∂t + div(u c) = div(κ grad c) + S.
```

`c` is synthetic relative concentration, `u` is in m/s, `κ` in m²/s and
the additive source `S` in relative concentration/s. Space is Cartesian meters
and time starts at zero seconds. Sources are prescribed independently of `c`;
state-dependent chemistry, reactions and data assimilation are outside this
model. The map controls retain their historical frozen-field interface.

## Run a configured case

```bash
.venv/bin/python -m scripts.run_research run configs/research/prescribed_transport.json --output data/experiments/research/my-prescribed
.venv/bin/python -m scripts.run_research verify data/experiments/research/my-prescribed
.venv/bin/python -m scripts.run_research replay data/experiments/research/my-prescribed --output data/experiments/research/my-prescribed-replay
```

The bundled example reverses a uniform wind, releases a spatial Gaussian
triangle pulse and supplies different incoming concentrations on the two x
boundaries. Each output directory must be new. A triangle pulse is continuous
and piecewise linear; it is not an instantaneous release or rectangular jump.

The `problem.forcing` field accepts `kind: analytic` or `kind: npz`. The legacy
`velocity` pair must be zero when forcing owns the wind. Without forcing, old
physical/configuration identities and the original solver path remain intact.

Analytic generator version 1 supports:

- `uniform` wind with a two-component `value`;
- `shear` wind with `axis`, `mean_speed_m_s`, and `gradient_s_inv`;
- `rotation` with `angular_rate_s_inv` and an optional center;
- `constant` source with a `value`, or `gaussian` source with `center`, positive
  `sigma_m` and `amplitude`. Gaussian spatial values are integrated over cells
  using erf;
- nonnegative `left`, `right`, `bottom`, `top` inflow values.

`wind_scale`, `source_scale` and `inflow_scale` give one multiplier per common
`times_s` knot; omitted or null scales are one. `inflow_scale` must be
nonnegative; wind and source multipliers may be signed. Sources may be signed for manufactured
solutions, but negative additive sources do not carry a positivity guarantee.
An x shear is x-directed speed varying linearly across y about the domain
midpoint; y shear swaps the axes. Rotation is the ordinary planar field about
its stated center. For these affine spatial fields, midpoint face values are
exact face averages. An omitted rotation center is the domain midpoint.
Incompatible closed normals or periodic seams are rejected.
A periodic rotation input defines a periodic extension of these normal face
fluxes; tangential components can jump at the identified domain edges. It is
not an unbounded planar rigid-rotation trajectory benchmark.

## Python and array contract

```python
from backend.app.diffusion import Grid
from backend.app.numerics import PrescribedFields, solve
import numpy as np

grid = Grid((0, 0, 2, 1), 24, 12)
times = np.array([0., .25, .5])
forcing = PrescribedFields(
    grid, times,
    velocity_x=np.broadcast_to(np.array([.5, 0., -.5])[:, None, None], (3, 12, 25)),
    source=np.broadcast_to(np.array([0., .1, 0.])[:, None, None], (3, 12, 24)),
    inflow={"left": np.ones((3, 12)), "right": np.full((3, 12), .5)},
    boundary="open",
)
frames, diagnostics = solve(
    grid, np.full((12, 24), .2), .01, times,
    method="imex_euler", backend="numpy_scipy", dt=.01,
    boundary="open", forcing=forcing,
)
```

Stored arrays are owned immutable float64 values. Coordinates increase along
x and y; row zero is the southern row. The arrays are:

| Array | Shape | Meaning |
| --- | --- | --- |
| `times_s` | `(nt,)` | Strictly increasing common knots, beginning at zero |
| `velocity_x` | `(nt, ny, nx+1)` | Face-mean x component, positive toward +x |
| `velocity_y` | `(nt, ny+1, nx)` | Face-mean y component, positive toward +y |
| `source` | `(nt, ny, nx)` | Additive cell-average concentration rate |
| `inflow_left`, `inflow_right` | `(nt, ny)` | Nonnegative incoming boundary face concentrations |
| `inflow_bottom`, `inflow_top` | `(nt, nx)` | Nonnegative incoming boundary face concentrations |

Python `None` arrays and omitted inflow sides become zero arrays. Supplied
arrays must have exact shapes; no implicit scalar broadcast, axis flip or
spatial resampling is performed. All quantities use piecewise-linear time
reconstruction. Extrapolation is forbidden, including beyond the last knot.
The configuration accepts 2–4096 knots, and the last knot must cover the entire
solve interval from zero through the final requested output, even if the first
requested output is later than zero. Stored NPZ times must exactly equal the
configuration's `times_s` values.
An imported NPZ must contain exactly the eight arrays above and declare its
`path`, lowercase hexadecimal `sha256`, `bounds`, `nx`, `ny` and `times_s`.
The following metadata have these defaults and reject other values:

```json
{
  "axis_order": "time,y,x",
  "y_direction": "south_to_north",
  "representation": "face_velocity_cell_average_source_boundary_face_inflow",
  "temporal_interpolation": "linear",
  "temporal_extrapolation": "reject",
  "velocity_unit": "m/s",
  "source_unit": "relative_concentration/s",
  "inflow_unit": "relative_concentration"
}
```

Grid bounds and dimensions must match the experiment exactly. NPZ numeric
integer and floating arrays are converted to float64 and must remain finite;
object, complex and boolean arrays are rejected. Headers and expanded input
budgets are checked before allocating the declared NumPy arrays; pickle is
never enabled. Local input paths are excluded from physical identities, while
the input SHA-256 and declared metadata remain part of them.

## Conservative faces and boundary conditions

For each interior face, the upwind flux is the oriented face velocity times
the upstream cell value. Its divergence contributes equal and opposite mass
changes to the adjacent cells. This implements `-div(u c)`, which equals
`-u·grad(c) - c div(u)`. The second term cannot be discarded for divergent wind.
Compression can increase a concentration maximum and mean-free energy while
the closed/periodic total mass remains conserved.

- Closed boundaries require zero exterior normal velocity and use zero total
  exterior flux; nonzero internal wind is allowed.
- Periodic duplicate boundary faces must have exactly equal velocities. Both
  cells use one shared seam flux; even a two-cell axis retains both physical
  faces.
- Open inflow prescribes the total incoming numerical flux `u_n c_in`.
  Outflow uses `u_n c_interior`. Exterior diffusion contributes no additional
  face term. This flux prescription is not a separate concentration Dirichlet
  condition plus an independently imposed diffusion boundary condition.

All four inflow arrays must be zero for closed or periodic boundaries. On an
open boundary, a prescribed concentration contributes only while that face's
velocity points inward; it does not override an outward face's interior value.

The implementation uses cell-centered finite volumes and integrated face
fluxes; see the [NIST FiPy finite-volume derivation](https://pages.nist.gov/fipy/en/stable/numerical/discret.html)
for the general balance formulation. This implementation is independently
tested and does not use FiPy as a dependency or claim FiPy equivalence.

## Time methods and CFL

| Method | Source quadrature per actual step | Wind / incoming concentration |
| --- | --- | --- |
| FE diffusion | Left endpoint | Zero |
| BE diffusion | Right endpoint | Zero |
| CN diffusion | Trapezoid over the two endpoints | Zero |
| Explicit upwind advection–diffusion | Left endpoint | Left endpoint |
| IMEX Euler | Left endpoint | Left endpoint |

Rannacher startup uses the **right endpoint of each of its two BE half steps**.
All methods land on requested output times and every forcing knot. The ledger
uses exactly the same source weights as the field update. CN is second order
for suitable smooth diffusion/source problems; transport remains first order.
Implicit diffusion does not make time-dependent advection implicit.

For a cell, its advective outgoing rate is

```text
r = max(u_right,0)/dx + max(-u_left,0)/dx
  + max(v_top,0)/dy   + max(-v_bottom,0)/dy.
```

The maximum over all cells and all stored knots bounds every intermediate
time: positive parts and their sums are convex along each linear interval.
Explicit transport uses `dt <= .9 / (max(r) + 2κ(dx^-2+dy^-2))`;
IMEX uses `dt <= .9 / max(r)`. The bound includes future wind values even when
the initial wind is zero. This is a sufficient global bound, not an optimal
adaptive step. Pure diffusion keeps its established diffusion bound.

The configured combinations are:

| `problem.model` | `numerical.method` | Explicit backend / `auto` selection | Boundaries |
| --- | --- | --- | --- |
| `diffusion` | `explicit_euler` | `numpy` | `zero_flux`, `periodic` |
| `diffusion` | `backward_euler`, `crank_nicolson` | `scipy` | `zero_flux`, `periodic` |
| `advection_diffusion` | `advection_explicit` | `numpy` | `zero_flux`, `periodic`, `open` |
| `advection_diffusion` | `imex_euler` | `numpy_scipy` | `zero_flux`, `periodic`, `open` |

Each row also accepts `backend: auto`, selecting the backend shown. Prescribed
pure diffusion permits a nonnegative scalar or positive cell-material κ and
requires zero wind and inflow. Prescribed transport currently requires a
nonnegative scalar κ, including when its resolved wind is zero. Only CN accepts
`startup: rannacher`; all methods accept `startup: none`. Explicit `cpp`
requests are rejected whenever forcing is present. The old native kernel
remains available for its original unforced constant-κ scope.

## What the diagnostics establish

At every internal step the solver checks the discrete residual

```text
R = M(t) - M(0) + accumulated_outgoing - accumulated_incoming - accumulated_source.
```

The three accumulated terms and their histories are saved separately.
`cumulative_outward_flux` retains its legacy meaning of **net** outward mass;
`cumulative_boundary_outward_mass` and `cumulative_boundary_inward_mass` are the
separate oriented contributions. An outgoing contribution can be signed if
the interior solution is signed. Source mass uses numerical quadrature and is
not labeled an exact physical time integral.

Small `R` demonstrates the accounting of the update. It does not demonstrate
small field error or accurate source/boundary integrals: those require an
independent reference. Raw mass drift remains visible and can be large in a
correctly forced calculation. No values are clipped or mass-renormalized.

Energy, extrema and field-level diagnostics are sampled at initial/output
frames; mass balance is additionally checked every internal step. A source,
open inflow or compressive wind removes general energy/maximum nonincrease
claims. Nonnegative data and nonnegative sources retain positivity under the
FE bound or BE/IMEX assumptions; CN has no unconditional positivity guarantee.

## Archives and limitations

Generic experiment bundles include `forcing.npz` with the actual materialized
inputs; imported inputs also retain their original bytes in
`forcing_input.npz`, with the archived configuration pointing to that file.
Each array has
shape, dtype and content identity. Verification validates the stored data and
boundary contract without reevaluating the analytic generator; imported
resolved arrays must also equal the preserved input after float64 conversion.
Bundles without configured forcing reject the reserved forcing files and
result record, and analytic bundles reject an imported-input file. Generic replay
recomputes with the current generator (or the preserved NPZ input) and compares
every forcing/output array, so changes to generated inputs are visible. Thus
generic analytic replay is not a solver run driven directly by its archived
`forcing.npz`; verification can still accept a historical bundle after the
current analytic implementation changes. The
independent study additionally preserves its generator in a source snapshot.
Hashes establish file integrity, not physical validity or authenticity.

Forcing input bytes, output bytes and estimated cell updates have separate
admission limits: `budget.max_forcing_bytes` and `budget.max_output_bytes`
default to 128 MiB each, and `budget.max_cell_updates` defaults to 500 million.
The forcing limit counts all eight resolved float64 arrays, including zero
arrays; output admission counts requested concentration frames. Original NPZ
file size and total expanded NPY member sizes each allow at most
`max_forcing_bytes + 1_000_000` bytes for container/header overhead, while
resolved-array admission still uses the strict forcing limit. Output and
forcing array estimates are checked before loading forcing inputs. Extra
interior knot/output splits and the two-half-step startup reserve enter the
work estimate. Resident
array and snapshot sizes are reported; neither the budget nor these byte
counts are a peak-RSS or wall-time guarantee. Immutable interpolation snapshots
and per-step validation carry costs; no new performance speedup is claimed.

The independent study runs with

```bash
.venv/bin/python -m scripts.run_forced_transport_experiments --output reports/research/my-forced-study
.venv/bin/python -m scripts.run_forced_transport_experiments --verify reports/research/my-forced-study
.venv/bin/python -m scripts.run_forced_transport_experiments --replot reports/research/my-forced-study
```

It separates continuum cell-average, fixed-grid time and exact mass-integral
references. Source snapshots and raw arrays support replay and replot into new
destinations. The [formal study report](../reports/research/forced_transport_baseline/REPORT.md)
records the measured convergence and conservation results.

Higher-order transport, variable-material transport, tensor/time-dependent
diffusion, discontinuous source schedules, nonlinear reactions and calibrated
air-quality prediction remain outside this milestone.
