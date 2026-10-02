# Positive variable diffusion: model, operator and evidence

The research interface solves

\[
\partial_t c=\nabla\cdot(\kappa(x,y)\nabla c)
\]

on a uniform Cartesian rectangle. The concentration is relative, coordinates
are metres, time is seconds, and diffusivity is in m²/s. A supplied array gives
one static, isotropic, strictly positive material value per cell, in `(ny,nx)`
order with rows increasing south to north. This is a material representation;
it is not a concentration cell average or a tensor coefficient. The initial
concentration can still use actual cell averages.

The supported variable-coefficient methods are NumPy Forward Euler (FE) and
SciPy Backward Euler (BE) / Crank–Nicolson (CN), with `zero_flux` or `periodic`
boundaries. CN can use Rannacher startup. `auto` chooses NumPy for variable FE
and SciPy for variable BE/CN, and diagnostics record this capability choice.
An explicit C++ request is rejected. Constant scalar diffusivity retains the
existing implementations, including scalar zero diffusion and the native FE
kernel. Zero, negative, complex, boolean or nonfinite coefficient arrays are
rejected. Variable wind, source terms, anisotropy, subcell interfaces and
variable-coefficient open boundaries are outside this milestone.

## From two half-cell resistances to a face flux

For a face between cells P and N, suppose the material is constant inside each
half-cell and the interface lies on the face. With normal cell-centre distance
h, concentration drop and flux satisfy

\[
q=-\frac{c_N-c_P}{h/(2\kappa_P)+h/(2\kappa_N)}
  =-\kappa_f\frac{c_N-c_P}{h},\qquad
\kappa_f=\frac{2\kappa_P\kappa_N}{\kappa_P+\kappa_N}.
\]

This derivation assumes an orthogonal grid, isotropic materials, continuity of
flux and no interfacial resistance or source. It is exact for the aligned
piecewise-constant one-dimensional resistance problem. For smooth material
fields it is a consistent two-point approximation. It is not a general formula
for a misaligned interface, discontinuous concentration, nonorthogonal mesh or
anisotropic tensor. The general cell-centred flux balance and orthogonality
condition are also described in the [NIST FiPy finite-volume documentation](https://pages.nist.gov/fipy/en/stable/numerical/discret.html).

Production code computes `min * (2 / (1 + min/max))`, avoiding the overflow-prone
product and sum in the displayed formula. Division by spacing occurs twice to
avoid explicitly squaring extreme spacings. An unrepresentable face rate or
outgoing sum is rejected; it is not replaced with zero or infinity.

Each face has a rate `a = kappa_face / h²`, and contributes
`+a*(c_N-c_P)` to P and its negative to N. Zero-flux boundaries add no exterior
face. Periodic boundaries add a seam face in each axis. A periodic axis with
two cells has **two physical faces** connecting the same pair; their
contributions must both be retained.

## Operator properties and time integration

Let L be the assembled diffusion matrix and let the sum below run over physical
faces once. On this equal-volume grid,

\[
L\mathbf1=0,\quad\mathbf1^TL=0,\quad L=L^T,\quad
z^TLz=-\sum_f a_f(z_N-z_P)^2\le0.
\]

L is symmetric **negative semidefinite**, with a constant nullspace on the
connected grid. The cell-volume inner product is a positive scalar multiple
of the Euclidean one here. For positive dt and theta, `I-theta*dt*L` is symmetric
positive definite. This property applies to the closed/periodic diffusion
operator; the existing nonzero-wind upwind transport operator is generally
nonsymmetric and has no such SPD claim.

If `s_P` is the sum of rates leaving cell P, an FE step is a convex combination
when `dt <= 1/max(s_P)`. The variable-field implementation uses this actual
row-rate bound and automatic FE uses 90% of it. The scalar implementation keeps
its historical bound. Supported FE and BE preserve positivity and the maximum
principle up to numerical roundoff. CN dissipates the quadratic energy for this
operator but does **not** preserve positivity for arbitrary dt. Rannacher
startup damps the first CN step with two BE half steps; it is not a general
positivity guarantee for later large CN steps. No clipping or mass rescaling is
performed.

Mass, mean-free quadratic energy, extrema and linear residual diagnostics are
retained. Mass and energy histories are sampled at saved output times and the
initial state; linear residuals are checked at each implicit solve. These
closed, source-free identities must not be reused as complete balances for a
future source/open-boundary equation.

## Prepared fields, memory and cache identity

`prepare_diffusivity(grid, values, boundary)` creates a `DiffusionCoefficients`
snapshot. It owns immutable bytes-backed cell, face and outgoing-rate arrays;
changing the original input cannot change a prepared field. Its content hash
includes canonical float64 material bytes, shape, units, interpretation and the
face policy. Reuse requires the exact grid and boundary. A scalar input returns
the validated scalar and preserves its legacy path.

The solver prepares a variable field once per solve. Both CSC assembly and
matrix-free application use those same face rates. Matrix-free application
holds the output plus one directional flux temporary at a time. Metadata
reports retained coefficient-array bytes and an arithmetic temporary-array
estimate, with validation/conversion/allocator overhead excluded. It does not
claim a peak process RSS bound. Implicit LU cache identity includes the
coefficient hash, geometry, boundary, method and exact hexadecimal actual step
size, so changing any coefficient or shortened output-alignment step cannot
reuse a different system's factor.

```python
import numpy as np
from backend.app.diffusion import Grid
from backend.app.numerics import prepare_diffusivity, solve

grid = Grid((0, 0, 2, 1), 32, 16)
kappa = np.full((16, 32), 0.02)
kappa[:, :16] = 0.2
material = prepare_diffusivity(grid, kappa, "zero_flux")
initial = 1 + .2*np.cos(np.pi*grid.x)[None, :]*np.ones((16, 1))
frames, diagnostics = solve(grid, initial, material, [0, .1, .5],
    method="backward_euler", backend="scipy", dt=.01)
```

## Configuration and portable inputs

`ExperimentSpec.problem.kappa` accepts its original nonnegative scalar, or a
tagged `layered`/`npz` object. The existing scalar configuration and identity
calculation remain compatible. An aligned layer can be declared as

```json
{"kind": "layered", "axis": "x", "interface_m": 1.0,
 "left": 0.2, "right": 0.02}
```

The physical interface must lie inside the domain on a grid face. The nearest
face is reconstructed in physical coordinates and compared within two
coordinate ULPs, including domains with large projected-coordinate offsets.
Spacing at or below 32 coordinate ULPs is rejected as insufficiently resolved.
No subcell material model or interpolation is inferred. Periodic layers also
meet across the domain seam, which is a second material interface.

Imported NPZ files must contain exactly a real, finite, positive `values` array.
The descriptor declares file SHA-256, path, bounds, nx, ny, `axis_order: "y,x"`,
`y_direction: "south_to_north"`, `interpretation: "cell_material"`, and
`unit: "m2/s"`. Grid and domain must match exactly. The physical identity uses
content/geometry, not the local pathname. An imported grid field is a discrete
physical input; changing its values or declared grid changes that identity.

Run a complete configured example:

```bash
.venv/bin/python -m scripts.run_research run configs/research/layered_diffusion.json --output data/experiments/research/my-layered
.venv/bin/python -m scripts.run_research verify data/experiments/research/my-layered
.venv/bin/python -m scripts.run_research replay data/experiments/research/my-layered --output data/experiments/research/my-layered-replay
```

New variable bundles store the actual coefficient array alongside concentration
arrays. Imported coefficient bytes are additionally preserved as
`kappa_input.npz`. Verification checks schema, array and artifact integrity,
coefficient identity and exact imported values; it does not re-evaluate an
analytical material formula from today's implementation. Replay performs that
new evaluation. Output admission is checked before input allocation, and work
admission uses the resolved material's CFL. These are admission estimates, not
CPU or sparse-factor peak memory limits.

## Independent numerical study

```bash
.venv/bin/python -m scripts.run_variable_diffusion_experiments --config configs/research/variable_diffusion_study.json --output reports/research/my-variable-study
.venv/bin/python -m scripts.run_variable_diffusion_experiments --verify reports/research/my-variable-study
.venv/bin/python -m scripts.run_variable_diffusion_experiments --replot reports/research/my-variable-study
```

The study separates three questions:

1. **Interface flux.** A one-dimensional two-layer steady problem uses prescribed
   endpoint concentrations and their half-cell resistances. The exact flux is
   the concentration drop divided by `length_left/kappa_left +
   length_right/kappa_right`. Every face flux, including endpoints and the
   interface, is compared with this value. An arithmetic-face control exposes
   the error from replacing the series resistance. Dirichlet terms are added
   only inside this independent validation problem; they are not a new
   production boundary option.
2. **Spatial error.** With `theta=pi*x/L`, `r=1+2*cos(2*theta)`, choose
   `kappa=D*(1+b*r/3)/(1+3*b*r)`, `abs(b)<1/9`. Then
   `u=1+A*exp(-D*pi²*t/L²)*(cos(theta)+b*cos(3*theta))` solves the closed,
   source-free equation exactly. Indeed
   `kappa*u_x=-A*exp(-D*pi²*t/L²)*D*pi/L*(sin(theta)+b*sin(3*theta)/3)`;
   another derivative gives `u_t`, and the endpoint fluxes vanish. Initial and
   reference concentrations use exact sinc cell averages. A semi-discrete
   exponential comparison quantifies remaining CN time error separately.
3. **Temporal error.** On a fixed small rectangular grid, an independent
   face-by-face dense assembly and its symmetric eigendecomposition provide
   the reference evolution. FE/BE/CN step refinements are compared with that
   reference, so their order is not confused with continuum spatial error.

The experiment archives configuration, full arrays, numerical records, plots,
report and the source used. Verify, then replay from `source_snapshot/` with
`PYTHONDONTWRITEBYTECODE=1` and the prepared Python interpreter into a **new**
directory. Redraw reads saved data and writes outside the sealed bundle. Hashes
establish integrity, not authorship or a scientific proof. The coefficient
formula above is a deliberately constructed verification problem, not a
measured environmental diffusivity.

The reference study also limits `T*max(outgoing_rate)` before its spatial
matrix exponential (default 10,000), separately from saved-array and solver
cell-update budgets. A stiff reference request can be rejected even when its
implicit solver would need few steps. This admission proxy does not provide a
wall-time or eigensolver memory guarantee. Errors below floating-point
resolution are reported as unresolved; they do not establish a convergence
order.

## Measure operator cost without assuming a winner

```bash
.venv/bin/python -m scripts.benchmark_diffusion_operator --nx 256 --ny 160 --repeats 21 --output reports/research/my-variable-operator-benchmark
```

The benchmark first checks matrix-free/CSC parity, then saves every preparation,
assembly and warm application timing. Both methods use identical input arrays;
applications are interleaved in alternating order. Matrix-free includes its
normal validation, and CSC uses its normal matrix-vector call. Traced allocation
and explicit resident-array byte accounting are reported separately. No PDE,
LU factorization, route search, graph loading, HTTP or UI cost enters these
measurements. A matrix-free representation can reduce storage without being
the fastest warm matvec on a particular NumPy/SciPy build.

## Recorded local results

The [sealed numerical study](../reports/research/variable_diffusion_baseline/REPORT.md)
has 66 raw arrays and 29 manifest-listed artifacts. Source replay reproduced
every array exactly, and redrawing from the saved report produced the same PNG
bytes. This is recorded-environment evidence, not a cross-platform bitwise
guarantee. The full suite passed 459 tests with native execution required;
after a plot-label adjustment, both relevant publication/redraw tests passed.

The spatial RMS orders were 1.9933, 1.9983 and 1.9996. At nx=128 the CN temporal
RMS error was 0.047% of the spatial RMS error, so it did not mask that curve.
Final temporal orders for zero-flux/periodic cases were FE 1.0059/1.0088,
BE 0.9942/0.9914 and CN 2.0002/2.0001. The aligned two-layer case had a
harmonic-face concentration Linf error of 5.55e-16 and maximum face-flux error
1.33e-14. Arithmetic faces produced approximately 2.63% flux error in the
same validation problem.

The [operator measurement](../reports/research/variable_operator_baseline/benchmark.json)
uses one periodic 256×160 grid and 21 repetitions in the recorded environment.

| Quantity | Prepared matrix-free | CSC |
| --- | ---: | ---: |
| Resident coefficient arrays / matrix arrays | 1,310,720 bytes | 2,621,444 bytes |
| Median warm apply | 0.144250 ms | 0.107083 ms |
| Traced peak allocation per warm call | 786,184 bytes | 327,923 bytes |

Preparation took a median 0.620625 ms; CSC assembly from a prepared field added
2.475833 ms. Maximum absolute output difference was 3.64e-12 on an output scale
of approximately 10,577. The resident-storage columns compare the coefficient
representation with the standalone CSC representation, not entire solver RSS;
the current implicit solver retains both and its LU factors. The matrix-free
array-only temporary estimate was 654,080 bytes; tracked allocation was higher,
as expected when including implementation overhead. CSC won warm throughput
here. These timings do not claim a faster complete PDE solve.
