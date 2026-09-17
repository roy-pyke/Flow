# Numerical methods and experimental protocol

Flow V2 solves idealized constant-coefficient diffusion and advection–diffusion on a uniform, cell-centered rectangular grid. Concentration is dimensionless, length is in meters, time in seconds, diffusivity κ in m²/s, and prescribed constant velocity u in m/s. All computation uses `float64`; arrays are ordered `(frame, y, x)` with increasing y northward. No clipping or concentration renormalization repairs the solution.

## Conservative spatial operators

For each interior face, one diffusive flux is applied with equal and opposite contributions to its neighboring cells. The resulting sparse operator L includes κ. Reflecting exterior faces contribute zero. On this uniform grid, L has zero row and column sums, is symmetric and negative semidefinite, and agrees with the vectorized face update. Boundary cells have only their existing neighbors. Periodic boundaries add wraparound faces.

The discrete mean and variance energy are `mean(c)` and `E = dx·dy·Σ(c−mean(c))²`. The mass is `M = dx·dy·Σc`. Closed diffusion should preserve M and decrease E within rounding and linear-solve error. Relative balance residuals use `max(|M0|, dx·dy·Σ|c0|, 1)` as their scale; this avoids division by zero for signed or zero-mass input. Extrema and energy diagnostics sample the initial state and requested output times; they do not claim to inspect every internal step.

## Time integration

The theta method solves `(I−θdtL)c_next = (I+(1−θ)dtL)c`. Forward Euler (FE), Backward Euler (BE), and Crank–Nicolson (CN) use θ = 0, 1, and 1/2. FE and BE are first order in time; CN is second order for smooth solutions. BE/CN use CSC sparse systems and SuperLU through [SciPy splu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html), without forming an inverse. Factors are cached by domain, grid, κ, boundary, method, and the exact actual step size, including shortened steps at output times. The cache retains at most eight entries and 256 MiB of counted CSC and LU arrays; allocator and temporary workspace overhead are additional.

FE automatically chooses `dt = 0.9 / [2κ(1/dx²+1/dy²)]`. Manually supplied FE steps must obey the unsafetied stability limit. All methods shorten a step to land on an output time. Implicit default steps are selected for a usable demonstration, not an accuracy guarantee. A normalized algebraic residual is measured at each implicit solve, `||Ax−b||₂ / max(||b||₂,1)`.

The mode amplification factors are `R_FE(z)=1+z`, `R_BE(z)=1/(1−z)`, and `R_CN(z)=(1+z/2)/(1−z/2)` for `z=dt·λ≤0`. CN can be linearly stable while alternating stiff-mode signs and producing negative concentrations. The spike experiment deliberately shows this effect. Optional Rannacher startup replaces the first actual CN macro step with two BE half steps; it damps initial stiff modes but is not a universal positivity guarantee. Negative fields remain visible and are rejected for route-cost construction.

## Transport and boundary fluxes

The model is `c_t + div(u c) = κ Δc`. Upwind face concentrations define a conservative advection operator A. The explicit reference advances `c_next=c+dt(Ac+Lc)`. IMEX Euler solves `(I−dtL)c_next=c+dtAc`; it is first order in time. Its implicit diffusion does not remove the explicit advection restriction:

```text
Explicit transport: dt [|ux|/dx + |uy|/dy + 2κ(1/dx²+1/dy²)] ≤ 0.9
IMEX Euler:         dt [|ux|/dx + |uy|/dy] ≤ 0.9
```

Zero velocity reduces IMEX to BE; κ = 0 reduces it to upwind advection. Grid Peclet numbers `|ui|·di/κ` describe relative advection and physical diffusion, while upwind adds its own numerical diffusion. Pure advection is recorded separately when κ = 0. [Clawpack's finite-volume examples](https://www.clawpack.org/gallery/gallery/gallery_fvmbook.html) provide related inflow and advection–diffusion problems.

Periodic boundaries are a verification problem, not a model of urban recirculation. Open-boundary inflow prescribes **total** outward flux `(u·n)c_in = 0`; no extra entrance diffusion condition is added. Outflow uses the interior upwind concentration and zero exterior diffusive flux. Faces with no normal wind have zero total exterior flux. Open mass is allowed to decrease. The conserved accounting identity is `M(t)−M0+Σ(actual_dt·outward_flux_at_explicit_time)=0`; the code accumulates the actual face flux used in each update. The scenario table covers both velocity signs, diagonal winds, rectangular cells, κ = 0 and κ > 0.

## Accuracy studies

Errors are `L2 = sqrt(cell_area/domain_area · Σ(error²))` (RMS on uniform grids) and `Linf = max|error|`. Time convergence fixes a rectangular grid and compares production solves to the exact discrete Neumann cosine eigenmode. Four successively halved steps isolate time error. Spatial verification first compares exact semi-discrete and continuous cosine solutions on 32² through 256² grids, then checks production FE/BE/CN refinements with measured time error at most 3% of spatial error. The two spatial studies use different final times, explicitly labeled in raw data and plots.

Transport time refinement uses [SciPy expm_multiply](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm_multiply.html) on an independently assembled sparse periodic upwind-plus-diffusion operator. Continuous periodic translated Fourier modes for κ = 0 and translated, decaying modes for κ = 0.03 verify spatial upwind accuracy. These are numerical/analytic references, not field observations. Predeclared acceptance windows are 0.8–1.2 for first order and 1.8–2.2 for second order; all measured refinement intervals and both norms are retained. Structure and API tests complement these experiments.

## Native implementation and performance

The project-owned C++17 kernel implements double-precision zero-flux FE, including single-step and complete time-loop entry points, independent double buffers, strict pybind11 bindings, GIL release during computation, and exact output-time alignment. It is single-threaded with no fast-math. Python normalizes input layout explicitly and includes conversion in full-call timing. A source/build-recipe hash detects an installed binary built from older sources. Explicit `cpp` requests fail if unavailable; `auto` records whether NumPy or C++ actually ran.

`benchmark.json` and `.csv` retain all repeated samples, machine and compiler details, original vectorized NumPy baseline, identical initial fields and output schedules, allocation/conversion/kernel/full-call timing, independent-process peak RSS, and output array sizes. Missing or budget-exceeded cases are labeled. RSS includes interpreter and dependencies; output bytes are not described as pure kernel workspace. Comparisons concern this implementation and machine, not a promised universal speedup.

The benchmark's separately named `gaussian_image_L2_error` is the **unnormalized** area-weighted norm `sqrt(cell_area·Σ(error²))` against its image-Gaussian reference, so its units are concentration × meters. Divide by the square root of domain area to compare with the RMS convention in the convergence and work–precision studies. NumPy/C++ parity uses direct pointwise tolerances, not this analytic-error diagnostic.

Work–precision studies compare methods on identical problems and the same numerical error target, separating freshly cleared LU caches from reused factors. Their raw samples include matrix, factorization, advancement and full-call timing. Route studies keep the graph and road sampling fixed, evaluate every candidate-selected route again in one finer numerical reference field, and check graph optimum costs against NetworkX. A further field refinement estimates reference uncertainty. Path changes with negligible reference regret can reflect nearly tied choices.

In timing fields, `matrix` measures construction of the spatial operator L. `factorization` includes constructing the system `I−θdtL` and its SuperLU factorization. The residual check is included in advancement; the total additionally includes input/output and diagnostic work.

## Reproduce

```bash
./setup.sh
./setup-native.sh
.venv/bin/python -m pytest -q
.venv/bin/python -m scripts.benchmark_numerics
.venv/bin/python -m scripts.run_numerical_experiments
```

The complete report requires a successfully built native backend and saved benchmark data. `--quick` on the numerical-experiment command uses smaller studies and marks that fact in the report. Run benchmarks without competing workloads. `summary.json` records source hashes and environment; the Git identifier is the pre-report commit, so hashes identify the working source used for the report. Generated local experiment history and development records are excluded from Git.
