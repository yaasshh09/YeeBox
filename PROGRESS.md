# Progress

State of the world. Read first every session, append before ending. Newest
entry at the top of the session log.

## Current state

**Active phase:** Phase 2 (Phases 0 and 1 done, CI not yet run on GitHub)
**Blocked on:** nothing
**Next action:** push so CI runs once, then plan Phase 2 (CPML) in detail

## Decisions log

Never silently reverse one of these. Add a new row instead.

| Date | Decision | Reason |
|---|---|---|
| init | Rust workspace; `wgpu` + WGSL for the GPU kernel | one kernel runs natively in tests and in the browser |
| init | Separate f64 CPU reference kernel as the oracle | f32 GPU cannot support tight physics tolerances; independent implementation catches indexing bugs |
| init | Storage buffers, in-place leapfrog, no ping-pong | Yee H and E updates never read what they write |
| init | Normalized units inside the solver | conditioning and clarity; physical units only at the UI boundary |
| init | Default Courant number 0.5 | margin for dispersive updates below the 0.7071 limit |
| init | CPML with CFS (kappa, alpha) from the start | first-order boundaries reflect visibly; this is the project's main differentiator |
| init | TF/SF axis-aligned only in v1 | exact dispersion match with the 1D auxiliary grid; oblique needs matching |
| init | TMz through Phase 5, TEz in Phase 6 | plasmonics requires TEz; everything earlier does not |
| init | No WebGL2 fallback; CPU WASM fallback at 256 x 256 instead | reuses the reference kernel, avoids a third implementation |
| init | 2D antenna validation by phased line-source array, not patch antenna | a 2D grid cannot represent a finite antenna; patch and dipole move to the 3D stretch phase |
| init | Presets are Scene JSON files | shareable, testable in the equivalence tier, migration-safe |
| 2026-11-13 | Grid positions and time levels stored doubled as i64 | half-cell staggering checks compare integers exactly |
| 2026-11-13 | Curl stencils kept as data in `grid.rs`; solver loops tested against them bit for bit | proves the hand loops match the documented staggering without trusting a reader |
| 2026-11-13 | Sources applied after the E update to step n + 1, at t = (n + 1) S | one fixed convention, documented on `Sim1d::step` |
| 2026-11-13 | Gaussian pulses use t0 >= 6 tau | a visible turn-on step leaves a (-1)^(i+n) checkerboard that Mur cannot absorb |
| 2026-11-13 | No placeholder gpu or wasm crates until their phases | Phase 0 allows them but nothing needs them yet |
| 2026-11-13 | Repo line endings are LF (`.gitattributes`) | matches the Linux CI runner, quiet on Windows |
| 2026-11-13 | CPU TMz fields are padded to `nx * ny` like the GPU buffers (Hx row ny-1, Hy column nx-1) | one layout for both kernels, and the padding invariant applies to the CPU too |
| 2026-11-13 | Outer ring of Ez nodes is the PEC wall; the E loop covers interior nodes only; PEC material is Ca = Cb = 0 | no special-case boundary code |
| 2026-11-13 | Energy computes H^(n+1/2) from the current state instead of storing the previous H | same expression as the solver, no extra buffer; replaces the earlier "keep a copy" plan |
| 2026-11-13 | Materials are a tagged enum (`dielectric {eps_inf, sigma}`, `pec`), no mu_r field | mu_r is 1 in v1; Drude and Lorentz become new variants |
| 2026-11-13 | Scene v1 holds only what Phase 1 uses, `deny_unknown_fields`, every enum tagged | later fields arrive with `#[serde(default)]`, so v1 files stay valid |
| 2026-11-13 | Geometry is in cell units, Ez node (i, j) sits at (i, j); rect and circle edges are inside | simple, symmetric, and testable on exact lattice points |
| 2026-11-13 | `Kernel` trait is minimal (from_scene, step, time_step, read_field, probe_history) and reuses `grid::Component` as the field id | paint and source edits wait for the UI phase |
| 2026-11-13 | `[profile.test] opt-level = 1` | the 2D acceptance tests took 22 s in debug, now about 1 s; the reference code is untouched |

## Measured numbers

Fill in as they become available. These go in the README.

| Quantity | Value | Conditions | Date |
|---|---|---|---|
| CPML reflection, 10 cells | | normal to 70 degrees | |
| CPML reflection, best tuned | | thickness, parameters | |
| TF/SF leakage | | | |
| GPU vs CPU max difference | | 2000 steps | |
| Throughput, integrated GPU | | grid, steps per frame | |
| Throughput, discrete GPU | | grid, steps per frame | |
| Cold load to first frame | | device | |
| A1 shape error, 1000 cells | 0 (exact) | S = 1, gate 1e-12 | 2026-11-13 |
| A2 phase velocity error | 0 at 10, 20, 40 cpw (below 1 ulp of k) | S = 0.5, L = 200, gate 1e-4 | 2026-11-13 |
| A3 PEC reflection error | 2.1e-15 of peak (S = 1), 0 (S = 0.5 image test) | gate 1e-10 | 2026-11-13 |
| A4 stability | S = 0.70: max 3.96x start over 20000 steps; S = 0.72: past 1e6x at step 44 | 32 x 32 PEC box, random field | 2026-11-13 |
| A5 2D dispersion error | 2.2e-16 diagonal (9.9, 19.8 cpw) and near axis (4.4 deg at 10 cpw, 9.0 deg at 20 cpw) | S = 0.5, guided mode, L = 200, gate 1e-4 | 2026-11-13 |
| A6 cavity resonances vs discrete | 5e-10 to 3e-8 for the first six modes | 40 x 23 cells, S = 0.5, gate 1e-4 | 2026-11-13 |
| A7 cavity convergence order | 2.0014 (Rust test), 2.0015 (independent NumPy measurement) | 4L x 3L cells, L = 4 to 32, gate [1.8, 2.2] | 2026-11-13 |
| Energy drift, lossless cavity | 3.1e-15 relative over 1e4 steps (single half-step H^2 swings 30 percent) | 40 x 30, dielectric block, gate 1e-10 | 2026-11-13 |
| Reciprocity | 2.8e-15 of peak | dielectric, lossy and PEC clutter, gate 1e-9 | 2026-11-13 |

## Known deviations from reference

| Item | Deviation | Reason | Acceptable? |
|---|---|---|---|
| 02-numerics CPML `sigma_opt` | I use `0.8(m+1)/(eta0 dx sqrt(eps_r mu_r))` (Taflove eq. 7.67); the doc's `1/eta` is off by a factor eps_r | the stretch uses eps0, so equal absorption needs 1/sqrt(eps_r) | yes, Phase 2 |
| 02-numerics "H coefficients half a cell deeper" | I evaluate every profile at the component's true coordinate | the doc's rule only holds on the high faces | yes, Phase 2 |
| 04-validation "Gaussian has zero DC" | Zero-DC tests target Ricker and the sine-modulated Gaussian; the plain Gaussian test checks DC = sqrt(pi) tau | a plain Gaussian has nonzero integral | yes |
| A4 at S = 0.72 | Runs through a test-only `from_scene_unchecked` inside fdtd-core | the public constructor must reject S above the limit | yes, done |
| Energy W | No stored copy of H after all; H^(n+1/2) is recomputed from the current state | simpler than a copy and bit-identical to the solver's own update | yes, done |
| A5 "along the axis" | Measured on the lowest guided mode of a wide PEC guide, 4.4 deg (10 cpw) and 9.0 deg (20 cpw) off axis, not at exactly 0 | a closed PEC domain cannot hold a ky = 0 wave; exact axis behaviour is already covered by A2 and `axis_2d_equals_1d` | yes |
| A6, A7 records | Probe records start after the Ricker pulse ends (t > 2 t0) | while the source drives the cavity, the spectrum carries a smooth background that biased the peaks by about 5e-6 and pulled the A7 order to 1.90 | yes |
| `courant_limit(dims)` | Returns `Result` and rejects dims outside 1..3 | no panics reachable from input | yes |

## CI coverage

Which tiers run in CI and which run only locally (GPU tests depend on a
software adapter being available on the runner).

| Tier | CI | Local only | Notes |
|---|---|---|---|
| 1 Unit | yes | | units, grid, dispersion, waveforms, 1D solver, npz |
| 2 Analytic | yes | | A1 to A3 (1D), A4 to A7 (2D TMz) |
| 3 Invariant | yes | | quiescence, linearity (1D and 2D); energy conservation and decay, reciprocity, mirror symmetry, padding (2D) |
| 4 GPU vs CPU | | | |
| 5 External | | | |

## Session log

Format:

    ### YYYY-MM-DD
    **Landed:** what works now
    **Broke:** what failed and what the cause turned out to be
    **Open:** unresolved
    **Next:** the single next action

Write the Broke field carefully, especially when the cause was embarrassing.
It becomes the devlog.

### 2026-11-13 (Phase 1)
**Landed:** Phase 1 is done. `fdtd-core` now has the 2D TMz solver
(`tm.rs`, padded layout, PEC walls, per node Ca and Cb), the material table, the
scene format with JSON, typed validation and rasterization, the discrete
energy, point probes, and the `Kernel` trait. `fdtd run --scene` writes the
final fields and probe records to NPZ, and there's an example scene in
`validation/scenes/cavity_tm.json`. The stencil oracle matches the hand loops
bit for bit, and a deliberate index mutation makes it fail. A4 to A7 all pass
with plenty of room (numbers above). The done plot,
`validation/report/cavity_convergence.png`, comes from a NumPy script that
measures the resonances on its own and lands on order 2.0015. 78 tests pass, and
`fdtd-core` still builds for wasm32.
**Broke:** A7 first came out at order 1.90. That was inside the gate, but the
measured vs discrete error was creeping up from 2.6e-6 to 4.9e-6 as the grid
got finer, which smelled like a measurement problem, not a solver problem. It
was. I was taking the spectrum of the whole probe record, including the stretch
where the Ricker source was still pushing on the cavity. That adds a smooth
background under each peak and nudges it sideways. Once I dropped the samples
before t = 2 t0, the bias fell to about 1e-8 and the order went to 2.0014.
Also, one of my scripted edits ran twice and duplicated the serde lines in
two manifests. I caught it in the diff before committing.
**Open:** CI still hasn't run on GitHub. The A5 "axis" case sits a few degrees
off axis by construction (see known deviations).
**Next:** push, watch CI go green, then plan Phase 2 (CPML) in detail.

### 2026-11-13
**Landed:** Phase 0 is done. I have a Cargo workspace with `fdtd-core` and
`fdtd-cli`, plus CI config, LF line endings, and the pinned Python env.
`fdtd-core` has units, Yee grid helpers with the stencil table, the
numerical dispersion solver (1D closed form, 2D safeguarded Newton), four
waveforms, and the 1D solver with PEC and Mur ends and soft and hard sources.
43 tests pass. A1 is exact, A2 matches the discrete wavenumber to the last bit
at 10, 20, and 40 cells per wavelength, and A3 is within 2e-15. `fdtd run`
writes NPZ, and `plot_1d.py` shows the pulse coming back inverted off the wall.
fdtd-core still builds for wasm32.
**Broke:** Two things, both in my tests, not the solver. First, my DC
integrals started at t = 0 and cut off a Gaussian tail of exp(-(t0/tau)^2),
about 1e-11, so I widened the window to +-10 tau. Second, `mur_exact_at_s1`
left 1.2e-11 behind. I traced it to the source turn-on. With t0 = 5 tau the
first sample is already 7e-11, and a soft E impulse at S = 1 leaves a
(-1)^(i+n) box that satisfies the Mur condition exactly, so it never leaves.
Moving to t0 >= 6 tau fixed it, and that rule is now in the Gaussian docs.
**Open:** CI has never run on GitHub (I haven't pushed yet). The AtomSIM
layout reference in 06-experience is still unknown to me, and I'll ask in
Phase 5.
**Next:** push, watch CI go green, then write the detailed Phase 1 plan.
