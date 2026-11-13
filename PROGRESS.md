# Progress

State of the world. Read first every session, append before ending. Newest
entry at the top of the session log.

## Current state

**Active phase:** Phase 3 (Phases 0, 1 and 2 done, CI not yet run on GitHub)
**Blocked on:** nothing
**Next action:** push so CI runs once, then plan Phase 3 (sources, TF/SF, DFT monitors) in detail

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
| 2026-11-13 | `Scene.pml: Option<PmlSpec>`, skipped when absent, no version bump | old v1 files parse and round trip unchanged, as the Phase 1 rule says |
| 2026-11-13 | One layer thickness on all four sides, PEC wall behind it, layer from the wall node to the interface node d | simplest layout that covers every face and corner |
| 2026-11-13 | CPML in the single TMz loop: every stencil term times the 1/kappa of its axis, plus psi | with identity profiles each operation is the Phase 1 operation in the same order, so a layer-free scene is bit for bit unchanged |
| 2026-11-13 | psi arrays are full `nx * ny` on the CPU | plain and obviously correct; the GPU can store slabs later |
| 2026-11-13 | `cpml.rs` owns every CPML formula; `Cpml` holds b, c, 1/kappa at integer and half positions per axis | both kernels upload the same arrays, no formula in WGSL |
| 2026-11-13 | Where sigma = 0, b = c = 0 but 1/kappa is kept | `sigma_scale = 0` with `kappa_max > 1` must still stretch; at depth 0 kappa is exactly 1 anyway |
| 2026-11-13 | Layer materials must match their clamped interior node | one rule covers "uniform along the normal" on edges and corners, and lets PEC guides run through |
| 2026-11-13 | Reflection harness lives in `fdtd-core::reflection`, shared by tests, sweep and report | one definition of the measurement |
| 2026-11-13 | `PmlSpec::recommended` picked by the worst reflection over every probe, 80 degrees included | the 0 to 70 degree optimum (kappa_max = 1) leaks 27 dB more at grazing; users will paint near edges |
| 2026-11-13 | `fdtd report` writes its SVG plot directly from Rust | no plotting stack in CI, output diffs cleanly, renders on GitHub; Python stays for golden data |

## Measured numbers

Fill in as they become available. These go in the README.

| Quantity | Value | Conditions | Date |
|---|---|---|---|
| CPML reflection, 10 cells | -86.5 dB worst over 0 to 70 deg (worst probe is the interface corner); -85.5 dB at 80 deg | m 3, sigma_scale 1.25, kappa_max 2, alpha_max 0.15; Ricker peak at 20 cpw, S = 0.5, 1000 steps; gate -60 | 2026-11-13 |
| CPML reflection, best tuned | -127.3 dB worst over 0 to 70 deg at 16 cells; -125.4 dB at 80 deg | m 4, sigma_scale 1.0, kappa_max 2, alpha_max 0.1; target -80 | 2026-11-13 |
| CPML sweep extremes | 0 to 70 deg only: -91.9 dB at 10 cells, -134.5 at 16, -144.9 at 20 (all kappa_max = 1); doc start values (m 3, scale 1, kappa 5, alpha 0.05): -75.1 / -99.1 / -104.9 | 720 + 1120 + 288 setting sweeps, `fdtd pml-sweep` | 2026-11-13 |
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
| A8 line source vs Hankel | 3.9e-4 complex error (amplitude 2.1e-5, phase 3.8e-4 rad); 5.9e-2 if the continuum k is used | 2 to 5 wavelengths, axis and diagonal, f0 = 0.05, S = 0.5, 16-cell layer, gate 0.02 | 2026-11-13 |
| Late-time energy with CPML | 2000-step maxima never rise; final 3.7e-32 of peak after 100000 steps | 50 x 50, 10-cell layer, Ricker over by step 200 | 2026-11-13 |
| Mirror symmetry with CPML | 2.1e-15 of peak about both axes | 61 x 51, 10-cell layer, dielectric, lossy and PEC shapes | 2026-11-13 |
| Reflection harness self-checks | reference vs 1.5x-margin reference: exactly equal; bare PEC box: -0.8 to +1.4 dB | small sweep, 400 steps | 2026-11-13 |

## Known deviations from reference

| Item | Deviation | Reason | Acceptable? |
|---|---|---|---|
| 02-numerics CPML `sigma_opt` | I use `0.8(m+1)/(eta0 dx sqrt(eps_r mu_r))` (Taflove eq. 7.67); the doc's `1/eta` is off by a factor eps_r | the stretch uses eps0, so equal absorption needs 1/sqrt(eps_r) | yes, done (eps_r of material 0) |
| 02-numerics "H coefficients half a cell deeper" | I evaluate every profile at the component's true coordinate | the doc's rule only holds on the high faces | yes, done |
| 02-numerics TMz PML update | `(Ez[j+1]*ik - Ez[j]*ik)` instead of `(Ez[j+1] - Ez[j]) * ik`, same for every term | needed for the bit-for-bit Phase 1 equivalence and the stencil oracle; same value to round-off | yes |
| 02-numerics `kappa_max` 5 to 15 | Recommended layers use kappa_max = 2 | the sweep found 2 best over all angles; 5 costs about 8 dB at 10 cells | yes, measured |
| Phase 2 late-time "energy decays monotonically" | Checked on 2000-step window maxima of the field energy | psi holds part of the energy; with alpha > 0 a slow beat makes 500-step maxima rise 1.5x at the 1e-14 level, while 2000-step maxima fall for alpha 0, 0.15, 0.5 and kappa 1 | yes |
| A10 "target, record" | Also asserted at -80 dB | it clears the target by 47 dB; the bound comes from the spec, not from my output | yes |
| 04-validation report "one plot per Tier 2 case" | report.md covers A9, A10 and links the A7 plot; A8 numbers live in the test output and here | putting A8 in the report means moving golden-data handling into a library for a printout | yes, revisit when the report grows |
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
| 2 Analytic | yes | | A1 to A3 (1D), A4 to A7 (2D TMz), A8 to A10 and late-time stability (CPML) |
| 3 Invariant | yes | | quiescence, linearity (1D and 2D); energy conservation and decay, reciprocity, mirror symmetry, padding (2D); mirror symmetry, padding, quiescence, linearity with CPML |
| Report | | yes | `fdtd report` regenerates `validation/report`; not wired into CI yet |
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

### 2026-11-13 (Phase 2)
**Landed:** Phase 2 is done. The CPML runs through the same TMz loops as
everything else, with identity profiles in the interior, so a scene without a
layer is still bit for bit the Phase 1 solver. The untouched Phase 1 oracle
proves that, and a second oracle written from doc 02 matches the CPML update
bit for bit with random profiles. A mirror test about both axes catches a
half-cell slip in any profile (a deliberate one-cell shift breaks it by 15
percent). The reflection harness, `fdtd pml-sweep`, and `fdtd report` all share
one measurement. A9 is -86.5 dB at 10 cells against a -60 gate, A10 is -127.3
dB at 16 cells against a -80 target, A8 matches Hankel to 3.9e-4 against 0.02,
and 100000 steps decay to 4e-32 of peak. The README draft shows the reflection
plot. 105 tests pass, the suite takes about 15 s, and fdtd-core still builds
for wasm32. 14 commits, nothing pushed.
**Broke:** Two of my own tests were wrong, not the code. I assumed the CPML c
coefficient grows with depth, but near the wall kappa grows and c shrinks like
(b - 1) / kappa, so only 1/kappa is monotone and I test that. The late-time
check failed at first with 500-step windows: the field energy rose from 4.9e-14
to 7.4e-14 at step 3500. Before touching the window I ran four layer variants.
Only alpha > 0 rises, and only in short windows. With 2000-step windows every
variant falls monotonically to the floor, so it's a slow beat between the
leftover low-frequency field and the psi memory, not growth. Also, the sweep's
first winner (kappa_max = 1, -91.9 dB over 0 to 70 deg) leaked -65 dB at 80 deg,
so I ranked by every probe instead and picked kappa_max = 2.
**Open:** CI still hasn't run on GitHub, and it doesn't regenerate the report
yet. Tuning used one pulse (20 cells per wavelength at its peak) with the
source 20 cells from the face; sources much closer to the layer or much lower
frequencies are untested.
**Next:** push, watch CI go green, then plan Phase 3 (sources, TF/SF, DFT
monitors) in detail.

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
