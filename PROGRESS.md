# Progress

State of the world. Read first every session, append before ending. Newest
entry at the top of the session log.

## Current state

**Active phase:** Phase 5 (Phases 0 to 4 done, CI not yet run on GitHub)
**Blocked on:** nothing
**Next action:** Phase 5 part C (`fdtd-wasm`: errors, GPU sim binding, edits, CPU fallback, self-test); parts A and B are done; the owner pushes when ready so CI runs once on lavapipe

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
| 2026-11-14 | `Scene.periodic_y`: Ez rows wrap, no y walls, CPML on the x faces only (`Cpml::x_only`) | a y-uniform plane wave cannot exist between PEC y walls in TMz; also what A24 needs later |
| 2026-11-14 | Periodic loops use wrapped neighbor indices that equal `k + nx` / `k - nx` off the seam | non-periodic scenes stay bit for bit the Phase 2 solver |
| 2026-11-14 | TF/SF box `Scene.plane_wave`, axis-aligned, faces and the nodes across them in material 0, outside the layer | the corrections assume the background on both sides of each face; typed `PlaneWaveFault` errors |
| 2026-11-14 | The auxiliary grid is `Sim1d` with background eps, a hard launch on node 0 and a 32-cell far-end CPML without alpha | its update is written in the 2D operation order, so a y-uniform 2D field reproduces it bit for bit |
| 2026-11-14 | TF/SF order: 2D H, H corrections from aux E^n, aux step, 2D E, E corrections from aux H^(n+1/2) | A15 at -302 dB, round-off |
| 2026-11-14 | Phased array is a helper (`scene::phased_array`) that expands to `RampedCw` point sources | no scene type until the UI needs one |
| 2026-11-14 | DFT samples enter at their true time; H at (n - 1/2) S | the half-step correction is the time argument, pinned by the plane-wave identity Hy_avg / Ez = -cos(k/2) |
| 2026-11-14 | Spectral monitors (`Monitor::Dft`, `Monitor::Flux`) are scene objects; `Kernel::monitor_spectra` reads them | the GPU kernel must produce the same spectra |
| 2026-11-14 | R from a flux line in the SF region behind the entry face, T in the TF region, both over a normalization run's incident flux (`spectra::rt_spectrum`) | no incident subtraction needed; one definition shared by tests, validate and report |
| 2026-11-14 | Spectrum runs end when max abs(Ez) stays below 1e-6 of peak for one pulse length, else `NotDecayed` | no guessed step counts, no ringing spectra |
| 2026-11-14 | Golden files carry both the continuum answer (the gate) and the exact discrete Yee answer (a guard at 1e-4) | separates the grid's own error from measurement error |
| 2026-11-14 | A13 and A14 bands picked from the discrete model before any run, predicted error at most half the gate | choosing the experiment from theory, not tuning to a result |
| 2026-11-14 | Normal-incidence scenes (A13, A14, the TF/SF aux grid) use the graded preset with alpha_max = 0 | the CFS alpha stops absorption below w of about alpha (-41 dB at f = 0.0025); a normal-incidence problem has no evanescent waves for alpha to help |
| 2026-11-14 | The acceptance cases live in `fdtd-core::cases`; tests, `fdtd validate` and `fdtd report` all run them | the table cannot drift from what the tests check |
| 2026-11-14 | One shared device, `Gpu::shared()`; `GpuSim::new(&'static Gpu, &Scene)` | one adapter per process in tests and CLI; the browser phase passes its own device |
| 2026-11-14 | GPU buffers: f32 fields and full-size psi padded like the CPU, u32 material ids, a material table with an extra wall entry | the spec layout; painting will write ids only |
| 2026-11-14 | CPML profiles and the material table packed into one `vec4` storage buffer | WebGPU allows 8 storage buffers per stage; the E update would need 11 |
| 2026-11-14 | Two main passes per step in one code path; the workgroup edge is a text constant swapped at pipeline creation | lets Tier 4 test 4, 8 and 16 from one shader |
| 2026-11-14 | Batches of up to 64 steps in one compute pass; per-step `Params` slots read through a dynamic uniform offset | one submit per frame, no per-step uploads |
| 2026-11-14 | Source samples and TF/SF incident values computed on the CPU in f64 and uploaded per batch; the auxiliary grid stays the core `Tfsf` | no formula and no second auxiliary update in WGSL; the 1D grid costs nothing |
| 2026-11-14 | TF/SF corrections are data (`Tfsf::h_corrections`, `e_corrections`), groups of two in loop order; the CPU iterates the same lists | one definition for both kernels, bit for bit with the old loops; one-node boxes stay race-free |
| 2026-11-14 | Sources run in one sequential invocation in scene order | hard and soft sources on one node land as on the CPU |
| 2026-11-14 | Monitors are taps (`Monitor::taps`): the GPU gathers samples, the CPU accumulates probes, DFTs and flux in f64 with the core code | no DFT in WGSL, no f32 accumulation over 1e4 steps |
| 2026-11-14 | `FdtdError::Gpu`; device errors are stored and returned by the next read | `Kernel::step` has no `Result`; nothing panics |
| 2026-11-14 | GPU tests use a libtest-mimic harness: no adapter means ignored with a banner; `FDTD_REQUIRE_GPU=1` (CI) makes it a failure | the built-in harness cannot skip at runtime |
| 2026-11-14 | `kernel::compare` normalizes by the reference peak over the run so far, checked every 250 steps | the CLAUDE.md wording "1e-4 of peak field"; f32 error scales with the largest field held (see deviations) |
| 2026-11-21 | The site deploys to GitHub Pages from an Actions job on main | no second account; the repo is already on GitHub |
| 2026-11-21 | The UI follows AtomSIM (`../atom_sim/web`): tokens, Poppins, three columns, mobile shell under 900 px, same web stack | the owner's sibling project; one lab-instrument idiom across both |
| 2026-11-21 | `Scene.paint`: a dense override id per node, JSON as runs `[[null, 500], [2, 12]]`, at most `MAX_NODES` entries checked before allocating | strokes cost a few runs; shared links are untrusted input |
| 2026-11-21 | Brush strokes skip sources and the TF/SF faces with the nodes across them; CPML nodes take the paint of the interface node they clamp to | a stroke on a valid scene leaves it valid, and walls run on through the layer |
| 2026-11-21 | A fixed paint palette (`material::PALETTE`), merged into the table before a kernel is built; the table is fixed for a kernel's life (`BuildKey`, else `FdtdError::Rebuild`) | painting writes ids only, the GPU coefficient buffer never resizes |
| 2026-11-21 | Source edits accept vacuum wavelengths of 4 to 200 cells; a CW retune shifts the phase so the drive is continuous at the edit time | 4 clears the grid cutoff on axis; 200 is where the measured CPML still holds -101 dB (closes the Phase 3 low-frequency question) |
| 2026-11-21 | `Kernel::set_materials(scene, bounds)` and `set_sources(scene)`; both kernels rewrite the same `material_edit` rectangle; a node turned PEC is zeroed by the next step, not at once | identical semantics on both kernels with no extra code |
| 2026-11-21 | The GPU sources stage always exists, empty when there are no sources | a live edit can add the first source by rebuilding one bind group |
| 2026-11-21 | `kernel::Comparator` and `Snapshot` hold the running-peak measure; `compare` is built on them | the async browser self-test reports the same numbers |
| 2026-11-21 | Full-size presets live under `web/`; their compact 256 versions go in `validation/scenes` | the scene set caps scenes at 200 x 160 nodes to keep 2000 CPU steps quick |
| 2026-11-21 | `Gpu::request(instance, surface)` is the async constructor; `Gpu::new`, `shared`, blocking reads and `impl Kernel for GpuSim` are native only; CI builds `fdtd-gpu` for wasm32 | a browser device is not `Send`, and nothing may block there |
| 2026-11-21 | Taps are copied to a staging buffer per batch and drained in submission order without blocking; native reads of probes and spectra wait for the queue first | one code path for browser and tests; throughput unchanged at 3.04 Gcells/s |
| 2026-11-21 | The colormaps are built in OKLab from stated rules (blue / dark neutral / orange diverging, dark to warm white sequential), 257 entries so zero is entry 128 | no published table offline; the rules are what the tests check |
| 2026-11-21 | The render pass reads Ez, ids and the intensity mean straight from the simulation buffers, nearest node per pixel; `render::Viewport` (y up) is the one view transform | no readback per frame; GPU render, CPU fallback and pointer mapping agree |
| 2026-11-21 | Intensity view: an exponential running mean of Ez^2 (time constant tau steps), drawn as `2 mean gain^2` on the sequential map; display only | a sinusoid at full-color amplitude reads 1 |

## Measured numbers

Fill in as they become available. These go in the README.

| Quantity | Value | Conditions | Date |
|---|---|---|---|
| CPML reflection, 10 cells | -86.5 dB worst over 0 to 70 deg (worst probe is the interface corner); -85.5 dB at 80 deg | m 3, sigma_scale 1.25, kappa_max 2, alpha_max 0.15; Ricker peak at 20 cpw, S = 0.5, 1000 steps; gate -60 | 2026-11-13 |
| CPML reflection, best tuned | -127.3 dB worst over 0 to 70 deg at 16 cells; -125.4 dB at 80 deg | m 4, sigma_scale 1.0, kappa_max 2, alpha_max 0.1; target -80 | 2026-11-13 |
| CPML sweep extremes | 0 to 70 deg only: -91.9 dB at 10 cells, -134.5 at 16, -144.9 at 20 (all kappa_max = 1); doc start values (m 3, scale 1, kappa 5, alpha 0.05): -75.1 / -99.1 / -104.9 | 720 + 1120 + 288 setting sweeps, `fdtd pml-sweep` | 2026-11-13 |
| TF/SF leakage (A15) | -303.9 dB worst | empty 100 x 100 domain, all four directions, closed box and periodic y; gate -120 | 2026-11-14 |
| Dielectric slab R, T vs Airy (A13) | 4.88e-3 (the discrete model predicts 4.88e-3); 1.4e-7 from the exact discrete slab; order 2.006 over L = 20, 40, 80 | eps 4, 20 cells, f 0.0025 to 0.025, alpha 0 layer; gate 0.01 | 2026-11-14 |
| R + T, lossless slab | 1 to 6.9e-8 | the A13 slab; gate 0.005 | 2026-11-14 |
| Lossy attenuation (A14) | 0.400 percent worst (the discrete model's own error); 2.5e-6 from the exact discrete alpha | eps 2.25, sigma 0.05, f 0.008 to 0.02, background medium, line source; gate 1 percent | 2026-11-14 |
| Waveguide guided wavelength, cutoff (A11) | 0.030 percent, 0.002 percent; beta 1.6e-6 from discrete | TM1, a = 40, 1.30 and 1.60 f_c, CW, DFT over whole periods; gate 0.5 percent | 2026-11-14 |
| Waveguide evanescent decay (A12) | 0.060 percent; 7.8e-8 from discrete | 0.70 f_c; gate 1 percent | 2026-11-14 |
| Half-step identity Hy_avg / Ez = -cos(k/2) | 3.6e-8 (the wrong time gives a phase of w S / 2, 0.03 to 0.13 rad) | periodic TF/SF plane wave, 5 frequencies | 2026-11-14 |
| Flux line vs analytic launch spectrum | about 1e-8 relative | periodic TF/SF, +x and -x, and normal y | 2026-11-14 |
| Reciprocity with TF/SF and monitors present | 6.1e-14 of peak | cluttered PEC box, differences against a TF/SF-only run; gate 1e-9 | 2026-11-14 |
| CPML reflection vs frequency, 16 cells, 1D | alpha 0.1: -40.8 dB at f = 0.0025, -101 to -113 dB at 0.005 to 0.01, -91 dB at 0.025; alpha 0: -157 dB or better | Ricker peak 0.01, hard launch | 2026-11-14 |
| 1D far-end CPML | 16 cells -127 / -136 dB, 32 cells -158 / -160 dB (vacuum / eps 4) | Ricker at 20 cpw in the medium | 2026-11-14 |
| `fdtd validate` runtime | about 35 s for 16 checks | release build | 2026-11-14 |
| GPU vs CPU max difference | 5.9e-6 of peak worst (`tfsf_thin_plus_x`), 3.3e-7 to 2.7e-6 elsewhere; probes 1.4e-6, spectra 6.1e-5 worst (slab flux) | 11 scenes, every field, checkpoints every 250 steps to 2000, GTX 1660 Ti; gate 1e-4 | 2026-11-14 |
| GPU vs CPU long run | 7.2e-6 after 50000 steps, fitted growth n^0.44 | `cavity_tm`, checkpoints every 5000; bound n^1.2 set before the run | 2026-11-14 |
| GPU workgroup independence | bit-identical for 4, 8, 16 | five field-only scenes, 300 steps | 2026-11-14 |
| GPU vs CPU with live edits | 6.4e-7 of peak (walls painted through the CPML, erased, source moved and retuned, a second source added); 2.0e-6 (first source added to a sourceless cavity) | checkpoints every 250 steps and at each edit, to 2000; gate 1e-4 | 2026-11-21 |
| Live paint at step 0 vs a scene painted from the start | bit-identical, with and without a CPML | CPU, 300 steps | 2026-11-21 |
| GPU intensity mean vs f64 mean of CPU Ez | 1.8e-7 of peak | `painted_walls` with CW 0.05, tau 40, 400 steps; gate 1e-4 | 2026-11-21 |
| Render at 1 px per cell | the exact table entry at a known node; CPML tint and conductor color within 1/255 | native offscreen Rgba8Unorm | 2026-11-21 |
| GPU f32 TF/SF leakage | about 5e-8 of the incident amplitude (-146 dB) | one-node box, no other source; CPU leaves 1e-16 | 2026-11-14 |
| Throughput, integrated GPU | not measured, no integrated GPU on this machine | | |
| Throughput, discrete GPU | 3.0 Gcells/s, 2864 steps/s, 47.7 steps per 60 fps frame (16-cell CPML: 3.0 Gcells/s, 48.1) | GTX 1660 Ti, Vulkan, 1024 x 1024, 8 steps per submit, `fdtd bench` | 2026-11-14 |
| Throughput, CPU reference | 68 Mcells/s | f64, one thread, 1024 x 1024 with CPML, release | 2026-11-14 |
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
| 04-validation report "one plot per Tier 2 case" | report.md has the full A1 to A15 table, the CPML and slab plots, and links the A7 plot; the other cases have a table row, not a plot | the golden handling moved into `cases` in Phase 3, so plots can follow case by case | yes, add plots as the report grows |
| 04-validation "Gaussian has zero DC" | Zero-DC tests target Ricker and the sine-modulated Gaussian; the plain Gaussian test checks DC = sqrt(pi) tau | a plain Gaussian has nonzero integral | yes |
| A4 at S = 0.72 | Runs through `from_scene_unchecked`, crate-private (no longer test-only, since `cases::tier2d::a4` uses it) | the public constructor must reject S above the limit | yes, done |
| Energy W | No stored copy of H after all; H^(n+1/2) is recomputed from the current state | simpler than a copy and bit-identical to the solver's own update | yes, done |
| A5 "along the axis" | Measured on the lowest guided mode of a wide PEC guide, 4.4 deg (10 cpw) and 9.0 deg (20 cpw) off axis, not at exactly 0 | a closed PEC domain cannot hold a ky = 0 wave; exact axis behaviour is already covered by A2 and `axis_2d_equals_1d` | yes |
| A6, A7 records | Probe records start after the Ricker pulse ends (t > 2 t0) | while the source drives the cavity, the spectrum carries a smooth background that biased the peaks by about 5e-6 and pulled the A7 order to 1.90 | yes |
| `courant_limit(dims)` | Returns `Result` and rejects dims outside 1..3 | no panics reachable from input | yes |
| 02-numerics "multiply the H transform by exp(-j w dt / 2)" | H samples enter the DFT at their true time (n - 1/2) S instead | the same correction written for H^(n+1/2) stored under index n; my storage labels H by the step that produced it | yes, documented in `monitor.rs` |
| A14 "lossy slab attenuation" | Measured as the attenuation constant of a lossy background medium driven by a line source, not a slab | a lossy slab inside a TF/SF box would put lossy material across the exit face, which the TF/SF rules forbid; the background medium has no interface at all | yes |
| Recommended CPML presets at long wavelengths | The preset's CFS alpha leaves the layer weak below w of about alpha_max (-41 dB at 400 cells per wavelength) | alpha is there for evanescent and grazing waves; normal-incidence scenes use alpha 0 | open: the UI phase needs a low-frequency preset or guidance for CW sources |
| A13 slab thickness | eps on Ez nodes i0..i1 makes a slab i1 - i0 + 1 cells thick, faces at half cells | the exact discrete transfer agrees with this to 1.4e-7 | yes |
| `fdtd run --scene` probes | It sized its probe array by every monitor; now it counts point monitors only | flux and DFT monitors have no time record | yes, fixed in 1042571 |
| 04-validation Tier 4 "max abs(gpu - cpu) / max abs(cpu)" | The denominator is the reference peak over the run so far, not at the final step | measured on `materials_clutter`: the absolute difference falls from 5e-8 to 9e-9 while the CPML drains the field 1000 times, so the final-step ratio reaches 2.8e-4 from round-off alone; without the layer it stays at 2e-6. I changed the measure after seeing that failure, so I am flagging it; it matches CLAUDE.md "1e-4 of peak field" | yes, signed off by the owner on 2026-11-21 |
| 02-numerics "six 1D buffers per axis" | One packed `vec4` buffer for all profiles and the material table | the 8 storage buffers per stage default limit | yes |
| 02-numerics "DFT accumulation on the GPU" | The GPU gathers monitor samples, the CPU accumulates in f64 | f32 accumulation loses digits; reuses the core monitor code | yes, revisit if readback ever costs frame time |
| Phase 4 "TF/SF corrections on the GPU" | Corrections run on the GPU, the auxiliary 1D grid on the CPU in f64 | no second auxiliary implementation; about 100 nodes per step | yes |
| `tfsf_thin_plus_x` scene | Has a point source besides the one-node TF/SF box | without it every H edge is scattered field, the CPU leaves 1e-16 and any per-field ratio divides two floors | yes |
| Phase 3 report text | `report.md` had lines indented by 9 spaces, so the slab section rendered as a code block | a scripted edit turned backslash line continuations into literal spaces; fixed in 7b12f54 | yes, fixed |

## CI coverage

Which tiers run in CI and which run only locally (GPU tests depend on a
software adapter being available on the runner).

| Tier | CI | Local only | Notes |
|---|---|---|---|
| 1 Unit | yes | | units, grid, dispersion, waveforms, 1D solver, npz |
| 2 Analytic | yes | | A1 to A15 through `fdtd_core::cases` (and R + T = 1), plus late-time stability with CPML |
| 3 Invariant | yes | | quiescence, linearity (1D and 2D); energy conservation and decay, reciprocity, mirror symmetry, padding (2D); the same with CPML; periodic y (1D equivalence, energy, translation); TF/SF linearity, quiescence, mirror and transpose; reciprocity with TF/SF and monitors |
| Report | | yes | `fdtd report` regenerates `validation/report` (now with the Tier 4 table and growth plot); not wired into CI yet |
| 4 GPU vs CPU | yes, on lavapipe (not yet seen running) | | `crates/fdtd-gpu/tests/tier4.rs`; CI sets `FDTD_REQUIRE_GPU=1`; locally on the GTX 1660 Ti |
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

### 2026-11-14 (Phase 4)
**Landed:** Phase 4 is done. `fdtd-gpu` runs the TMz solver in WGSL through
wgpu 30: H and E updates with the CPML in one code path, periodic y, sources,
TF/SF corrections and monitor taps. Every number the shaders use comes from
`fdtd-core`. To get there, the CPML choice, source values, monitor reads and
TF/SF corrections in the core became shared functions or data, with
`fdtd validate` byte-identical before and after each step. Tier 4 runs every
scene under `validation/scenes` (11 now, with a coverage test listing every
feature): all within 5.9e-6 of peak against the 1e-4 gate, padding exactly
zero, workgroups 4, 8 and 16 bit-identical, and 50000 steps growing like
n^0.44. `fdtd bench` measures 3.0 Gcells/s at 1024 x 1024 on the GTX 1660 Ti,
about 48 steps per frame. `fdtd report` adds the Tier 4 table and the GPU
minus CPU plot, and the README shows them. 175 tests pass. 19 commits, nothing
pushed.
**Broke:** Three things. First, `materials_clutter` failed the gate at 2.8e-4.
Before touching anything I measured: the absolute difference fell the whole
run, the CPML just drained the reference peak 1000 times, and removing any
shape changed nothing. So the measure divided round-off by a residue; I now
normalize by the peak so far and recorded it as a deviation. Second, the
one-node TF/SF box blew the ratio to 5e8: every H edge there is scattered
field, so the CPU holds 1e-16 and the GPU 5e-8. I gave the scene a source.
Third, my own mistakes: a `git checkout` while checking a sabotage threw away
an uncommitted refactor (restored from a copy I had taken), and scripted edits
turned backslash continuations into spaces, which also hit the Phase 3 report
text. Sabotage checks now run only on committed files.
**Open:** CI has never run on GitHub, and lavapipe is untested. No integrated
GPU throughput yet. The CPML low-frequency preset question from Phase 3 is
still open for the UI. The running-peak measure needs your sign-off.
**Next:** push, watch CI (lavapipe), then plan Phase 5.

### 2026-11-14 (Phase 3, part 2)
**Landed:** Phase 3 is done. The acceptance cases moved into
`fdtd-core::cases` (A1 to A15 plus R + T), and the tests now assert on them,
so the numbers are the same everywhere: every value reproduced exactly after
the move. `fdtd validate` prints the 16 checks with gates and exits non-zero
on a failure; all pass. `fdtd report` adds the full table and a slab R/T plot
(`slab_rt.svg`, `slab_rt.csv`). New example scenes: `slab_tfsf.json` (the
exact A13 scene, guarded by a test) and `waveguide_tm.json`. README and these
tables are up to date.
**Broke:** Two things. SVG tick labels printed 0.6000000000000001 (ticks are
min + k step); labels now round to the step's decimals. And `fdtd run --scene`
failed on any scene with flux or DFT monitors, because it sized the probe
array by every monitor; it counts point monitors now, with a regression test.
That fix commit (1042571) also carries the slab drift-guard test, which reads
`slab_tfsf.json` from the next commit, so the tests are red at 1042571 alone
and green from 19678fc on.
**Open:** CI has still never run on GitHub. The recommended CPML presets are
weak at long wavelengths (see deviations), which matters for CW sources in
the UI.
**Next:** push, watch CI, then plan Phase 4 (wgpu kernel matched to the CPU
reference) in detail.

### 2026-11-14 (Phase 3, part 1)
**Landed:** I added periodic y, with the CPML on the x faces only when it's
on. The 1D grid now takes a background eps and has a CPML end, and its update
matches the 2D periodic slice bit for bit. The TF/SF plane wave works in all
four directions. A15 leakage is -301.9 dB worst against a -120 dB gate.
There's a phased array helper. I added a running DFT and pinned the H half
step with a plane-wave identity: H enters at its true time (n - 1/2) S, which
is the doc's exp(-j w dt / 2) for H^(n+1/2) stored under index n. Flux line
monitors match the analytic launch spectrum to 1e-8. DFT and flux monitors
are scene objects now, and `Kernel::monitor_spectra` reads them back. The
`spectra::rt_spectrum` workflow measures R and T against a normalization run.
Golden generators `slab_airy.py` and `waveguide.py` write both the continuum
and the exact discrete Yee answer.
- A13: 4.88e-3 against the 0.01 gate (exactly the predicted discretization
  error), R + T - 1 at 6.9e-8, 1.4e-7 from the exact discrete slab, order 2.006
- A14: 0.400% against the 1% gate (lossy medium as background, line source)
- A11: guided wavelength 0.030%, cutoff 0.002%
- A12: alpha 0.060%
- Reciprocity with TF/SF and monitors present: 6.1e-14
**Broke:** At first A13 missed the exact discrete slab by 3e-3 at its lowest
frequencies. The cause was the CFS alpha in the CPML. At f = 0.0025 the
16-cell preset reflects -40.8 dB, against -157 dB with alpha 0. That's
measured in `fdtd1d` layer_reflection_against_frequency. The TF/SF 1D layer
now uses alpha 0 (it's 32 cells, -158 dB), and the 1D-like spectral scenes use
alpha 0 too. Known deviation: the recommended 2D presets absorb poorly at long
wavelengths, where w < alpha_max. That matters for CW sources in the UI phase.
Also: A14 moved from a lossy half-space inside the TF region (the TF/SF rules
forbid it) to a lossy background with a line source. One commit went in with
a clippy lint, fixed in the next commit (`279a4be`).
**Open:** I haven't written the decisions and measured-numbers tables for
Phase 3 yet. CI still hasn't run on GitHub.
**Next:** commits 17 to 21 of the plan. Move A1 to A15 into a shared `cases`
module, add `fdtd validate`, extend `fdtd report` with the table and a slab
R/T plot, add the `slab_tfsf.json` and `waveguide_tm.json` example scenes,
then the README and PROGRESS tables.

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
