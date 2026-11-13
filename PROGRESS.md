# Progress

State of the world. Read first every session, append before ending. Newest
entry at the top of the session log.

## Current state

**Active phase:** Phase 1 (Phase 0 done, CI not yet run on GitHub)
**Blocked on:** nothing
**Next action:** push so CI runs once, then plan Phase 1 in detail (TMz
reference kernel, materials, scene format, energy)

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

## Known deviations from reference

| Item | Deviation | Reason | Acceptable? |
|---|---|---|---|
| 02-numerics CPML `sigma_opt` | I use `0.8(m+1)/(eta0 dx sqrt(eps_r mu_r))` (Taflove eq. 7.67); the doc's `1/eta` is off by a factor eps_r | the stretch uses eps0, so equal absorption needs 1/sqrt(eps_r) | yes, Phase 2 |
| 02-numerics "H coefficients half a cell deeper" | I evaluate every profile at the component's true coordinate | the doc's rule only holds on the high faces | yes, Phase 2 |
| 04-validation "Gaussian has zero DC" | Zero-DC tests target Ricker and the sine-modulated Gaussian; the plain Gaussian test checks DC = sqrt(pi) tau | a plain Gaussian has nonzero integral | yes |
| A4 at S = 0.72 | Will run through a `pub(crate)` unchecked constructor inside fdtd-core | the public constructor must reject S above the limit | yes, Phase 1 |
| Energy W | Will keep a copy of the previous H | after a full step the solver holds H^{n+1/2} and E^{n+1} | yes, Phase 1 |
| `courant_limit(dims)` | Returns `Result` and rejects dims outside 1..3 | no panics reachable from input | yes |

## CI coverage

Which tiers run in CI and which run only locally (GPU tests depend on a
software adapter being available on the runner).

| Tier | CI | Local only | Notes |
|---|---|---|---|
| 1 Unit | yes | | units, grid, dispersion, waveforms, 1D solver, npz |
| 2 Analytic | yes | | A1 to A3 (1D) |
| 3 Invariant | yes | | quiescence, linearity (1D) |
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
