# Progress

State of the world. Read first every session, append before ending. Newest
entry at the top of the session log.

## Current state

**Active phase:** Phase 0
**Blocked on:** nothing
**Next action:** scaffold the workspace, then write the first test of the
project: the 1D magic time step (A1)

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

## Known deviations from reference

| Item | Deviation | Reason | Acceptable? |
|---|---|---|---|
| (none yet) | | | |

## CI coverage

Which tiers run in CI and which run only locally (GPU tests depend on a
software adapter being available on the runner).

| Tier | CI | Local only | Notes |
|---|---|---|---|
| 1 Unit | | | |
| 2 Analytic | | | |
| 3 Invariant | | | |
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

### (no entries yet)
