# Pitfalls

Read this when a field explodes, leaks, drifts, or simply looks wrong.

## Debugging order

1. **Courant number.** Print `S`. If it exceeds `1/sqrt(2)` nothing else
   matters.
2. **Index staggering.** Re-derive which neighbor each difference uses from
   the component positions in `docs/01-theory.md`. Most bugs are a `+1` that
   should be `-1`, or an Hx difference taken along x.
3. **Run the invariants.** Reciprocity and mirror symmetry localize indexing
   bugs fast. Energy growth localizes sign errors.
4. **Reduce.** 1D, vacuum, one pulse, `S = 1`. That case is exact. If it
   fails the bug is structural.
5. Only then look at PML tuning, material coefficients, or precision.

## Symptom to cause

| Symptom | Most likely cause |
|---|---|
| Field grows exponentially everywhere, checkerboard pattern | Courant limit violated |
| Field grows slowly inside the PML | alpha too small or wrong sign in `b`; psi update using wrong derivative |
| Growth only in a dispersive region | ADE coefficient error, or `w0*dt` too large |
| Visible reflection from the boundary, about -30 to -40 dB | H-update PML coefficients not half-cell offset; or sigma_max far from optimal |
| Reflection only at grazing angles | no CFS term (alpha = 0, kappa = 1); layer too thin |
| Wave travels at visibly different speed along axes vs diagonals | under-resolved, below about 10 cells per wavelength. This is physics of the grid, not a bug |
| Pulse develops a trailing ripple | numerical dispersion on a broadband pulse; expected, shrinks with resolution |
| Wave propagates in only some directions | one of the two H components never updated, or updated with the wrong E difference |
| Field mirrored or rotated versus expectation | row-major versus column-major confusion between CPU and GPU buffers |
| DC offset slowly filling the domain | source with nonzero time integral, or unramped CW source |
| Scattered-field region not quiet with TF/SF | correction applied to the wrong component or wrong side; auxiliary grid offset by half a cell or half a step |
| TF/SF leaks about -40 dB at oblique incidence | dispersion mismatch between 1D and 2D grids; expected, which is why v1 is axis-aligned only |
| Energy of a closed cavity oscillates | energy computed with H squared at one half step instead of the two-half-step product |
| Spectrum has ripples | DFT stopped before fields decayed |
| Flux spectrum tilted or phase-shifted | missing half-step phase correction on H |
| Transmission above 1 | normalization run differs from the structure run in source position, monitor position, or resolution |
| Material appears to have gain | time-convention mismatch between solver and reference formula |
| Metal objects absorb instead of reflect | PEC implemented as large sigma with too few cells; use a true PEC mask |
| Plasmon demo shows nothing in TMz | correct. Surface plasmons need TEz |
| Extra resonance peaks on a curved metal particle | staircasing; reduce cell size, report honestly |
| GPU and CPU differ by a constant factor | coefficient precompute duplicated in the shader with a different `dt` |
| GPU and CPU differ only near edges | shader reading padded entries or missing bounds check |
| GPU result changes with workgroup size | out-of-bounds writes at the dispatch overhang |
| Works natively, black canvas in browser | surface format or alpha mode mismatch; device lost; check the console |
| Works in Chrome, fails in Safari or Firefox | relying on an optional WebGPU feature or a limit above the default |
| Painting lags the cursor | uploading the whole material buffer per stroke instead of a sub-rectangle |
| Frame rate collapses when a probe is open | full-field readback every frame |

## Specific traps

**Comparing against continuum physics at tight tolerance.** The grid obeys its
own dispersion relation. Tests that need better than about 1 percent must
compare against the discrete prediction.

**One code path per kernel for PML and interior.** Separate interior and PML
loops double the chances of an indexing mismatch at the interface. Use
identity coefficients in the interior.

**Duplicated formulas.** Every coefficient (`Ca`, `Cb`, CPML `b` and `c`, ADE
constants) is computed in exactly one Rust function and uploaded. If a
formula appears in WGSL it is a bug waiting to happen.

**Hard sources as the default.** A hard source is a perfect reflector at the
source point. Users will not understand why the wave bounces off nothing.

**Soft source amplitude.** A soft source adds current, so the radiated
amplitude depends on `dx` and `dt`. Normalize if a preset needs a specific
field level, and never compare absolute amplitudes across resolutions without
it.

**Dispersive material against the PML.** Unstable in the standard
formulation. The scene validator must reject it rather than let users
discover it.

**f32 for PML measurement.** The noise floor of f32 accumulation sits near the
numbers being measured. Measure on f64.

**Assuming WebGPU defaults are generous.** Default limits for buffer size,
bindings per group, and storage buffers per stage are modest. Request what is
needed explicitly and degrade the grid size when a limit is refused.

**Storing presets as code.** Presets are `Scene` JSON files. That keeps them
shareable, testable in Tier 4, and migration-safe.

**Calling a 2D scene a patch antenna or a dipole.** In 2D everything is
infinite along z. Name presets for what they are: line source, slot, slab,
rod, cylinder.

**Mobile.** Touch events, device pixel ratio, and thermal throttling. Default
to a smaller grid on mobile and let the adaptive step count do its job.

## When stuck

Write the smallest scene that shows the problem, run it on the CPU reference,
and dump fields every step to NPZ. Plot in Python. A wrong field is usually
obvious in a picture within the first 20 steps, long before it shows in a
summary number.
