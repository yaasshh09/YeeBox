# Validation

"I built a wave simulator" is unverifiable. "Absorbing boundary measured at
-7x dB, phase velocity matched to the discrete dispersion relation to 0.01
percent, GPU kernel equal to an independent reference, all in CI" is a claim
anyone can check. This file is the difference.

## Five tiers

1. **Unit**: single functions against closed forms
2. **Analytic**: full solves against exact physics or exact discrete identities
3. **Invariant**: properties that hold for any correct run
4. **Cross-implementation**: GPU f32 kernel against CPU f64 reference
5. **External reference**: Python-generated analytic series, and a small set of
   scenes cross-checked against Meep

All tiers run from `cargo test` and `fdtd validate`. Tolerances below are
gates unless marked target.

## Tier 1: unit

- Unit conversion round trip to 1e-14
- Yee index helpers: position of every component, array sizes, neighbor maps
- Material coefficient precompute: lossless gives `Ca = 1`, `Cb = dt/eps`;
  large sigma drives `Ca` toward -1 from above, never beyond
- CPML profiles: interior values are exactly `kappa = 1, b = 0, c = 0`;
  profiles are monotonic; half-cell profiles are offset by half a cell
- Numerical wavenumber solver: recovers the continuum `k` as `dx -> 0`,
  second-order convergence
- Waveforms: Gaussian has zero DC to within tolerance after windowing; CW ramp
  is continuous in value and first derivative
- Scene JSON round trip, and migration from every older format version

## Tier 2: analytic

| # | Case | Check | Gate |
|---|---|---|---|
| A1 | 1D, S = 1 | pulse shape after 1000 cells equals initial shape | 1e-12 |
| A2 | 1D, S = 0.5 | measured phase velocity vs discrete relation | 0.01% |
| A3 | 1D PEC wall | reflected pulse is inverted, equal magnitude | 1e-10 |
| A4 | 2D stability | bounded at S = 0.70, unbounded at S = 0.72 | pass/fail |
| A5 | 2D dispersion | wavelength on axis and diagonal vs discrete relation | 0.01% |
| A6 | PEC cavity | first 6 resonances vs discrete prediction | 0.01% |
| A7 | PEC cavity | convergence to analytic `f_mn`, order 2 | order in [1.8, 2.2] |
| A8 | Line source | amplitude and phase vs Hankel, r > 2 wavelengths | 2% |
| A9 | CPML | reflection, normal to 70 degrees, 10 cells | <= -60 dB |
| A10 | CPML tuned | reflection, 16+ cells | -80 dB target |
| A11 | Waveguide | cutoff frequency and guided wavelength | 0.5% |
| A12 | Waveguide | evanescent decay constant below cutoff | 1% |
| A13 | Dielectric slab | R(f) and T(f) vs Airy formula | 0.01 absolute |
| A14 | Lossy slab | attenuation vs analytic | 1% |
| A15 | TF/SF | leakage into scattered region, empty domain | <= -120 dB |
| A16 | Drude slab | R(f), T(f) vs analytic complex eps | 0.02 absolute |
| A17 | Lorentz slab | R(f), T(f) vs analytic complex eps | 0.02 absolute |
| A18 | TEz/TMz duality | vacuum line source fields equal under duality | 1e-10 |
| A19 | PEC cylinder | scattering width vs series, ka = 1, 2, 5 | 1 dB |
| A20 | Dielectric cylinder | scattering width vs series, eps = 4 | 1 dB |
| A21 | Drude cylinder TEz | resonance frequency vs wp/sqrt(eps_inf+1) | 5% |
| A22 | Double slit | far-field pattern vs Fraunhofer | 1 dB over main lobes |
| A23 | Phased array | beam direction vs array factor | 1 degree |
| A24 | Photonic crystal | gap edges vs reference | 3% |

Notes:

- A19, A20: compare only at angles where the reference is within 20 dB of its
  peak; deep nulls are dominated by staircasing error. Also demonstrate that
  the error decreases as the cylinder radius in cells doubles.
- A13, A16, A17: valid over the band where the wavelength inside the material
  is at least 20 cells. State the band in the test. A16 and A17 use
  lambda0 / sqrt(|eps(w)|) with the complex eps, run in both TMz and TEz, and
  put the plasma or resonance frequency inside the band (f = 0.02).
- A18: compare Hz of the TEz kernel with Ez of the TMz kernel after the same
  number of steps, with the TEz waveform moved half a step earlier. The
  duality is exact on this layout, closed box and CPML alike; a guard checks
  that the unshifted source is visibly off.
- A21: k a = 0.1 at the quasi-static resonance, Q about 10, gated at a radius
  of 10 cells. The peak is the tallest one, picked blind. Staircasing splits
  the surface mode at small radii (several peaks at 5 cells); the trend over
  5, 10, 20 cells is `validation/report/drude_cylinder.png`.
- A24: confirm the reference gap edges with MPB or the textbook before
  enabling this as a gate. Until confirmed it is a recorded measurement, not a
  pass/fail test.

## Tier 3: invariants

Run in debug builds on every validation scene.

- **Energy conservation.** Lossless closed PEC cavity: the discrete energy
  defined in `docs/01-theory.md` (with H at two adjacent half steps) is
  constant to 1e-10 relative over 1e4 steps on the f64 kernel.
- **Energy decay.** With lossy material and no source, energy is
  non-increasing. Any increase is an instability or a sign error.
- **Divergence.** Discrete divergence of D stays at round-off level in
  source-free regions (TEz).
- **Reciprocity.** Swap source and probe positions in any linear scene: the
  recorded signal is identical to 1e-9. Cheap, powerful, and catches
  asymmetric indexing bugs that nothing else does.
- **Symmetry.** A mirror-symmetric scene with a centered source gives a
  mirror-symmetric field to round-off.
- **Linearity.** Doubling the source amplitude doubles every field value.
- **Padding.** Padded buffer entries are zero at all times.
- **Quiescence.** No source means all fields stay exactly zero.

## Tier 4: GPU against CPU

For every scene in `validation/scenes/` (at least 12, covering every feature
flag: plain, lossy, PEC shapes, CPML, each source type, TF/SF, Drude, Lorentz,
TEz, painting mid-run):

    run both kernels for 2000 steps
    assert max |gpu - cpu| / max |cpu| < 1e-4, for every field

Also a long run (50000 steps) asserting the difference grows no faster than
linearly, which rules out a hidden instability in the f32 kernel.

A feature is not done until its scene is in this set.

## Tier 5: external reference

- Bessel/Hankel series, Airy slab formulas, array factors: generated by
  `validation/gen/*.py` with SciPy, committed as CSV or NPZ with the generator
  script and pinned package versions beside them.
- Meep cross-check: three scenes (dielectric cylinder scattering, Drude slab,
  waveguide bend transmission) reproduced in Meep, results committed. Meep is
  the standard open FDTD package; agreement with it is a strong, independent
  statement. This runs offline, not in CI. Document resolution and settings so
  the comparison is like for like.

## The validation report

`fdtd report` regenerates `validation/report/report.md` with one plot per
Tier 2 case: simulation overlaid on reference, the measured error, the gate,
and pass/fail. CI regenerates it on every push so it cannot go stale. The web
app links to it from a "Verify" panel. The README shows four plots: CPML
reflection vs angle, dispersion error vs cells per wavelength, cylinder
scattering width, and GPU vs CPU difference over time.

## Convergence studies

For A7, A13, A19: run at three or four resolutions, plot error vs cell size on
log-log axes, report the fitted order. Second order is expected for smooth
problems. First order is expected and correct for staircased curved surfaces.
Reporting the observed order, including where it drops, demonstrates
understanding better than any single tight number.

## What is never acceptable

- A test that asserts the code's current output with no independent reference
- A tolerance loosened to make a failing test pass without a written reason in
  `PROGRESS.md`
- A GPU test that silently passes when no adapter is available
