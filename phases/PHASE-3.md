# Phase 3: Sources, plane waves, monitors

**Target: 2 weeks. Docs: 02-numerics (TF/SF, DFT monitors), 01-theory (analytic cases).**

Everything needed to measure spectra and to inject a clean plane wave.

## Scope

1. `tfsf.rs`: axis-aligned plane wave injection with a 1D auxiliary grid
2. Phased array source: N elements, programmable phase and amplitude
3. `monitor.rs`: running DFT at a frequency list; flux through a line segment
   with co-located E and H and the half-step phase correction
4. Transmission/reflection workflow with a normalization run
5. `validation/gen/` scripts for Airy slab formulas and waveguide dispersion
6. `fdtd validate` command that runs every case and prints a table

## Acceptance

- A15: TF/SF leakage at or below -120 dB in an empty domain
- A13: dielectric slab R and T within 0.01 absolute across the valid band
- A14: lossy slab attenuation within 1 percent
- A11, A12: waveguide cutoff, guided wavelength, and evanescent decay
- R + T = 1 to 0.005 for lossless structures across the band
- Reciprocity invariant passes with TF/SF and monitors present

## Do not

- Ship an oblique TF/SF source without dispersion matching
- Start GPU work

## Done when

`fdtd validate` prints cases A1 through A15 with measured value, gate, and
pass/fail, and `fdtd report` renders the same into `validation/report/`.
