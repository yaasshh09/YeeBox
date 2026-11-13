# Phase 1: 2D TMz reference

**Target: 1 to 2 weeks. Docs: 01-theory, 02-numerics (stability, dispersion), 04-validation.**

The oracle is born. Closed domains only, PEC walls, no absorbing boundary yet.

## Scope

1. `tm.rs`: Ez, Hx, Hy update with per-cell `Ca`, `Cb` from the material table
2. `material.rs`: table, coefficient precompute, vacuum, dielectric, lossy, PEC
3. `scene.rs`: `Scene` struct, serde JSON, `validate()`, rasterization of
   rect, circle, polygon to material ids
4. `energy.rs`: discrete energy with the two-half-step H product
5. `monitor.rs`: point probes with time history
6. `Kernel` trait and the CPU implementation of it
7. `fdtd-cli run` for 2D scenes with NPZ output

## Acceptance

- A4: bounded at S = 0.70 for 20000 steps, unbounded at S = 0.72
- A5: wavelength along the axis and along the diagonal matches the discrete
  dispersion relation to 0.01 percent
- A6: first six cavity resonances match the discrete prediction to 0.01 percent
- A7: convergence to analytic cavity frequencies at order 1.8 to 2.2
- Invariants: energy conserved to 1e-10 over 1e4 steps in a lossless cavity;
  energy non-increasing with loss; reciprocity to 1e-9; mirror symmetry;
  padding stays zero
- Scene JSON round trip

## Do not

- Add CPML, TF/SF, or dispersive media
- Optimize the loops

## Done when

A committed plot shows cavity resonance error vs resolution on log-log axes
with the fitted order printed on it.
