# Phase 0: Foundation and 1D

**Target: 1 week. Docs: 01-theory (units, staggering), 02-numerics (stability, dispersion), 03-architecture.**

No 2D, no GPU, no web. Build the substrate and prove the simplest exact case.

## Scope

1. Cargo workspace with `fdtd-core` and `fdtd-cli` only. Empty placeholders for
   the other crates are fine; no code in them.
2. `units.rs`: normalized and physical conversion with round-trip tests
3. `grid.rs`: Yee index helpers for 1D and 2D TMz, array sizes, component
   positions as exact rationals (integer or half-integer)
4. `dispersion.rs`: numerical wavenumber solver for 1D and 2D
5. `source.rs`: Gaussian, modulated Gaussian, Ricker, ramped CW
6. `fdtd1d.rs`: 1D solver with PEC ends and a simple first-order absorbing end
   (exact at S = 1), soft and hard sources
7. `validation/gen/` Python environment with pinned requirements and a README
8. CI: fmt, clippy with warnings denied, tests

## Acceptance

- A1: at S = 1 a pulse travels 1000 cells with shape error under 1e-12
- A2: at S = 0.5 the measured phase velocity matches the discrete relation to
  0.01 percent at 10, 20, and 40 cells per wavelength
- A3: PEC reflection is inverted and equal in magnitude to 1e-10
- Numerical wavenumber converges to the continuum value at second order
- Solver construction with S above the limit returns an error
- Quiescence and linearity invariants pass
- CI green

## Do not

- Write any 2D update
- Add wgpu or wasm-bindgen as dependencies
- Create the web project

## Done when

`cargo test` is green, and `fdtd run` can dump a 1D space-time field to NPZ
that plots as a clean pulse reflecting off a wall.
