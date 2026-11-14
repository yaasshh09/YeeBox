# Golden-data generators

Python scripts that produce independent reference data (analytic series,
Airy slab formulas, array factors) and plots. Outputs are committed under
`validation/golden/`. Versions are pinned in `requirements.txt`; regenerate
golden data only with those versions.

## Setup

    python -m venv .venv
    .venv/Scripts/python -m pip install -r validation/gen/requirements.txt   # Windows
    .venv/bin/python -m pip install -r validation/gen/requirements.txt       # Linux, macOS

## Scripts

| Script | Input | Output |
|---|---|---|
| `plot_1d.py` | NPZ from `fdtd run` | space-time plot of Ez (Phase 0 done check) |
| `plot_cavity_convergence.py` | runs `fdtd run --scene` itself | `validation/report/cavity_convergence.png`, cavity error vs resolution with the fitted order (Phase 1 done check) |
| `line_source_hankel.py` | none | `validation/golden/line_source_hankel.csv`, `H0^(2)(k r) / H0^(2)(k r0)` at the A8 probes, with the numerical and the continuum k |
| `slab_airy.py` | none | `validation/golden/slab_airy.csv`, A13 slab R and T (continuum Airy and the exact discrete Yee slab) at L = 20, 40, 80; `validation/golden/lossy_halfspace.csv`, A14 attenuation per cell (continuum and discrete) |
| `waveguide.py` | none | `validation/golden/waveguide.csv`, A11 and A12 TM1 guided beta, evanescent alpha and the cutoff (continuum and discrete) |

Phases 0 and 1 need no golden data: every reference so far is an exact
discrete identity or a closed form computed in Rust. The cavity plot script
re-measures the resonances in NumPy on its own, as a cross-check of A7.

Phase 2 adds the first golden file. Rust has no Bessel functions, so the
Hankel values for A8 come from SciPy. The script output is byte for byte
reproducible with the pinned versions; rerun it and `git diff` should be empty.

Phase 3 adds the slab and waveguide files. Their gates are the continuum
formulas; each file also carries the exact discrete Yee answer, so the
measured error can be split into what the grid itself predicts and what the
measurement adds. The A13 and A14 bands were picked from those discrete
predictions before any FDTD run (predicted error at most half the gate); the
scripts print the budget.
