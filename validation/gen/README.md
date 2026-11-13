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

Phases 0 and 1 need no golden data: every reference so far is an exact
discrete identity or a closed form computed in Rust. The cavity plot script
re-measures the resonances in NumPy on its own, as a cross-check of A7.
