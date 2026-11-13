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
| `plot_1d.py` | NPZ from `fdtd run1d` | space-time plot of Ez (Phase 0 done check) |

Phase 0 needs no golden data: every Phase 0 reference is an exact discrete
identity computed in Rust.
