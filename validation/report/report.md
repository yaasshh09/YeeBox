# Validation report

`fdtd report` writes this file, so I regenerate it instead of editing it.

## CPML reflection (A9, A10)

![CPML reflection vs angle](cpml_reflection.svg)

I measure it with the reference-domain method. A Ricker pulse (spectral peak at 20 cells per wavelength, S = 0.5) starts 20 cells above the bottom interface of a 165 x 60 node vacuum box. I record Ez on the interface row, where the direct wave meets the layer at each angle, and at the interface corner, then compare with the same box inside PEC walls so far away that nothing comes back within the 1000 step window. Per probe, R = 20 log10(max |test - reference| / max |reference|). Everything runs on the f64 reference kernel.

| Probe | Angle (deg) | 10 cells (dB) | 16 cells (dB) |
|---|---|---|---|
| interface | 0.0 | -89.41 | -128.47 |
| interface | 11.3 | -89.63 | -129.85 |
| interface | 19.3 | -90.02 | -130.94 |
| interface | 31.0 | -91.15 | -131.27 |
| interface | 40.4 | -91.89 | -130.86 |
| interface | 50.2 | -92.52 | -132.82 |
| interface | 60.3 | -86.88 | -132.66 |
| interface | 70.0 | -89.23 | -127.25 |
| interface | 80.0 | -85.51 | -125.37 |
| corner | 51.3 | -86.47 | -132.86 |

| Check | Layer | Gate | Worst, 0 to 70 deg | Result |
|---|---|---|---|---|
| A9 | 10 cells: m = 3, sigma_scale = 1.25, kappa_max = 2, alpha_max = 0.15 | at most -60 dB | -86.47 dB | pass |
| A10 | 16 cells: m = 4, sigma_scale = 1, kappa_max = 2, alpha_max = 0.1 | -80 dB target | -127.25 dB | pass |

The layers are `PmlSpec::recommended`, tuned with `fdtd pml-sweep` against the worst reflection over every probe above, grazing 80 degrees included.

## Cavity convergence (A7)

![Cavity resonance error vs resolution](cavity_convergence.png)

`validation/gen/plot_cavity_convergence.py` makes this plot. It re-measures the resonances in NumPy on its own, as a cross-check of the Rust test.
