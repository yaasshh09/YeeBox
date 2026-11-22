"""A21 done plot: the small Drude cylinder resonance in TEz against the
quasi-static wp / sqrt(eps_inf + 1), at radius 5, 10 and 20 cells.

Builds the same scene as `fdtd_core::cases::dispersive::drude_cylinder`
(eps_inf 1, wp so that k a = 0.1 at resonance, gamma = 0.1 of it, a +x TF/SF
Ricker pulse peaking there, Hz probe on the backscatter axis in the
scattered-field region), runs each through `fdtd run --scene`, and measures
the resonance on its own in NumPy: the probe spectrum over the Ricker's
analytic spectrum, peak found on a grid and refined with a bounded search.
At fixed k a the error left as the radius grows is the retardation of a
cylinder that is not quite small; the trend down to it is the staircasing
of the circle.

Usage (from the repo root):
    python validation/gen/plot_drude_cylinder.py [out.png]
Default output: validation/report/drude_cylinder.png
"""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "out" / "drude_cylinder"
S = 0.5
RADII = [5, 10, 20]


def scene(a: int) -> tuple[dict, float, int]:
    w_res = 0.1 / a
    wp, gamma = w_res * np.sqrt(2), 0.1 * w_res
    d, gap, half = 16, 4, a + 6
    c = d + gap + 4 + half
    n = 2 * c + 1
    f_p = w_res / (2 * np.pi)
    steps = int((4 / f_p + 20 / gamma) / S)
    s = {
        "version": 1,
        "mode": "te",
        "grid": {"nx": n, "ny": n, "dx_phys_m": 1e-9, "courant": S},
        "materials": [
            {"kind": "dielectric", "eps_inf": 1, "sigma": 0},
            {"kind": "drude", "eps_inf": 1, "wp": wp, "gamma": gamma},
        ],
        "geometry": [{"shape": "circle", "cx": c, "cy": c, "r": a, "material": 1}],
        "monitors": [{"type": "point", "i": c - half - 3, "j": c}],
        "pml": {"thickness": d, "m": 4, "sigma_scale": 1, "kappa_max": 2, "alpha_max": 0.1},
        "plane_wave": {
            "x0": c - half, "y0": c - half, "x1": c + half, "y1": c + half,
            "direction": "+x",
            "waveform": {"type": "ricker", "t0": 2 / f_p, "f_peak": f_p},
            "amplitude": 1,
        },
    }
    return s, w_res, steps


def measure(a: int) -> tuple[float, float, np.ndarray, np.ndarray]:
    s, w_res, steps = scene(a)
    path, out = WORK / f"cylinder_a{a}.json", WORK / f"cylinder_a{a}.npz"
    path.write_text(json.dumps(s, indent=2))
    subprocess.run(
        ["cargo", "run", "-q", "--release", "-p", "fdtd-cli", "--", "run",
         "--scene", str(path), "--steps", str(steps), "--out", str(out)],
        cwd=ROOT, check=True,
    )
    rec = np.load(out)["probes"][0]
    n = np.arange(len(rec))

    def response(w: float) -> float:
        x = w / w_res
        return abs(np.sum(rec * np.exp(-1j * w * n * S))) * S / (x * x * np.exp(-x * x))

    grid = w_res * np.linspace(0.5, 1.5, 201)
    resp = np.array([response(w) for w in grid])
    k = int(np.argmax(resp))
    res = minimize_scalar(lambda w: -response(w), bounds=(grid[k - 1], grid[k + 1]),
                          method="bounded", options={"xatol": 1e-14})
    theory = s["materials"][1]["wp"] / np.sqrt(2)
    return float(res.x), float(theory), grid / theory, resp / resp.max()


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "validation/report/drude_cylinder.png"
    WORK.mkdir(parents=True, exist_ok=True)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2))
    errs = []
    for a in RADII:
        w, theory, x, resp = measure(a)
        errs.append(w / theory - 1)
        print(f"a = {a:2d} cells: resonance {w:.6e}, theory {theory:.6e}, rel err {errs[-1]:+.4e}")
        left.plot(x, resp, label=f"radius {a} cells")
    left.axvline(1.0, color="gray", ls=":", label="wp / sqrt(eps_inf + 1)")
    left.set_xlabel("w / (wp / sqrt(eps_inf + 1))")
    left.set_ylabel("scattered Hz over the incident spectrum (normalized)")
    left.set_title("Drude cylinder, TEz, k a = 0.1, Q 10")
    left.legend(fontsize=8)
    left.grid(alpha=0.3)
    right.loglog(RADII, np.abs(errs), "o-", ms=7, label="measured")
    right.axhline(0.05, color="tab:red", ls="--", lw=1, label="A21 gate (5 percent)")
    right.set_xlabel("cylinder radius in cells")
    right.set_ylabel("|relative error| of the resonance")
    right.set_title("staircasing trend at fixed k a")
    right.grid(True, which="both", alpha=0.3)
    right.legend(fontsize=8)
    fig.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=120)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
