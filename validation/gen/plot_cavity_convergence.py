"""Cavity resonance error vs resolution, log-log, with the fitted order (Phase 1 done check).

Builds the 4L x 3L cell PEC cavity used by test A7 at L = 4, 8, 16, 32, runs each
through `fdtd run --scene`, and measures the (1, 1) resonance from the center probe
record on its own in NumPy: drop the samples while the Ricker pulse is still on,
apply a Blackman-Harris window, find the spectral peak, and refine it with a
bounded scalar search. It then compares the result with the continuum value
pi sqrt(1/a^2 + 1/b^2) and with the discrete prediction.

Usage (from the repo root):
    python validation/gen/plot_cavity_convergence.py [out.png]
Default output: validation/report/cavity_convergence.png
"""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal.windows import blackmanharris

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "out" / "cavity_convergence"
S = 0.5
LS = [4, 8, 16, 32]


def continuum(a: int, b: int) -> float:
    return np.pi * np.hypot(1 / a, 1 / b)


def discrete(a: int, b: int) -> float:
    r = np.hypot(np.sin(np.pi / (2 * a)), np.sin(np.pi / (2 * b)))
    return 2 / S * np.arcsin(S * r)


def scene(a: int, b: int, t0: float, f_peak: float) -> dict:
    c = {"i": a // 2, "j": b // 2}
    return {
        "version": 1,
        "grid": {"nx": a + 1, "ny": b + 1, "dx_phys_m": 1e-3, "courant": S},
        "materials": [{"kind": "dielectric", "eps_inf": 1, "sigma": 0}],
        "sources": [
            {**c, "waveform": {"type": "ricker", "t0": t0, "f_peak": f_peak}, "amplitude": 1}
        ],
        "monitors": [{"type": "point", **c}],
    }


def measure(l: int) -> float:
    a, b = 4 * l, 3 * l
    w_c = continuum(a, b)
    f_peak = w_c / (2 * np.pi)
    t0 = 6 / (np.pi * f_peak)
    skip = int(np.ceil(2 * t0 / S))
    steps = skip + int(60 * a / S)
    path = WORK / f"cavity_L{l}.json"
    path.write_text(json.dumps(scene(a, b, t0, f_peak), indent=2))
    out = WORK / f"cavity_L{l}.npz"
    subprocess.run(
        ["cargo", "run", "-q", "--release", "-p", "fdtd-cli", "--", "run",
         "--scene", str(path), "--steps", str(steps), "--out", str(out)],
        cwd=ROOT, check=True,
    )
    rec = np.load(out)["probes"][0, skip:]
    win = blackmanharris(len(rec), sym=True)
    n = np.arange(len(rec))

    def mag(w: float) -> float:
        return -abs(np.sum(rec * win * np.exp(-1j * w * n * S)))

    bin_w = 2 * np.pi / (len(rec) * S)
    grid = np.arange(0.5 * w_c, 1.5 * w_c, bin_w / 4)
    k = int(np.argmin([mag(w) for w in grid]))
    res = minimize_scalar(mag, bounds=(grid[k - 1], grid[k + 1]), method="bounded",
                          options={"xatol": 1e-14})
    return float(res.x)


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "validation/report/cavity_convergence.png"
    WORK.mkdir(parents=True, exist_ok=True)
    dx, err, err_d = [], [], []
    for l in LS:
        a, b = 4 * l, 3 * l
        w = measure(l)
        dx.append(1 / l)
        err.append(abs(w / continuum(a, b) - 1))
        err_d.append(abs(discrete(a, b) / continuum(a, b) - 1))
        print(f"L = {l:3d}: measured {w:.12f}, rel err vs continuum {err[-1]:.4e}, "
              f"vs discrete {abs(w / discrete(a, b) - 1):.1e}")
    order = np.polyfit(np.log(dx), np.log(err), 1)[0]
    print(f"fitted order {order:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(dx, err, "o", ms=8, label="measured (FDTD, center probe)")
    ax.loglog(dx, err_d, "-", lw=1, label="discrete dispersion prediction")
    ref = err[0] * (np.array(dx) / dx[0]) ** 2
    ax.loglog(dx, ref, ":", color="gray", label="slope 2 reference")
    ax.set_xlabel("cell size dx (cavity is 4 x 3 units)")
    ax.set_ylabel("relative error of mode (1, 1) vs continuum")
    ax.set_title(f"PEC cavity, TMz, S = {S}: fitted order {order:.3f}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=120)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
