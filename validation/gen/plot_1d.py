"""Space-time plot of a 1D run written by `fdtd run --out file.npz`.

Expects keys: ez (steps x nx, f64; row n is Ez at time n, column i is node i),
courant (scalar), dt (scalar, normalized units).

Usage: python plot_1d.py out/pulse.npz [out.png]
"""

import sys

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    data = np.load(sys.argv[1])
    ez = data["ez"]
    lim = np.max(np.abs(ez)) or 1.0
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(ez, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xlabel("node i")
    ax.set_ylabel("time step n")
    ax.set_title(f"Ez, 1D, S = {float(data['courant']):g}")
    fig.colorbar(im, ax=ax, label="Ez (normalized)")
    fig.tight_layout()
    if len(sys.argv) > 2:
        fig.savefig(sys.argv[2], dpi=120)
    else:
        plt.show()


if __name__ == "__main__":
    main()
