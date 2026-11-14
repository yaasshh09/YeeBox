"""Golden data for A13 (dielectric slab R, T) and A14 (lossy attenuation).

Normal incidence on a slab of relative permittivity eps and thickness L
cells, vacuum on both sides. Two references per frequency:

Continuum (the gate): the Airy / Fabry-Perot formulas with
n = sqrt(eps - j sigma / w), exp(+j w t) convention,
    r = (1 - n) / (1 + n),  d = n w L
    R = |r (1 - e^{-2jd}) / (1 - r^2 e^{-2jd})|^2
    T = |(1 - r^2) e^{-jd} / (1 - r^2 e^{-2jd})|^2

Discrete (exact for the Yee grid, recorded, not gated): in each uniform
region the grid carries E_i = A e^{-jki} + B e^{jki} with the region's
discrete k. Where node m is one medium and node m + 1 the other, the H
update on the edge between them does not depend on the medium, and each
side's E update reads that edge, so the two representations agree at both
nodes m and m + 1. That gives one 2 x 2 transfer per interface. eps on Ez
nodes 1..L makes the slab L cells thick between faces at 1/2 and L + 1/2.

The discrete k with the semi-implicit conductivity: the E update
Ez^{n+1} = Ca Ez^n + Cb (dH) gives G E = dH with
G = (e^{jwS/2} - Ca e^{-jwS/2}) / Cb, and with the H update
sin^2(k/2) = -j G sin(wS/2) / (2S). Lossless, that is
sin(k/2) = sqrt(eps) sin(wS/2) / S. The branch has Re k > 0, Im k <= 0.

A13 is gated at the base resolution (L = 20, band down to 20 cells per
wavelength inside the slab); L = 40 and 80 cover the same physical slab with
frequencies scaled by 20 / L, for the convergence order.

A14: a uniform lossy medium. The attenuation constant alpha = -Im k per
cell, continuum (gate) and discrete (recorded).

Normalized units: dx = 1, c0 = 1, dt = S, sigma normalized as in
`Material::Dielectric` (eps0 = 1). Frequencies are cycles per unit time.

Usage (from the repo root):
    python validation/gen/slab_airy.py
Output: validation/golden/slab_airy.csv, validation/golden/lossy_medium.csv
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_SLAB = ROOT / "validation" / "golden" / "slab_airy.csv"
OUT_LOSSY = ROOT / "validation" / "golden" / "lossy_medium.csv"

S = 0.5
SLAB_EPS = 4.0
BASE_L = 20
RESOLUTIONS = (20, 40, 80)
# Base band: f_max puts 20 cells per wavelength in the slab; 19 points from
# f_max / 10 to f_max. Scaled by BASE_L / L for the finer grids.
F_MAX_BASE = 1.0 / (20.0 * np.sqrt(SLAB_EPS))
F_STEPS = 19

LOSSY_EPS = 2.25
LOSSY_SIGMA = 0.05
# Band picked from the discrete model before any FDTD run, by the same rule
# as A13 (predicted discretization error at most half the 1 percent gate):
# alpha_d is 0.40 percent off alpha_c at f = 0.02 (33 cells per wavelength in
# the medium) but 0.9 percent at f = 0.03, so the band stops at 0.02.
LOSSY_FREQS = (0.008, 0.011, 0.014, 0.017, 0.02)


def k_disc(w, eps, sigma=0.0):
    """Discrete Yee wavenumber at angular frequency w, Re k > 0, Im k <= 0."""
    l = sigma * S / (2 * eps)
    ca, cb = (1 - l) / (1 + l), (S / eps) / (1 + l)
    g = (np.exp(1j * w * S / 2) - ca * np.exp(-1j * w * S / 2)) / cb
    s2 = -1j * g * np.sin(w * S / 2) / (2 * S)
    k = 2 * np.arcsin(np.sqrt(complex(s2)))
    if k.real < 0:
        k = -k
    if k.imag > 0:
        k = np.conj(k)
    return complex(k)


def k_cont(w, eps, sigma=0.0):
    n = np.sqrt(complex(eps, -sigma / w))
    return complex(n * w)


def slab_disc(w, eps, sigma, length):
    """Exact discrete R, T for eps on nodes 1..length, vacuum elsewhere."""
    k1, k2 = k_disc(w, 1.0), k_disc(w, eps, sigma)

    def m(k, node):
        return np.array(
            [
                [np.exp(-1j * k * node), np.exp(1j * k * node)],
                [np.exp(-1j * k * (node + 1)), np.exp(1j * k * (node + 1))],
            ]
        )

    t = np.linalg.solve(m(k1, length), m(k2, length)) @ np.linalg.solve(m(k2, 0), m(k1, 0))
    b1 = -t[1, 0] / t[1, 1]
    a3 = t[0, 0] + t[0, 1] * b1
    return float(abs(b1) ** 2), float(abs(a3) ** 2)


def slab_cont(w, eps, sigma, length):
    n = np.sqrt(complex(eps, -sigma / w))
    r = (1 - n) / (1 + n)
    e2 = np.exp(-2j * n * w * length)
    den = 1 - r**2 * e2
    big_r = abs(r * (1 - e2) / den) ** 2
    big_t = abs((1 - r**2) * np.exp(-1j * n * w * length) / den) ** 2
    return float(big_r), float(big_t)


def slab_rows():
    for length in RESOLUTIONS:
        f_max = F_MAX_BASE * BASE_L / length
        for step in range(1, F_STEPS + 1):
            f = f_max * (step + 1) / (F_STEPS + 1)
            w = 2 * np.pi * f
            rc, tc = slab_cont(w, SLAB_EPS, 0.0, length)
            rd, td = slab_disc(w, SLAB_EPS, 0.0, length)
            vals = [float(f), rc, tc, rd, td]
            yield ",".join([str(length)] + [repr(v) for v in vals])


def lossy_rows():
    for f in LOSSY_FREQS:
        w = 2 * np.pi * f
        kc, kd = k_cont(w, LOSSY_EPS, LOSSY_SIGMA), k_disc(w, LOSSY_EPS, LOSSY_SIGMA)
        vals = [float(f), -kc.imag, -kd.imag, kc.real, kd.real]
        yield ",".join(repr(float(v)) for v in vals)


def main():
    slab = [
        "# A13 golden data, generated by validation/gen/slab_airy.py",
        f"# S = {S!r}, eps = {SLAB_EPS!r}, base L = {BASE_L}, slab on nodes 1..L",
        "# R_c, T_c: continuum Airy (the gate). R_d, T_d: exact discrete Yee slab.",
        "L,f,R_c,T_c,R_d,T_d",
    ]
    slab += slab_rows()
    lossy = [
        "# A14 golden data, generated by validation/gen/slab_airy.py",
        f"# S = {S!r}, eps = {LOSSY_EPS!r}, sigma = {LOSSY_SIGMA!r}",
        "# alpha = -Im k per cell, beta = Re k: _c continuum (gate), _d discrete Yee",
        "f,alpha_c,alpha_d,beta_c,beta_d",
    ]
    lossy += lossy_rows()
    for path, lines in ((OUT_SLAB, slab), (OUT_LOSSY, lossy)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {path.relative_to(ROOT)} ({len(lines) - 4} rows)")

    # The error budget the A13 band was picked from (printed, not stored).
    for length in RESOLUTIONS:
        f_max = F_MAX_BASE * BASE_L / length
        fs = [f_max * (s + 1) / (F_STEPS + 1) for s in range(1, F_STEPS + 1)]
        worst = max(
            abs(slab_disc(2 * np.pi * f, SLAB_EPS, 0, length)[0] - slab_cont(2 * np.pi * f, SLAB_EPS, 0, length)[0])
            for f in fs
        )
        print(f"L = {length}: predicted max |R_d - R_c| = {worst:.2e}")


if __name__ == "__main__":
    main()
