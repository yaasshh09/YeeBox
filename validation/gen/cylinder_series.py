"""Golden data for A19 and A20: plane wave scattering by an infinite circular
cylinder, the 2D analogue of Mie theory.

A plane wave travelling along +x, exp(+j w t) convention, so the incident
field is exp(-j k x) = sum_n j^-n J_n(k rho) exp(j n phi). The scattered
field is sum_n j^-n a_n H_n^(2)(k rho) exp(j n phi), and with
H_n^(2)(x) ~ sqrt(2j / (pi x)) j^n exp(-j x) the scattering width is

    sigma_2D(phi) = (4 / k) |sum_n a_n exp(j n phi)|^2

with phi from the propagation direction (phi = 0 is forward scattering).
TMz scatters Ez, TEz scatters Hz. The coefficients, for a cylinder of radius
a, x = k a, and m = sqrt(eps) (mu = 1 everywhere):

    PEC, TMz:        a_n = -J_n(x) / H_n(x)                  (Ez = 0 on the wall)
    PEC, TEz:        a_n = -J_n'(x) / H_n'(x)                (E_phi = 0)
    dielectric, TMz: a_n = -[m J_n(x) J_n'(mx) - J_n'(x) J_n(mx)]
                          / [m H_n(x) J_n'(mx) - H_n'(x) J_n(mx)]
    dielectric, TEz: a_n = -[J_n(x) J_n'(mx) - m J_n'(x) J_n(mx)]
                          / [H_n(x) J_n'(mx) - m H_n'(x) J_n(mx)]

The dielectric ones come from continuity of Ez and H_phi ~ dEz/drho (TMz),
and of Hz and E_phi ~ (1 / eps) dHz/drho (TEz), at rho = a, as in Balanis
(Advanced Engineering Electromagnetics, cylinder scattering chapter).

Self-checks, all asserted before anything is written:
- PEC limit: the dielectric TMz coefficients approach the PEC ones as eps
  grows (eps = 1 - 1e10 j, a very good conductor).
- Rayleigh limit: at k a = 1e-3 the dielectric TMz width is isotropic and
  equals pi^2 k^3 a^4 (eps - 1)^2 / 4 (the next term in a_0 is of order
  (k a)^2 ln(k a) |eps - 1|, 1.5e-3 at k a = 0.01).
- Optical theorem, mode by mode: |a_n|^2 = -Re a_n for every lossless
  cylinder (no power lost), so the scattered power equals the extinction.
- Series truncation: the last kept term is below 1e-15 of the largest.

Output: one row per case, k a and angle (0 to 359 degrees, 1 degree steps,
the angles of a 360-angle far-field monitor), sigma / lambda in dB.
Normalized: sigma / lambda does not depend on units.

Usage (from the repo root):
    python validation/gen/cylinder_series.py
Output: validation/golden/cylinder_series.csv
"""

from pathlib import Path

import numpy as np
from scipy.special import h2vp, hankel2, jv, jve, jvp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation" / "golden" / "cylinder_series.csv"

KA = [1.0, 2.0, 5.0]
EPS = 4.0
CASES = ["pec_tm", "pec_te", "dielectric_tm", "dielectric_te"]


def terms(x):
    """Highest order kept: Wiscombe's rule plus margin."""
    return int(np.ceil(x + 4 * x ** (1 / 3) + 10))


def coefficients(case, x, eps=EPS, top=None):
    """a_n for n = 0 .. top (a_-n = a_n), top from `terms` by default."""
    if top is None:
        top = terms(abs(x) * abs(np.sqrt(eps)))
    n = np.arange(top + 1)
    j, jd, h, hd = jv(n, x), jvp(n, x), hankel2(n, x), h2vp(n, x)
    if case == "pec_tm":
        return -j / h
    if case == "pec_te":
        return -jd / hd
    m = np.sqrt(complex(eps))
    # r = J_n'(mx) / J_n(mx) from the exponentially scaled jve, whose scale
    # cancels, so a lossy mx with a huge imaginary part does not overflow.
    z = m * x
    r = (jve(n - 1, z) - jve(n + 1, z)) / (2 * jve(n, z))
    if case == "dielectric_tm":
        return -(m * j * r - jd) / (m * h * r - hd)
    if case == "dielectric_te":
        return -(j * r - m * jd) / (h * r - m * hd)
    raise ValueError(case)


def sigma(case, ka, phi, eps=EPS, k=1.0):
    """sigma_2D at angles phi (radians) for a cylinder with k a = ka, at
    wavenumber k. Returns sigma in the units of 1 / k."""
    a = coefficients(case, ka, eps)
    n = np.arange(len(a))
    # a_0 + 2 sum_{n >= 1} a_n cos(n phi), since a_-n = a_n.
    s = a[0] + 2 * np.sum(a[1:, None] * np.cos(n[1:, None] * phi[None, :]), axis=0)
    return 4 / k * np.abs(s) ** 2


def sigma_over_lambda_db(case, ka, phi, eps=EPS):
    return 10 * np.log10(sigma(case, ka, phi, eps) / (2 * np.pi))


def self_checks():
    for ka in KA:
        pec = coefficients("pec_tm", ka)
        near = coefficients("dielectric_tm", ka, eps=1 - 1e10j, top=len(pec) - 1)
        err = np.max(np.abs(near - pec)) / np.max(np.abs(pec))
        assert err < 1e-4, f"PEC limit at ka {ka}: {err}"
        for case in CASES:
            a = coefficients(case, ka)
            theorem = np.max(np.abs(np.abs(a) ** 2 + a.real))
            assert theorem < 1e-12, f"optical theorem {case} ka {ka}: {theorem}"
            tail = np.abs(a[-1]) / np.max(np.abs(a))
            assert tail < 1e-15, f"truncation {case} ka {ka}: {tail}"
    ka, k = 1e-3, 1.0
    phi = np.linspace(0, 2 * np.pi, 13)
    got = sigma("dielectric_tm", ka, phi, k=k)
    want = np.pi**2 * k**3 * ka**4 * (EPS - 1) ** 2 / 4
    err = np.max(np.abs(got / want - 1))
    assert err < 1e-3, f"Rayleigh limit: {err}"
    print(f"self-checks pass (Rayleigh {err:.2e})")


def main():
    self_checks()
    phi_deg = np.arange(360)
    phi = np.deg2rad(phi_deg)
    lines = [
        "# A19, A20: sigma_2D / lambda of a circular cylinder, plane wave along +x, "
        f"exp(+jwt), eps {EPS} for the dielectric; phi from +x (forward); "
        "generated by validation/gen/cylinder_series.py",
        "case,ka,phi_deg,sigma_over_lambda_db",
    ]
    for case in CASES:
        for ka in KA:
            for p, v in zip(phi_deg, sigma_over_lambda_db(case, ka, phi)):
                lines.append(f"{case},{ka!r},{int(p)},{float(v)!r}")
    OUT.write_text("\n".join(lines) + "\n", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(lines) - 2} rows)")


if __name__ == "__main__":
    main()
