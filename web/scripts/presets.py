"""Writes the preset scenes: a full size for the GPU (web/src/presets) and a
compact one for the CPU fallback (validation/scenes, so Tier 4 runs it).
Geometry is in cells; frequencies are normalized (f = 1 / wavelength in
cells, vacuum). Run from the repo root: python web/scripts/presets.py

Display gains are about 1 / (99th percentile of |Ez|) after the wave has
crossed the grid, measured on the GPU kernel, so most of the field sits in
the color range and only the hot spots saturate.
"""
import json
import math
from pathlib import Path

VAC = {"kind": "dielectric", "eps_inf": 1, "sigma": 0}
GLASS = {"kind": "dielectric", "eps_inf": 2.25, "sigma": 0}
PEC = {"kind": "pec"}


def pml(d):
    # PmlSpec::recommended(16) settings for 16 cells, the 10-cell preset otherwise.
    if d >= 16:
        return {"thickness": d, "m": 4, "sigma_scale": 1.0, "kappa_max": 2, "alpha_max": 0.1}
    return {"thickness": d, "m": 3, "sigma_scale": 1.25, "kappa_max": 2, "alpha_max": 0.15}


def cw(f0, amplitude, phase=0.0):
    return {"type": "ramped_cw", "f0": f0, "ramp_periods": 3, "phase": phase}, amplitude


def base(nx, ny, dx, d):
    return {
        "version": 1,
        "grid": {"nx": nx, "ny": ny, "dx_phys_m": dx, "courant": 0.5},
        "materials": [VAC, GLASS, PEC],
        "pml": pml(d),
    }


def double_slit(nx, ny, d, wl, x_src, x_wall, thick, width, sep, gain):
    s = base(nx, ny, 500e-9 / wl, d)  # 500 nm light
    lo, hi = d, ny - 1 - d
    s["sources"] = [
        {"i": x_src, "j": j, "waveform": cw(1 / wl, 1)[0], "amplitude": 1}
        for j in range(lo, hi + 1)
    ]
    center = (ny - 1) / 2
    walls, run = [], None
    for j in range(lo, hi + 2):
        closed = j <= hi and all(abs(j - c) >= width / 2 for c in (center - sep / 2, center + sep / 2))
        if closed and run is None:
            run = j
        elif not closed and run is not None:
            walls.append({"shape": "rect", "x0": x_wall, "y0": run, "x1": x_wall + thick - 1, "y1": j - 1, "material": 2})
            run = None
    s["geometry"] = walls
    s["view"] = {"field": "ez", "gain": gain, "auto_gain": False}
    return s


def waveguide(nx, ny, d, y_lo, y_hi, x_start, x_src, wl, gain):
    s = base(nx, ny, 0.06 / wl, d)  # a 60 mm wave, 5 GHz
    s["geometry"] = [
        {"shape": "rect", "x0": x_start, "y0": y_lo, "x1": nx - 1, "y1": y_lo, "material": 2},
        {"shape": "rect", "x0": x_start, "y0": y_hi, "x1": nx - 1, "y1": y_hi, "material": 2},
    ]
    a = y_hi - y_lo
    s["sources"] = [
        {"i": x_src, "j": j, "waveform": cw(1 / wl, 1)[0], "amplitude": round(math.sin(math.pi * (j - y_lo) / a), 12)}
        for j in range(y_lo + 1, y_hi)
    ]
    s["view"] = {"field": "ez", "gain": gain, "auto_gain": False}
    return s


def lens(nx, ny, d, margin, cx, half, radius, wl, gain):
    s = base(nx, ny, 500e-9 / wl, d)  # 500 nm light
    # Biconvex: two arcs of radius R whose chord is the aperture 2 * half.
    off = math.sqrt(radius * radius - half * half)
    a0 = math.asin(half / radius)
    pts = []
    for k in range(25):
        t = -a0 + 2 * a0 * k / 24
        pts.append([round(cx - off + radius * math.cos(t), 4), round(radius * math.sin(t) + (ny - 1) / 2, 4)])
    for k in range(25):
        t = math.pi - a0 + 2 * a0 * k / 24
        pts.append([round(cx + off + radius * math.cos(t), 4), round(radius * math.sin(t) + (ny - 1) / 2, 4)])
    s["geometry"] = [{"shape": "polygon", "points": pts, "material": 1}]
    s["plane_wave"] = {
        "x0": margin, "y0": margin, "x1": nx - 1 - margin, "y1": ny - 1 - margin,
        "direction": "+x", "waveform": cw(1 / wl, 1)[0], "amplitude": 1,
    }
    s["view"] = {"field": "ez", "gain": gain, "auto_gain": False}
    return s


C0 = 299_792_458.0


def silver(dx):
    # material::silver in fdtd-core (Vial et al. 2005 Drude fit), the same
    # numbers the palette uses, so painted silver is the same metal.
    k = dx / C0
    return {"kind": "drude", "eps_inf": 3.718, "wp": 1.3659e16 * k, "gamma": 2.7909e13 * k}


def plasmon(n, d, dx, radius, gain):
    # TEz: a silver cylinder under a CW plane wave at 360 nm, near its
    # surface plasmon (wp / sqrt(eps_inf + 1) is about 300 nm for a small
    # cylinder; a 24 nm one sits redder).
    wl = 360e-9 / dx
    s = {
        "version": 1,
        "mode": "te",
        "grid": {"nx": n, "ny": n, "dx_phys_m": dx, "courant": 0.5},
        "materials": [VAC, GLASS, PEC, silver(dx)],
        "pml": pml(d),
        "geometry": [{"shape": "circle", "cx": (n - 1) / 2, "cy": (n - 1) / 2, "r": radius, "material": 3}],
    }
    m = d + 6
    s["plane_wave"] = {
        "x0": m, "y0": m, "x1": n - 1 - m, "y1": n - 1 - m,
        "direction": "+x", "waveform": cw(1 / wl, 1)[0], "amplitude": 1,
    }
    s["view"] = {"field": "ez", "gain": gain, "auto_gain": False}
    return s


def mirror(nx, ny, d, wl, x_src, x_wall, thick, gain):
    # TMz: silver at 25 nm cells (skin depth under a cell, the status strip
    # says so), a point source in front, standing waves between them. The
    # wall stops two nodes short of the layer, as dispersive cells must.
    dx = 500e-9 / wl
    s = base(nx, ny, dx, d)
    s["materials"] = [VAC, GLASS, PEC, silver(dx)]
    s["geometry"] = [{"shape": "rect", "x0": x_wall, "y0": d + 2, "x1": x_wall + thick - 1, "y1": ny - 3 - d, "material": 3}]
    s["sources"] = [{"i": x_src, "j": (ny - 1) // 2, "waveform": cw(1 / wl, 1)[0], "amplitude": 1}]
    s["view"] = {"field": "ez", "gain": gain, "auto_gain": False}
    return s


def absorber(nx, ny, d, wl, x0, cols, rows, pitch, radius, gain):
    # TMz: a lattice of Lorentz rods resonant at the drive (w0 = 2 pi f0)
    # under a plane wave; behind the lattice, a shadow.
    s = base(nx, ny, 500e-9 / wl, d)
    w0 = 2 * math.pi / wl
    s["materials"] = [VAC, GLASS, PEC, {"kind": "lorentz", "eps_inf": 1.0, "d_eps": 1.5, "w0": w0, "delta": 0.1 * w0}]
    y0 = (ny - 1) / 2 - (rows - 1) * pitch / 2
    s["geometry"] = [
        {"shape": "circle", "cx": x0 + c * pitch, "cy": round(y0 + r * pitch, 4), "r": radius, "material": 3}
        for c in range(cols) for r in range(rows)
    ]
    m = d + 6
    s["plane_wave"] = {
        "x0": m, "y0": m, "x1": nx - 1 - m, "y1": ny - 1 - m,
        "direction": "+x", "waveform": cw(1 / wl, 1)[0], "amplitude": 1,
    }
    s["view"] = {"field": "ez", "gain": gain, "auto_gain": False}
    return s


PRESETS = {
    "plasmon": (
        plasmon(400, 16, 2e-9, 12, 0.95),
        plasmon(160, 10, 5e-9, 5, 0.95),
    ),
    "mirror": (
        mirror(600, 400, 16, 20, 200, 420, 12, 27),
        mirror(200, 140, 10, 10, 70, 140, 6, 27),
    ),
    "absorber": (
        absorber(600, 400, 16, 20, 220, 3, 12, 22, 5, 0.75),
        absorber(200, 140, 10, 10, 80, 3, 8, 11, 2.5, 0.75),
    ),
    "double_slit": (
        double_slit(800, 500, 16, 20, 60, 260, 4, 16, 80, 0.45),
        double_slit(200, 128, 10, 10, 20, 70, 2, 5, 24, 0.45),
    ),
    "waveguide": (
        waveguide(700, 300, 16, 120, 180, 100, 150, 60, 0.9),
        waveguide(200, 90, 10, 30, 60, 30, 50, 30, 0.9),
    ),
    "lens": (
        lens(700, 450, 16, 30, 200, 120, 200, 20, 0.65),
        lens(200, 140, 10, 16, 60, 40, 70, 12, 0.65),
    ),
}

root = Path(__file__).resolve().parents[2]
for name, (full, compact) in PRESETS.items():
    (root / "web/src/presets" / f"{name}.json").write_text(json.dumps(full, indent=1) + "\n", newline="\n")
    (root / "validation/scenes" / f"preset_{name}_compact.json").write_text(json.dumps(compact, indent=1) + "\n", newline="\n")
    print(name, full["grid"]["nx"], full["grid"]["ny"], "/", compact["grid"]["nx"], compact["grid"]["ny"])
