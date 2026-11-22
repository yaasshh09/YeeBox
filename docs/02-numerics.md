# Numerics

The most important file in the repo. Read fully before writing solver code.

## Stability: the Courant limit

For a square 2D grid the scheme is stable only if

    S = c_max * dt / dx <= 1 / sqrt(2)  (about 0.7071)

In 1D the limit is 1, in 3D it is `1/sqrt(3)`. `c_max` is the fastest phase
velocity anywhere in the domain, which is vacuum unless a material has
relative permittivity below 1. Reject material definitions with
`eps_inf < 1`.

Default `S = 0.5`. It leaves margin for the dispersive-media updates, whose
stability bound is slightly tighter than the vacuum bound. Derive the 2D limit
from the von Neumann analysis in a doc comment and enforce it in the
constructor: a solver built with `S` above the limit must return an error, not
run and explode.

Test: at `S = 0.70` a random initial field stays bounded for 20000 steps. At
`S = 0.72` it grows without bound. Both are asserted.

## Numerical dispersion

The grid has its own dispersion relation, different from the continuum:

    [ sin(w*dt/2) / (c*dt) ]^2 = [ sin(kx*dx/2) / dx ]^2 + [ sin(ky*dy/2) / dy ]^2

Consequences:

- Phase velocity depends on direction and on cells per wavelength. Along the
  axes it is slowest. Along the diagonal at `S = 1/sqrt(2)` it is exact.
- At 20 cells per wavelength the axis phase velocity error is a few tenths of
  a percent. Compute the exact value from the relation above; do not quote a
  number from memory.
- In 1D with `S = 1` the scheme is exact (the magic time step). This is the
  first test of the project.

Write `numerical_wavenumber(w, direction, S)` that solves the relation with
Newton iteration. Tests compare measured wavelengths against this function to
0.01 percent, and against the continuum value only to confirm O(dx^2)
convergence. Comparing a simulation directly against continuum physics at
tight tolerance is a category error and will fail for the wrong reason.

Guidance surfaced in the UI: below 10 cells per wavelength in the densest
material, show a resolution warning.

## CPML absorbing boundary

Convolutional PML (Roden and Gedney, 2000) with the complex-frequency-shifted
stretching function. This is the filter that separates this project from hobby
FDTD codes, which use a first-order Mur condition and reflect visibly.

Stretching function along x:

    s_x = kappa_x + sigma_x / (alpha_x + j*w*eps0)

In the time domain this becomes a recursive convolution carried by auxiliary
fields `psi`. For each spatial derivative taken normal to a PML face:

    psi_new = b * psi_old + c * (d field / d normal)
    derivative_used = (d field / d normal) / kappa + psi_new

Coefficients (normalized, eps0 = 1):

    b = exp( -(sigma/kappa + alpha) * dt )
    c = sigma / (sigma*kappa + kappa^2 * alpha) * (b - 1)

Guard the `c` expression when `sigma = 0`: there `c = 0` and `b` is
irrelevant. Outside the PML set `kappa = 1`, `b = 0`, `c = 0`, which reduces
to the ordinary update. That lets one kernel handle interior and PML cells
with no branching.

Grading over a layer of thickness `d`, with `rho` the depth into the layer:

    sigma(rho) = sigma_max * (rho/d)^m
    kappa(rho) = 1 + (kappa_max - 1) * (rho/d)^m
    alpha(rho) = alpha_max * (1 - rho/d)^m_a

    sigma_opt = 0.8 * (m + 1) / (eta * dx),  eta = sqrt(mu/eps) of the
                                             adjacent medium

Starting values: `m = 3`, `m_a = 1`, `sigma_max = sigma_opt`,
`kappa_max` between 5 and 15, `alpha_max` between 0.05 and 0.25, thickness 10
to 20 cells. These are starting points. Tune against the reflection test and
record the final values and the measured reflection in `PROGRESS.md`.

Staggering matters. Coefficients for E updates are evaluated at E positions.
Coefficients for H updates are evaluated half a cell deeper. Store four 1D
coefficient profiles per axis: `b`, `c`, `1/kappa` at integer positions and at
half-integer positions. Using integer-position coefficients for both is the
most common CPML bug and costs about 20 dB.

TMz PML update, written out:

    psi_hxy = b_y_half * psi_hxy + c_y_half * (Ez[i,j+1] - Ez[i,j]) / dy
    Hx -= (dt/mu) * ( (Ez[i,j+1] - Ez[i,j]) / (kappa_y_half * dy) + psi_hxy )

    psi_hyx = b_x_half * psi_hyx + c_x_half * (Ez[i+1,j] - Ez[i,j]) / dx
    Hy += (dt/mu) * ( (Ez[i+1,j] - Ez[i,j]) / (kappa_x_half * dx) + psi_hyx )

    psi_ezx = b_x * psi_ezx + c_x * (Hy[i,j] - Hy[i-1,j]) / dx
    psi_ezy = b_y * psi_ezy + c_y * (Hx[i,j] - Hx[i,j-1]) / dy
    Ez = Ca*Ez + Cb * ( (Hy[i,j] - Hy[i-1,j]) / (kappa_x * dx) + psi_ezx
                      - (Hx[i,j] - Hx[i,j-1]) / (kappa_y * dy) - psi_ezy )

The outer wall behind the PML is PEC.

Rules for v1: no dispersive material may touch or enter the PML (the scene
validator rejects it), and plain dielectrics that extend into the PML must be
uniform along the PML normal.

### Measuring reflection

Reference-domain method. Run the same source in (a) a small domain with CPML
and (b) a domain large enough that no boundary reflection returns within the
observation window. Record Ez at probe points near the PML in both.

    R_dB = 20 * log10( max_t |Ez_small - Ez_big| / max_t |Ez_big| )

Do this for a point source placed so that waves hit the PML at angles from
normal to about 70 degrees. Report reflection versus angle. Run on the f64 CPU
reference; f32 cannot resolve -80 dB reliably.

## Dispersive media by auxiliary differential equation (ADE)

### Drude

Polarization current obeys `dJ/dt + gamma*J = eps0 * wp^2 * E`. Central
differencing about step n + 1/2:

    k    = (1 - gamma*dt/2) / (1 + gamma*dt/2)
    beta = (eps0 * wp^2 * dt/2) / (1 + gamma*dt/2)

    E_new = [ (2*eps - beta*dt) / (2*eps + beta*dt) ] * E_old
          + [ 2*dt / (2*eps + beta*dt) ] * ( curl_H - (1 + k)/2 * J_old )
    J_new = k * J_old + beta * (E_new + E_old)

with `eps = eps0 * eps_inf`. One extra stored field per E component.

### Lorentz

Polarization obeys `d2P/dt2 + 2*delta*dP/dt + w0^2*P = eps0*d_eps*w0^2*E`.
Central differencing about step n:

    a1 = (2 - w0^2 * dt^2) / (1 + delta*dt)
    a2 = -(1 - delta*dt) / (1 + delta*dt)
    a3 = eps0 * d_eps * w0^2 * dt^2 / (1 + delta*dt)

    P_new = a1 * P + a2 * P_old + a3 * E
    E_new = E + (dt/eps) * curl_H - (P_new - P) / eps

Two extra stored fields per E component (P and P_old).

Both derivations must be reproduced in doc comments so the coefficients can be
re-derived, not trusted. Validation is a slab reflection/transmission spectrum
against the analytic Fresnel result using the complex `eps(w)`.

Stability: require `w0 * dt` well below 2 and resolve the plasma wavelength
and skin depth. The scene validator warns when a Drude metal's skin depth is
under 2 cells.

## Total-field / scattered-field plane wave source

The domain is split by a closed rectangular boundary. Inside is total field,
outside is scattered field only. The incident field is supplied by a 1D
auxiliary FDTD grid running in lockstep, and consistency corrections are
applied to the field components adjacent to the boundary.

v1 supports axis-aligned incidence only. For propagation along a grid axis the
2D dispersion relation reduces exactly to the 1D relation at the same `dx` and
`dt`, so the 1D auxiliary grid matches the 2D grid perfectly and leakage into
the scattered-field region is at round-off level. Oblique incidence needs a
dispersion-matched auxiliary grid and is a stretch item; do not ship an
unmatched oblique source, it leaks at about -40 dB and looks broken.

The 1D grid needs its own absorbing end and a source far enough upstream.

Test: empty domain, pulse through the TF/SF box. Scattered-field region stays
below -120 dB relative to the incident peak on the f64 reference.

## Frequency-domain monitors

Running discrete Fourier transform, accumulated during time stepping, at a
chosen set of frequencies:

    F(w) += field(n) * exp(-j * w * n * dt) * dt

E is known at integer steps and H at half-integer steps. Multiply the H
transform by `exp(-j*w*dt/2)` so both refer to the same time origin. Skipping
this biases every flux and far-field result by a frequency-dependent phase.

Flux monitor: Poynting flux through a line segment, from co-located E and H
(average H across the two half-cell neighbors). Transmission spectrum is the
flux ratio between a run with the structure and a normalization run without
it. Reflection needs the incident field subtracted before computing flux.

Run until the fields in the domain have decayed to below 1e-6 of peak before
reading a spectrum, or the spectrum rings.

## Near-to-far-field transform (2D)

On a closed rectangular contour in the scattered-field region, record the DFT
of tangential E and H. Equivalent surface currents: `J = n x H`,
`M = -n x E`. Far field in direction `phi`, TMz:

    integrate around the contour:
      (terms in J_z and M_tangential) * exp(+j * k * r' . r_hat)

using the large-argument Hankel form for the 2D Green's function. The exact
prefactors follow Taflove and Hagness (Computational Electrodynamics, the
near-to-far-field chapter); implement from there and verify with the line
source test: a single line source inside the contour must give an isotropic
far field with the correct absolute level.

Use the numerical wavenumber for the contour phase terms where it improves
agreement, and say so in the validation report.

As built (`ntff.rs`): the contour samples come from flux lines, so E and H
are co-located on the Ez nodes (H averaged across the two half-cell
neighbors) and each enters the DFT at its true time. Trapezoid weights along
each side, corners counted once in total. The phase term uses the continuum
k: the numerical k(phi) made no difference to the line source test, because
the error left there is the co-location. The node average of H is low by
cos(k_n / 2), 1.2 percent at 20 cells per wavelength, which unbalances J and
M; the line source far field is flat to 0.08 dB at 20 cells per wavelength
and 0.005 dB at 40. An open contour (one side, `sides`) is the half-space
form used behind the double slit screen. The scattered field for a
scattering width needs no subtraction: the contour sits in the TF/SF
scattered-field region.

## GPU kernel

- Fields live in storage buffers of f32, one buffer per field, row-major,
  padded to `nx * ny`. Storage buffers are simpler and more portable than
  storage textures for read-write compute.
- Two compute passes per step: `update_h` then `update_e`. In-place, no
  ping-pong.
- PML coefficients as six 1D buffers per axis. Interior values make the
  kernel reduce to the plain update, so there is one code path.
- Material id buffer (u32) plus a small uniform or storage array of per
  material coefficients, precomputed on the CPU for the current `dt`.
- Dispersive auxiliary fields are full-size buffers allocated only when the
  scene contains a dispersive material.
- Workgroup size 8 x 8. Dispatch `ceil(nx/8) x ceil(ny/8)` and bounds-check
  inside the shader.
- N steps per frame are encoded as N pairs of passes in one command buffer.
  Source time-dependence is passed per step through a small uniform array or
  by precomputing source samples into a buffer indexed by step.
- Painting writes a sub-rectangle of the material buffer with
  `queue.write_buffer`.
- Rendering: a fullscreen triangle whose fragment shader reads the field
  buffer and applies a colormap, with the material mask composited on top.
- Readback (probes, monitors) uses a small staging buffer and async map.
  Never read back the whole field every frame.

Monitors that need DFT accumulation run on the GPU as an extra compute pass
over the monitor cells only.

## Precision

The CPU reference is f64 and is where all tight-tolerance physics validation
happens. The GPU is f32. GPU acceptance is agreement with the CPU reference on
identical scenes, not independent agreement with physics. Expect relative
differences near 1e-5 growing slowly with step count.

## Performance model

Cost per step is about `nx * ny` cell updates times a small constant. Report
throughput in million cell updates per second. Record numbers for at least one
integrated GPU and one discrete GPU in `PROGRESS.md`, with the grid size and
steps per frame used.
