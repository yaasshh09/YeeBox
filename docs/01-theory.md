# Theory

## Governing equations

Source-free Maxwell curl equations in linear, isotropic media:

    dH/dt = -(1/mu)  * curl(E)
    dE/dt =  (1/eps) * curl(H) - (sigma/eps) * E - (1/eps) * J

The divergence equations are satisfied automatically by the Yee scheme if they
hold at t = 0, which they do when all fields start at zero. This is a property
worth a unit test (discrete div(D) stays zero in a source-free region).

## Normalized units

All solver code works in normalized units. Conversion to physical units happens
only at the UI boundary.

    c0 = 1,  eps0 = 1,  mu0 = 1,  eta0 = 1
    dx = dy = 1 (one cell)
    dt = S * dx / c0, where S is the Courant number

Physical mapping: the user picks a cell size `dx_phys` in meters. Then
`dt_phys = S * dx_phys / c`, and a normalized frequency `f_n` (cycles per
step) maps to `f_n / dt_phys` hertz. Put this in one module, `units.rs`, with
round-trip tests. A wavelength of N cells is the only resolution number the
solver cares about.

## 2D modes

A 2D grid means nothing varies in z. Maxwell then decouples into two
independent polarizations.

**TMz** (code prefix `tm`): fields `Ez, Hx, Hy`. Electric field out of plane.
Scalar-like behavior, the right mode for the double slit, waveguides with
perfect-conductor walls, dielectric lenses, and photonic crystal rod lattices.

**TEz** (code prefix `te`): fields `Hz, Ex, Ey`. Electric field in the plane.
Required for surface plasmons: a plasmon needs an electric field component
normal to the metal surface, which TMz does not have. A plasmonic demo run in
TMz is physically wrong no matter how good it looks.

Phases 1 to 5 implement TMz only. TEz arrives in Phase 6.

## Yee grid staggering, TMz

Fixed convention. Code and tests must match it exactly.

    Ez[i, j]   lives at ( i * dx,        j * dy )         time n
    Hx[i, j]   lives at ( i * dx,       (j + 1/2) * dy )  time n + 1/2
    Hy[i, j]   lives at ((i + 1/2) * dx, j * dy )         time n + 1/2

Array sizes for an `nx` by `ny` grid of Ez nodes: `Ez` is `nx * ny`, `Hx` is
`nx * (ny - 1)`, `Hy` is `(nx - 1) * ny`. On the GPU all three are padded to
`nx * ny` for uniform indexing; the padded entries are never read by a correct
kernel, and a test asserts they stay zero.

Update equations, lossless, normalized:

    Hx[i,j] -= (dt / (mu * dy)) * (Ez[i, j+1] - Ez[i, j])
    Hy[i,j] += (dt / (mu * dx)) * (Ez[i+1, j] - Ez[i, j])
    Ez[i,j] += (dt / eps) * ( (Hy[i,j] - Hy[i-1,j]) / dx
                            - (Hx[i,j] - Hx[i,j-1]) / dy )

With electric conductivity `sigma`, the Ez update becomes

    Ez = Ca * Ez + Cb * (curl_H)
    Ca = (1 - sigma*dt/(2*eps)) / (1 + sigma*dt/(2*eps))
    Cb = (dt/eps) / (1 + sigma*dt/(2*eps))

The leapfrog is in-place safe: the H update reads only E and writes only H, and
the E update reads only H and writes only E. No double buffering is needed, on
CPU or GPU. If someone adds ping-pong buffers they have misunderstood the
scheme.

TEz is the dual: swap E and H roles, with material coefficients now living on
the E edges. Because `Ex` and `Ey` sit at different points than cell centers,
permittivity must be evaluated at each edge location. See material averaging
below.

## Boundaries

- **PEC** (perfect electric conductor): tangential E is zero. In TMz, set
  `Ez = 0` on PEC cells. Also the default outer wall behind the PML.
- **PMC**: tangential H zero. Useful as a symmetry plane.
- **CPML**: absorbing layer, see `docs/02-numerics.md`.
- **Periodic / Bloch**: stretch, needed only for band-structure computation.

## Materials

Per-cell material id (u32) indexing a material table. Table entry:

| Field | Meaning |
|---|---|
| eps_inf | high-frequency relative permittivity |
| mu_r | relative permeability (1 in v1) |
| sigma | electric conductivity, normalized |
| kind | vacuum, dielectric, PEC, Drude, Lorentz |
| wp, gamma | Drude plasma frequency and collision rate |
| d_eps, w0, delta | Lorentz strength, resonance, damping |

Material interfaces on a staggered grid are staircased. For Ez nodes the
permittivity is that of the cell. For TEz edge components use the arithmetic
mean of the two cells sharing the edge. State this limitation in the README;
staircasing makes curved-surface convergence first order.

### Dispersive models

Drude (metals):

    eps(w) = eps_inf - wp^2 / (w^2 - j*gamma*w)        (exp(+jwt) convention)

Lorentz (resonant dielectrics):

    eps(w) = eps_inf + d_eps * w0^2 / (w0^2 - w^2 + 2j*delta*w)

Pick one sign convention for the time dependence, write it at the top of the
dispersive module, and use the same one in the Python golden generators. A
convention mismatch flips the sign of the loss and shows up as gain.

## Sources

- **Soft source**: adds to the field. `Ez[i,j] += s(t)`. Transparent to
  returning waves. Default.
- **Hard source**: sets the field. Reflects returning waves like a PEC point.
  Offer it, label it.
- **Waveforms**: Gaussian pulse, modulated Gaussian (center frequency and
  bandwidth), Ricker wavelet, continuous wave with a smooth ramp over at least
  3 periods. An unramped CW source injects broadband junk and a DC offset.
- **Plane wave**: total-field/scattered-field injection, see numerics.
- **Phased array**: N soft sources with programmable phase and amplitude.

A point in 2D is an infinite line source in 3D. Its field decays as
`1/sqrt(r)`, not `1/r`.

## Analytic solutions used for validation

Each of these has a generator script in `validation/gen/`.

**Line source in free space.** For a time-harmonic line current, the field is
a Hankel function of the second kind (with the `exp(+jwt)` convention):

    Ez(rho) proportional to H0^(2)(k * rho)

Compare normalized amplitude and phase versus radius beyond two wavelengths.

**Rectangular PEC cavity, TMz.** Resonant frequencies

    f_mn = (c/2) * sqrt( (m/a)^2 + (n/b)^2 ),  m, n >= 1

The FDTD grid resonates at the frequencies predicted by the *discrete*
dispersion relation, which differ from the above by O(dx^2). Test against both:
exact agreement with the discrete prediction, second-order convergence to the
analytic one.

**Parallel-plate waveguide.** Cutoff `f_c = m*c/(2a)`. Above cutoff the guided
wavenumber is `beta = sqrt(k^2 - (m*pi/a)^2)`. Below cutoff the field decays
as `exp(-alpha*x)` with `alpha = sqrt((m*pi/a)^2 - k^2)`. Measure both.

**Dielectric slab, normal incidence.** Fresnel/Airy reflection and
transmission versus frequency, including Fabry-Perot fringes. One broadband
pulse and two flux monitors give the full spectrum.

**Circular cylinder scattering.** The 2D analogue of Mie theory. For a PEC
cylinder of radius `a`, TMz plane wave incidence:

    a_n = -J_n(ka) / H_n^(2)(ka)
    sigma_2D(phi) = (4/k) * | sum over n of a_n * exp(j*n*phi) |^2

For TEz replace the ratio with derivatives: `-J_n'(ka) / H_n^(2)'(ka)`. The
dielectric cylinder coefficients come from matching tangential E and H at the
surface; the generator script implements them from Balanis (Advanced
Engineering Electromagnetics, cylinder scattering chapter) and must be
cross-checked two ways: the PEC limit as permittivity goes to infinity (TMz),
and the Rayleigh limit for `ka << 1`.

**Small Drude cylinder, TEz.** Quasi-static dipole resonance where
`eps(w) = -1` in vacuum, giving `w_res = wp / sqrt(eps_inf + 1)`. The sphere
condition `eps = -2` does not apply in 2D.

**Double slit.** Fraunhofer far-field intensity

    I(theta) proportional to cos^2(pi*d*sin(theta)/lambda)
                           * sinc^2(pi*a*sin(theta)/lambda)

with slit separation `d` and width `a`. Valid in the far field only; compare
through the near-to-far-field transform, not on a nearby screen line.

**Phased array of line sources.** Array factor for N elements with spacing
`d` and progressive phase `beta`:

    AF(phi) = sin(N*psi/2) / sin(psi/2),  psi = k*d*cos(phi) + beta

This is the honest 2D antenna validation.

**Photonic crystal.** Square lattice of dielectric rods in air has a TM band
gap. The commonly quoted case is rod permittivity 8.9 and radius 0.2a with a
gap near normalized frequency 0.32 to 0.44 (in units of c/a). Treat those
numbers as approximate and confirm against MPB or Joannopoulos et al.
(Photonic Crystals: Molding the Flow of Light) before using them as a gate.

## Energy

Discrete electromagnetic energy for the Yee scheme. The quantity that is
exactly conserved in a lossless closed domain uses H at two adjacent half
steps:

    W^n = (1/2) * sum( eps * Ez^n * Ez^n )
        + (1/2) * sum( mu * H^(n+1/2) . H^(n-1/2) )

Using `H^(n+1/2)` squared instead gives a quantity that oscillates and will
make a correct solver look broken. This is a strong invariant test.

## Known limits, state them in the README

- 2D only until the stretch phase. Structures are infinite in z.
- Staircased geometry. Curved metal surfaces converge slowly and can show
  spurious plasmon peaks.
- f32 on the GPU. Dynamic range about 120 dB, which is why PML numbers are
  measured on the f64 reference.
- Numerical dispersion is anisotropic: waves travel slightly slower along the
  grid axes than along diagonals.
