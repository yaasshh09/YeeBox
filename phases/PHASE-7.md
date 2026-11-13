# Phase 7: Far field and scattering

**Target: 2 to 3 weeks. Docs: 02-numerics (NTFF), 01-theory (cylinder series, array factor).**

Turn near fields into radiation patterns and prove the solver against the 2D
analogue of Mie theory.

## Scope

1. `ntff.rs`: contour DFT of tangential E and H, co-location, far-field
   integration, scattering width
2. Python generators: PEC and dielectric cylinder series (TMz and TEz), with
   the PEC-limit and Rayleigh-limit cross-checks; array factor; Fraunhofer
   double slit
3. GPU DFT accumulation over contour cells only
4. Polar plot instrument with analytic overlay
5. Presets: scattering cylinder, phased array beam steering; upgrade double
   slit with the far-field comparison
6. Meep cross-check of three scenes, results committed with settings

## Acceptance

- Single line source inside the contour gives an isotropic far field within
  0.1 dB
- A19: PEC cylinder scattering width within 1 dB at ka = 1, 2, 5
- A20: dielectric cylinder (eps = 4) within 1 dB
- Error decreases when the cylinder radius in cells doubles
- A22: double slit far field within 1 dB over the main lobes
- A23: phased array beam direction within 1 degree across the steering range
- A24 recorded; promoted to a gate only after the reference is confirmed

## Do not

- Compare in deep nulls
- Present any 2D result as a 3D antenna pattern

## Done when

README shows scattering width vs angle, simulation over series, for three
values of ka.
