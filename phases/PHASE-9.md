# Phase 9 (stretch): 3D and real antennas

**Only after Phase 8 ships. Docs: all.**

The 2D solver cannot simulate a finite antenna. This phase can.

## Scope

1. Full 3D Yee update in `fdtd-core`, six field components, CPML on six faces,
   Courant limit 1/sqrt(3)
2. 3D GPU kernel; realistic grid about 128 cubed to 192 cubed depending on
   device memory
3. Slice rendering (three orthogonal planes) and a simple volume view
4. 3D near-to-far-field on a closed box, directivity pattern
5. Thin-wire or staircased dipole; microstrip patch on a dielectric substrate
   with a simple feed model

## Acceptance

- 3D cavity resonances match the discrete prediction to 0.01 percent
- CPML reflection at or below -60 dB
- Half-wave dipole: pattern and input impedance trend against NEC2 (thin-wire
  method of moments). Expect impedance to be the hard part; report it.
- Patch antenna: resonant frequency within 3 percent of the cavity-model
  estimate and of an openEMS run at matched resolution

## Note

This is where the words "patch antenna" and "NEC2" become legitimate. Do not
use them before.
