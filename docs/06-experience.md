# Experience

What a person sees and does. The solver earns respect; this file earns the
click, the second minute, and the share.

## The ten-second test

A stranger opens the link on whatever device is in their hand. Within ten
seconds, with no instructions:

1. Something is already moving: the double slit preset runs on load.
2. They touch or drag on the canvas and a wall appears under their finger.
3. The wave pattern changes because of what they did.

If any of the three fails, the launch is not ready. Test it on real people who
have never seen it.

## Layout

Lab instrument idiom, consistent with AtomSIM (the sibling project in
`../atom_sim/web`): monochrome `light-dark()` tokens, Poppins, a top bar over
three columns (300 px rail, the stage, 300 px panel), and below 900 px a
separate mobile shell where the stage fills the screen and the controls live
in a bottom sheet that drags up.

- **Canvas** fills most of the window. The field is the hero.
- **Tool rail** (left or bottom on mobile): paint, erase, shapes, source,
  probe, monitor line, select/move.
- **Material picker**: vacuum, glass (eps 2.25), high-index (eps 12),
  perfect conductor, absorber, metal (Drude), resonant (Lorentz), custom.
- **Transport**: play/pause, single step, speed (steps per frame), reset
  fields, clear scene.
- **Instruments** (collapsible right panel): oscilloscope, spectrum, far-field
  polar plot, flux/transmission, energy meter.
- **Status strip**: grid size, cell size in physical units, frequency and
  wavelength of the active source, cells per wavelength with a warning below
  10, simulated time, throughput in Mcells/s, kernel in use (GPU or CPU).
- **Verify** button: opens the validation summary with links to the full
  report and the source.

`text-transform: uppercase` on section headings only.

## Tools

- **Paint / erase**: brush with size control, writes material ids.
- **Shapes**: rectangle, circle, line wall, slit pair (two parameters:
  separation and width), lens (plano-convex, biconvex), rod lattice
  (square or triangular, period and radius), ring.
- **Source**: place point source, line of sources, plane wave box; choose
  waveform, frequency (shown as wavelength in cells and in physical units),
  phase and amplitude. Dragging a source while running is allowed and looks
  great.
- **Probe**: click to drop; feeds the oscilloscope and spectrum.
- **Monitor line**: drag a segment; feeds flux and transmission.
- **Far-field contour**: a rectangle; feeds the polar plot.

Every edit applies live without restarting the simulation, except changes to
grid size or mode, which reset with a confirmation.

## Field display

- Selectable quantity: Ez (or Hz), |E|, time-averaged intensity, energy
  density, Poynting vector magnitude.
- Diverging colormap for signed fields with zero as neutral, sequential for
  intensity. Gain control and auto-gain.
- Materials drawn as a translucent overlay with crisp outlines, PML region
  subtly marked so users can see where absorption happens.
- Optional time-averaged intensity mode is what makes interference patterns
  read clearly; the double slit preset uses a split view (instantaneous field
  left, averaged intensity right) or a toggle.

## Presets

Each preset is a `Scene` JSON file with a title, one sentence on what to
watch, one suggested interaction, and a link to the related validation case.

| Preset | Watch | Try | Validation |
|---|---|---|---|
| Double slit | fringes on the far side, live intensity plot vs Fraunhofer curve | change slit separation | A22 |
| Single slit | diffraction widening as the slit narrows | drag slit width | A22 |
| Waveguide and cutoff | guided mode, then evanescent decay below cutoff | lower the frequency | A11, A12 |
| Lens | plane wave focusing to a spot | swap material index | A13 |
| Total internal reflection | evanescent tail, then tunneling through a thin gap | shrink the gap | A13 |
| Fabry-Perot slab | transmission peaks in the spectrum panel | change thickness | A13 |
| Bragg mirror | stopband in transmission | add or remove layers | A13 |
| Photonic crystal | no propagation inside the gap, then guiding along a line defect and around a bend | erase rods to draw a path | A24 |
| Ring resonator | light coupling into the ring only at resonance | sweep frequency | none, qualitative |
| Phased array | beam steering, live polar plot | drag the phase slider | A23 |
| Scattering cylinder | shadow and forward lobe, polar plot vs series | change radius | A19, A20 |
| Plasmonic cylinder (TEz) | field enhancement at the surface at resonance | sweep frequency | A21 |
| Silver mirror | standing waves in front of a real metal | bend the mirror with silver paint | A16 |
| Resonant absorber | Lorentz rods soak up light at their resonance | change the wavelength | A17 |
| Empty sandbox | nothing | everything | none |

Presets marked with a validation case show a small "matches theory" badge
that opens the comparison plot. That badge is the bridge between the two
audiences.

The View panel has a TM/TE switch that rebuilds the current scene in the other
polarization. Switching the plasmon preset to TM turns it into a custom scene
and says why: a plasmon shown in TMz would be physically wrong. Silver is in
the paint palette; a silver stroke stays two cells clear of the absorbing
border, and the status strip warns when a metal or a dispersive material is
under-resolved.

## Instruments

- **Oscilloscope**: time trace at a probe, with trigger-free rolling display.
- **Spectrum**: FFT of the probe trace, peak marker with frequency readout.
- **Polar plot**: far-field pattern from the contour, with the analytic curve
  overlaid when the preset has one.
- **Transmission**: flux ratio spectrum; guides the user through the
  normalization run ("remove the structure, press Reference, restore").
- **Energy**: total field energy vs time; shows absorption by the PML when
  the source stops.

## Sharing

- "Share" encodes the current `Scene` into the URL fragment (JSON,
  compressed, base64url). Opening the link reproduces the scene exactly.
- If the encoded scene exceeds a safe URL length, offer file export instead
  and say why.
- Export: PNG of the canvas, short WebM clip, scene JSON.

## Onboarding

No modal tutorial. Three one-line hints that appear once each and disappear on
first use: "Drag to draw a wall", "Tap to add a source", "Try another
experiment" pointing at presets.

## Failure states

- No WebGPU: CPU fallback at 256 x 256 with a one-line notice and a link
  explaining which browsers run the full version. Never a blank page.
- Device lost: recreate and restore from the current `Scene`, tell the user.
- Scene validation error: show the specific message next to the offending
  control.
- Under-resolved source: warning in the status strip, not a block.

## Accessibility and reach

- Full keyboard access to tools and transport; canvas painting has a
  shape-entry alternative.
- Colormaps chosen to be readable with common color vision deficiencies.
- Respect reduced-motion preference by starting paused.
- Touch: one finger paints, two fingers pan and zoom.
- Page weight budget: under 2 MB compressed for first load.

## What makes people share it

A wave simulation is shared when someone makes something they did not expect.
Protect the conditions for that: instant response, no way to get stuck, easy
undo, a reset that always works, and a share link that reproduces exactly
what they saw.
