# Phase 5: Browser

**Target: 3 weeks. Docs: 03-architecture (WASM boundary, frame loop), 06-experience.**

This is the application cut line. At the end of this phase the project is
public-ready in its minimal honest form.

## Scope

1. `fdtd-wasm`: wasm-bindgen API as specified in architecture; panic hook;
   structured errors
2. `web/`: Vite, React, TypeScript shell; canvas; capability detection
3. Render pipeline: field colormap, material overlay, PML marking
4. Tools: paint, erase, rectangle, circle, slit pair, point source, probe
5. Transport: play, pause, single step, speed, reset
6. Status strip with physical units and cells-per-wavelength warning
7. Presets: double slit, waveguide and cutoff, lens
8. Share: scene to URL fragment and back, with format versioning
9. CPU fallback path in a Web Worker for browsers without WebGPU
10. Touch input
11. Deploy to a static host; CI builds and publishes

## Acceptance

- The ten-second test passes with three people who have never seen it
- Cold load to first moving frame under 3 seconds on a mid-range laptop
- 1024 x 1024 at 60 fps with at least 8 steps per frame on an integrated GPU
  (target; record the measured numbers and the adaptive fallback behavior)
- A shared URL reproduces the scene exactly on a second device
- Browser GPU kernel matches the native CPU kernel on three scenes through an
  in-browser self-test page (same 1e-4 gate)
- No WebGPU: fallback runs, notice shown, no blank page
- Works in current Chrome, Edge, Safari, and Firefox, or documents exactly
  what fails where

## Do not

- Put numerical code in TypeScript
- Add instruments beyond the oscilloscope yet

## Done when

The URL is live, the Verify panel links to the generated validation report,
and the README has a 15 second screen capture at the top.
