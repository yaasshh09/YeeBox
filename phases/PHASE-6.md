# Phase 6: Dispersive media and TEz

**Target: 3 weeks. Docs: 02-numerics (ADE), 01-theory (dispersive models, TEz).**

Metals and resonant dielectrics that behave correctly across frequency, and
the polarization that plasmonics requires.

## Scope

1. `te.rs`: Hz, Ex, Ey update with edge-averaged permittivity, CPML, sources
2. `dispersive.rs`: Drude and Lorentz ADE for both modes, CPU first
3. Scene validator rules: no dispersive material in the PML; skin-depth and
   resolution warnings
4. Python generators for slab R and T with complex permittivity
5. GPU port of TEz and both ADE models; auxiliary buffers allocated only when
   needed
6. Presets: plasmonic cylinder (TEz), metal mirror with realistic loss,
   resonant absorber

## Acceptance

- A18: TEz and TMz vacuum line-source fields agree under duality to 1e-10
- A16, A17: Drude and Lorentz slab R and T within 0.02 absolute over the valid
  band
- A21: small Drude cylinder resonance within 5 percent of
  wp / sqrt(eps_inf + 1), with a convergence study showing the staircasing
  trend
- Passive media never gain energy: energy non-increasing with source off
- Tier 4 equivalence for four new scenes (Drude TM, Drude TE, Lorentz TM,
  Lorentz TE)
- ADE coefficient derivations present in doc comments

## Do not

- Show a plasmon preset in TMz
- Let dispersive material touch the PML

## Done when

The plasmonic preset runs in the browser with its "matches theory" badge
linked to the A21 plot.
