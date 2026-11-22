# YeeBox

![The double slit running live: painting over one slit turns the fringes into a single wave, and erasing it brings them back](docs/media/demo.gif)

Try it at https://yaasshh09.github.io/YeeBox/ (live once the site is published).

Working name. I'm building a real-time solver for Maxwell's equations that runs
in a browser tab on the GPU: you paint materials, drop a source, and watch
waves reflect, diffract, and resonate, with every frame a live FDTD solve on a
Yee grid.

That's the goal. Here's where it stands today, honestly:

- **Done:** a plain f64 CPU reference solver in Rust (1D and 2D TMz), checked
  against exact discrete identities and analytic physics; a convolutional PML
  absorbing boundary with its reflection measured in dB; an axis-aligned
  total-field / scattered-field plane wave source; and frequency-domain
  monitors that measure reflection, transmission and guided modes. On top of
  that, an f32 GPU kernel (wgpu and WGSL compute shaders) that matches the
  reference on every committed scene, and a browser app that runs it live:
  paint walls and glass, drop and drag sources, probe the field, share the
  exact scene as a link.
- **Also done:** the second polarization, TEz (electric field in the plane,
  which surface plasmons need), and frequency-dependent materials: Drude
  metals and Lorentz resonators by auxiliary differential equation, in both
  polarizations, on the CPU reference, on the GPU, and in the browser. The app
  has a TM/TE switch, silver to paint with, and three new presets: a plasmon
  on a silver wire, a silver mirror and a resonant absorber.
- **Not done yet:** oblique plane waves, scattering and far-field checks.
  I'll only claim those once their tests pass.

The reference solver stays deliberately simple. Its only job is to be
obviously correct, so it can prove the fast GPU kernel right.

## The absorbing boundary

A simulation box needs edges that swallow outgoing waves instead of bouncing
them back. I use a CPML (Roden and Gedney, with the complex frequency shift)
and I measure it rather than eyeball it: the same pulse runs in a small box
wrapped in the layer and in a huge box whose walls are too far away to matter,
and the difference at probes along the layer face is the reflection.

![CPML reflection vs angle of incidence](validation/report/cpml_reflection.svg)

| Layer | Worst reflection, 0 to 70 deg | At 80 deg (grazing) |
|---|---|---|
| 10 cells | -86.5 dB | -85.5 dB |
| 16 cells | -127.3 dB | -125.4 dB |

The gate was -60 dB for 10 cells and the target -80 dB for 16 or more. I tuned
the layer parameters with a sweep against this measurement, ranking by the
worst angle including grazing, and I never tuned anything by looking at a
picture. `fdtd report` regenerates the plot and
[the full report](validation/report/report.md).

## What else is measured

| Check | Result | Gate |
|---|---|---|
| 1D pulse at the magic time step, 1000 cells | exact | 1e-12 |
| Phase velocity vs the discrete dispersion relation, 1D and 2D | 2.2e-16 | 1e-4 |
| Cavity resonances vs the discrete prediction, first six modes | 5e-10 to 3e-8 | 1e-4 |
| Cavity convergence order | 2.0014 | 1.8 to 2.2 |
| Energy drift, closed lossless cavity, 10000 steps | 3.1e-15 | 1e-10 |
| Line source vs the Hankel solution, 2 to 5 wavelengths | 3.9e-4 | 0.02 |
| Late time with the CPML, 100000 steps | decays to 4e-32 of peak, never grows | no growth |
| Plane wave leakage into the scattered-field region | -303.9 dB | -120 dB |
| Dielectric slab R and T vs the Airy formula | 4.9e-3 | 0.01 |
| R + T for the lossless slab | 1 to 6.9e-8 | 1 to 0.005 |
| Attenuation in a lossy medium | 0.40 percent | 1 percent |
| Waveguide guided wavelength and cutoff | 0.030 percent | 0.5 percent |
| Waveguide evanescent decay below cutoff | 0.060 percent | 1 percent |
| TEz against TMz under duality, vacuum line source, with reflections and a CPML | bit for bit | 1e-10 |
| Drude slab R and T vs the complex permittivity, both polarizations | 2.2e-3 | 0.02 |
| Lorentz slab R and T vs the complex permittivity, both polarizations | 4.8e-3 | 0.02 |
| Small Drude cylinder resonance vs wp / sqrt(eps_inf + 1), TEz, radius 10 cells | 0.51 percent | 5 percent |
| Energy with lossless Drude or Lorentz media, J and P included, 10000 steps | 8e-14 | 1e-10 |

For the slab, the lossy medium and the waveguide, the error that remains is
the grid's own dispersion, and I can show it: the golden data also carries the
exact answer of the discrete Yee scheme, and the measurements land on it to
1e-7 or better. The bands were picked from that prediction before any run.

![Slab R and T vs frequency](validation/report/slab_rt.svg)

The GPU kernel runs all of this too, checked against the CPU on Drude and
Lorentz scenes in both polarizations, with live painting of metal mid-run:
the worst difference is 1.2e-5 of the peak field against a 1e-4 gate, and the
browser self-test now includes the TEz Drude scene (1.3e-6).

Silver is a Drude fit (Vial et al. 2005). At the 25 nm cells of the optical
presets its skin depth is under one cell, so the app says it acts as a lossy
mirror there; the plasmon preset uses 2 nm cells, where the skin depth is
about 11. A TE box with no absorbing layer has a magnetic outer wall, not a
metal one, because that keeps it the exact mirror image of the TM box.

The Drude cylinder is the plasmon check, and it shows staircasing honestly. A
circle drawn on a square grid is not a circle, and at a radius of 5 cells the
staircase splits the surface mode into several peaks; the tallest is 14
percent off. At 10 cells the dipole peak is 0.5 percent off, and at 20 cells
0.3 percent, which is about what a cylinder that is not quite small (k a =
0.1) should show. The peak is picked blind, the tallest one, and a NumPy script
re-measures every run on its own.

![Drude cylinder resonance vs radius](validation/report/drude_cylinder.png)

`fdtd validate` runs every case and prints this table with its gates;
`cargo test` checks the same cases.

## The GPU kernel

The GPU kernel is a port, not a rewrite. Every coefficient, CPML profile,
source sample and plane-wave correction is computed by the reference crate in
f64 and uploaded; the shaders hold no formula (a test checks they contain no
float literal at all). Then I run both kernels on the same scene and diff
every field.

| Check | Result | Gate |
|---|---|---|
| 11 scenes, every field, probe and spectrum, 2000 steps | 5.9e-6 worst, most near 1e-6 | 1e-4 of peak |
| Growth over 50000 steps, lossless cavity | 7.2e-6, grows like n^0.44 | no faster than n^1.2 |
| Workgroup size 4, 8, 16 | bit-identical | identical |
| Padding and wall nodes | exactly zero | zero |
| 1024 x 1024, GTX 1660 Ti | 3.0 Gcells/s, about 48 steps per 60 fps frame | 8 steps per frame |

![GPU minus CPU vs step](validation/report/gpu_cpu_diff.svg)

The difference is f32 round-off measured against the peak field of the run.
It levels off instead of climbing, which is what rules out a hidden
instability. The scenes cover closed boxes, the CPML, periodic y, PEC,
lossy and dielectric shapes, every waveform, soft and hard sources, a phased
array, plane waves in all four directions and every monitor type. Any new
feature has to add a scene there.

## The browser app

The page runs the same GPU kernel through WebGPU, or the f64 CPU reference in a
Web Worker on a smaller grid when the browser has no WebGPU. It opens on a
double slit, and a drag paints a wall. There are three presets (double slit,
waveguide and cutoff, lens), tools for paint, erase, rectangles, circles, slit
pairs, sources and a probe with an oscilloscope, and a time-averaged intensity
view. Share puts the whole scene, paint included, in the URL; opening it
reproduces the scene byte for byte.

What I measured on my machine (GTX 1660 Ti):

| | |
|---|---|
| 1024 x 1024 grid in Chrome | about 2.2 Gcells/s, around 30 steps per 60 fps frame |
| Load to a running solver | 367 ms, local production build |
| First-load weight | 562 kB gzipped, budget 2 MB |
| Browser GPU vs native CPU (self-test page) | 5.1e-6 of the peak field worst, gate 1e-4 |
| Browsers checked | Chrome 157 and Firefox 159 (GPU path), headless Chromium (CPU path) |

The Verify button lists every acceptance case, and `selftest.html` runs the
GPU kernel in your browser against native reference fields shipped with the
page. I haven't measured an integrated laptop GPU, Safari, or Edge yet.

## Running it

    cargo test --workspace                       # everything, under a minute
    cargo run --release -p fdtd-cli -- validate  # every acceptance case, with gates
    cargo run --release -p fdtd-cli -- report    # regenerate validation/report
    cargo run --release -p fdtd-cli -- bench     # GPU and CPU throughput
    cargo run --release -p fdtd-cli -- run --scene validation/scenes/line_source_cpml.json

For the web app (Node 24, and `wasm-bindgen-cli` 0.2.130 to match the crate):

    cd web
    npm ci
    npm run wasm      # build the Rust side and its JS bindings
    npm run dev       # http://localhost:5173
    npm test          # unit tests
    npm run build && npx playwright test

The GPU tests need an adapter. Without one they show up as ignored, with a
banner, never as passed. `fdtd pml-sweep` reruns the layer tuning and writes a CSV ranked from best to
worst.
