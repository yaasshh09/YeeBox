# Architecture

## Workspace layout

    Cargo.toml                  workspace
    crates/
      fdtd-core/                CPU reference, f64, the oracle
        src/
          units.rs              normalized <-> physical conversion
          grid.rs               Yee indexing, staggering helpers
          material.rs           material table, coefficient precompute
          scene.rs              Scene description, validation, serde
          fdtd1d.rs             1D solver (tests + TF/SF auxiliary grid)
          tm.rs                 2D TMz update
          te.rs                 2D TEz update (Phase 6)
          cpml.rs               coefficient profiles, psi updates
          dispersive.rs         Drude and Lorentz ADE
          source.rs             waveforms, soft/hard, phased array
          tfsf.rs               plane wave injection
          monitor.rs            probes, DFT, flux
          ntff.rs               near-to-far-field, steering, array factor
          dispersion.rs         numerical wavenumber solver
          energy.rs             discrete energy
      fdtd-gpu/                 wgpu kernel, f32
        src/
          device.rs             adapter/device setup, native and web
          buffers.rs            field, coefficient, material buffers
          pipeline.rs           compute and render pipelines
          sim.rs                GpuSim: step, paint, probe, readback
        shaders/
          update_h.wgsl
          update_e.wgsl
          dft.wgsl
          render.wgsl
      fdtd-wasm/                wasm-bindgen API over fdtd-gpu and fdtd-core
      fdtd-cli/                 headless runs, validation report generator
    web/                        Vite + React + TypeScript
      src/
        sim/                    thin wrapper over the WASM API, worker glue
        ui/                     panels, tools, instruments
        presets/                scene JSON files
        share/                  URL encode/decode
    validation/
      gen/                      Python golden generators (pinned requirements)
      golden/                   committed reference data
      report/                   generated plots and report.md
    docs/  phases/  PROGRESS.md  CLAUDE.md

## Dependency rules

- `fdtd-core` depends on nothing GPU or web related. It compiles to native and
  to `wasm32-unknown-unknown` unchanged.
- `fdtd-gpu` depends on `fdtd-core` for `Scene`, materials, coefficient
  precompute, and units. It never reimplements a coefficient formula. Both
  kernels consume the same precomputed coefficient arrays, so a formula bug
  cannot hide in one kernel only.
- `fdtd-wasm` is glue. No logic.
- `web/` contains no numerical code. If a number is computed in TypeScript it
  is a display transform only.

## Scene: the single source of truth

A `Scene` fully determines a simulation. Both kernels are constructed from it.
It serializes to JSON (and from there to a compressed URL fragment).

    Scene {
      version: u32,
      grid: { nx, ny, dx_phys_m, courant },
      mode: TM | TE,
      pml: { thickness, m, sigma_scale, kappa_max, alpha_max },
      materials: [ Material ],          // table
      geometry: [ Shape ],              // rect, circle, polygon, each with a
                                        // material index, painted in order
      paint: Option<RLE mask>,          // freehand strokes, run-length encoded
      sources: [ Source ],
      monitors: [ Monitor ],
      view: { colormap, field, gain },  // display only
    }

Geometry is rasterized to the material id grid by one function in `fdtd-core`,
used by both kernels. Rasterization rule: a cell takes the material of the
last shape containing its center.

`Scene::validate()` returns typed errors for: Courant above the limit,
`eps_inf < 1`, dispersive material inside the PML (or within two nodes of
its interface), a Lorentz `w0 * dt >= 1`, two different Drude or Lorentz cells
sharing a TEz edge, a TEz plane wave in a non-vacuum background, a source
inside the PML, monitors outside the domain, grid too large for the device
limits. Resolution problems that are allowed but worth knowing (a Drude skin
depth under 2 cells, a material wavelength under 10 cells) come back as
warnings in `Scene::status`, not errors.

A far-field monitor (`"type": "ntff"`) is a box of Ez nodes, all four sides
or a listed subset (`sides`), with `freqs` and `angles` (360 when left out).
Its taps are the flux-line taps of each side, so the GPU gathers them and the
CPU accumulates the DFTs in f64, as for every other monitor; `Spectrum::
FarField` carries the far-field amplitude per frequency and angle. The
contour must stay clear of the CPML and of the TF/SF box (one node around
it), and periodic y has no far field. Material on the contour is a warning.

`mode` defaults to TM and is left out of the JSON when it is TM, so every
file and link from before Phase 6 decodes and re-encodes unchanged; no
migration was needed. Anything that builds a CPU kernel from an arbitrary
scene goes through `kernel::CpuKernel`, which picks `SimTm` or `SimTe`.

Version the format from day one. Shared URLs must keep working after updates,
so every format change adds a migration.

## The two kernels share an interface

    trait Kernel {
        fn from_scene(scene: &Scene) -> Result<Self>;
        fn step(&mut self, n: u32);
        fn time_step(&self) -> u64;
        fn read_field(&mut self, which: FieldId) -> Vec<f32 or f64>;
        fn probe(&mut self, i: u32, j: u32) -> f64;
        fn paint(&mut self, rect: Rect, material_ids: &[u32]);
        fn set_source_params(...);
    }

The cross-implementation test harness takes a `Scene`, runs both kernels for N
steps, and diffs every field. Both kernels run either mode: the GPU kernel
keeps TEz in the same buffer slots (node field, y-edge field, x-edge field)
and reads its medium rows from one packed table (`fdtd-gpu/src/buffers.rs`):
one row per material in TMz, and in TEz one per unordered pair of materials,
so painting rewrites only the edge ids. Drude J and Lorentz P live in one
`aux` buffer per E array, allocated only for dispersive scenes, and every E
shader calls the same `medium.wgsl`, a port of `dispersive::Aux::update`. Any new feature lands in `fdtd-core` first with
physics tests, then in `fdtd-gpu` with an equivalence test. Never the other
order.

## WASM boundary

`fdtd-wasm` exports a small imperative API:

    createSim(canvas, sceneJson) -> Promise<SimHandle>
    sim.step(n)
    sim.render()
    sim.paint(x0, y0, w, h, Uint32Array)
    sim.setSource(index, paramsJson)
    sim.probe(i, j) -> Promise<number>
    sim.readMonitor(index) -> Promise<Float32Array>
    sim.scene() -> string
    sim.dispose()

Phase 7 adds `sim.farField()` (JSON: the first contour's pattern in dB
below its peak, the live array factor when the scene is a plain CW array,
and an edit count) and `sim.steer(degrees)`, which sets every CW source's
phase in Rust. Every live edit restarts the spectral accumulators, so the
polar plot shows the new scene once it settles.

Rust owns the wgpu device, pipelines, and the canvas surface. TypeScript never
touches WebGPU directly. Errors cross the boundary as structured objects with
a stable `code` field, never as panics. Install a panic hook that reports to
the UI in debug builds.

Capability detection happens once at startup: WebGPU present and adapter
acquired, else fall back to the `fdtd-core` CPU kernel compiled to WASM at a
reduced grid, running in a Web Worker and rendering through a 2D canvas.

## Frame loop

    requestAnimationFrame:
      apply queued paint strokes and parameter changes
      sim.step(steps_per_frame)
      sim.render()
      every k frames: read probes, update instrument panels

`steps_per_frame` adapts to hold the frame budget: measure GPU time with
timestamp queries where available, otherwise by frame pacing.

Pause, single-step, and speed controls are first-class. Single-stepping a
wave is one of the most instructive things the tool can do.

## Native headless use

`fdtd-cli` runs scenes without a window, on either kernel:

    fdtd run scene.json --kernel cpu --steps 5000 --out fields.npz
    fdtd validate                      # runs every validation case
    fdtd report                        # regenerates validation/report/

GPU tests in CI need a software adapter. Try the Vulkan software rasterizer
that ships with Mesa on the Linux runner. If no adapter can be created, the
GPU equivalence tests must report as skipped with a loud message, never as
passed. Record in `PROGRESS.md` which tests run in CI and which only run
locally.

## Error handling and determinism

- Library crates return `Result<T, FdtdError>`; no `unwrap` outside tests.
- The CPU kernel is fully deterministic. The GPU kernel is deterministic on a
  given device and driver; cross-device differences at the f32 round-off level
  are expected and tolerated by the equivalence tolerance.
- Fixed RNG seeds in all tests.

## Performance work

Not before Phase 8. Then profile first. Likely levers in order: steps per
dispatch batching, avoiding per-frame readback, reducing buffer count by
packing fields, restricting dispersive updates to a bounding box.
