# Phase 4: GPU kernel

**Target: 2 to 3 weeks. Docs: 02-numerics (GPU kernel, precision), 03-architecture.**

Port, do not redesign. Every formula already lives in `fdtd-core`.

## Scope

1. `fdtd-gpu`: device setup that works natively; storage buffers for fields,
   coefficients, material ids
2. `update_h.wgsl`, `update_e.wgsl` including CPML terms, identity coefficients
   in the interior, bounds checks at the dispatch overhang
3. Sources on the GPU: point, array, TF/SF corrections
4. Probe readback through a staging buffer; full-field readback for tests only
5. `Kernel` trait implementation for `GpuSim`
6. Cross-implementation harness: one `Scene`, both kernels, field diff
7. At least 8 scenes under `validation/scenes/` covering every Phase 1 to 3
   feature
8. Headless native benchmark reporting Mcells/s

## Acceptance

- Tier 4: max relative field difference under 1e-4 after 2000 steps on every
  scene
- 50000 step run: difference grows no faster than linearly
- Padding entries stay zero on the GPU
- Result is independent of workgroup size (test with 4, 8, 16)
- Coefficient formulas appear nowhere in WGSL; shaders only consume uploaded
  arrays
- GPU tests report as skipped, loudly, when no adapter exists; never as passed
- Throughput recorded in `PROGRESS.md` for the development machine

## Do not

- Add any feature to the GPU kernel that the CPU kernel lacks
- Use storage textures or ping-pong buffers without a written reason

## Done when

The diff harness is green for all scenes natively, and the README draft has a
plot of GPU minus CPU difference vs step count.
