//! The f32 GPU kernel for the 2D TMz solver, on `wgpu` and WGSL compute
//! shaders. It is a port of `fdtd-core::tm`, not a redesign: every
//! coefficient, profile, source sample and TF/SF correction is computed by
//! `fdtd-core` in f64 and uploaded, so the shaders hold no formula. Its
//! acceptance is agreement with the CPU reference on identical scenes
//! (Tier 4 in `docs/04-validation.md`), never agreement with physics alone.

pub mod buffers;
pub mod colormap;
pub mod device;
pub mod pipeline;
pub mod render;
pub mod sim;
#[cfg(not(target_arch = "wasm32"))]
pub mod tier4;

pub use device::Gpu;
pub use sim::GpuSim;
