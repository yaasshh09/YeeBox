//! Tier 4: the GPU kernel against the CPU reference.
//!
//! Its own harness (libtest-mimic) so that a machine without a GPU adapter
//! reports every trial as ignored, with a banner, instead of as passed.
//! With `FDTD_REQUIRE_GPU=1` (set in CI) a missing adapter is a failure.

use fdtd_gpu::Gpu;
use libtest_mimic::{Arguments, Failed, Trial};
use wgpu::util::DeviceExt;

type Check = fn(&'static Gpu) -> Result<(), Failed>;

/// A buffer written on the CPU reads back bit for bit.
fn readback_round_trip(gpu: &'static Gpu) -> Result<(), Failed> {
    let values: Vec<f32> = (0..1000).map(|k| (k as f32 * 0.37).sin()).collect();
    let buf = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&values),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    let back = gpu.read_f32(&buf)?;
    let want: Vec<f64> = values.iter().map(|&v| v as f64).collect();
    if back != want {
        return Err("readback differs from the upload".into());
    }
    Ok(())
}

fn main() {
    let args = Arguments::from_args();
    let gpu = match Gpu::shared() {
        Ok(gpu) => {
            eprintln!("tier 4 on {}", gpu.describe());
            Some(gpu)
        }
        Err(e) => {
            if std::env::var("FDTD_REQUIRE_GPU").is_ok_and(|v| v == "1") {
                eprintln!("FDTD_REQUIRE_GPU=1 but no GPU: {e}");
                std::process::exit(1);
            }
            eprintln!(
                "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            );
            eprintln!("!!! TIER 4 SKIPPED: no GPU adapter ({e})");
            eprintln!("!!! every GPU trial below is IGNORED, not passed");
            eprintln!(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            );
            None
        }
    };
    let checks: Vec<(&str, Check)> = vec![("readback_round_trip", readback_round_trip)];
    let trials = checks
        .into_iter()
        .map(|(name, check)| {
            Trial::test(name, move || check(gpu.expect("ignored without a gpu")))
                .with_ignored_flag(gpu.is_none())
        })
        .collect();
    libtest_mimic::run(&args, trials).exit();
}
