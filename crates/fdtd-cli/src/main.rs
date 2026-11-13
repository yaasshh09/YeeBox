//! `fdtd`: command line front end for the reference solver.

mod npz;

use clap::{Parser, Subcommand, ValueEnum};
use fdtd_core::fdtd1d::{Boundary1d, Sim1d, Source1d};
use fdtd_core::grid::Component;
use fdtd_core::kernel::Kernel;
use fdtd_core::scene::Scene;
use fdtd_core::source::{SourceKind, Waveform};
use fdtd_core::tm::SimTm;
use std::path::{Path, PathBuf};

/// Largest space-time record we allow, in f64 values (2 GiB).
const MAX_RECORD: usize = 1 << 28;

#[derive(Parser)]
#[command(name = "fdtd", version, about = "Reference FDTD solver")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the 1D solver with a Gaussian pulse and dump the space-time Ez field to NPZ,
    /// or, with --scene, run a 2D TMz scene and dump its final fields and probe records.
    /// All quantities are normalized: dx = 1, c0 = 1, dt = S.
    Run(RunArgs),
}

#[derive(clap::Args)]
struct RunArgs {
    /// 2D scene JSON. When given, the 1D options do not apply and the NPZ holds
    /// ez, hx, hy (ny by nx, padded, index [j, i]), probes (one row per point
    /// monitor, steps + 1 samples), courant, and dt.
    #[arg(long, conflicts_with_all = ["nx", "courant", "left", "right", "source_node", "tau", "t0", "kind"])]
    scene: Option<PathBuf>,
    /// Number of Ez nodes.
    #[arg(long, default_value_t = 400)]
    nx: usize,
    /// Number of time steps.
    #[arg(long, default_value_t = 1200)]
    steps: usize,
    /// Courant number S, 0 < S <= 1.
    #[arg(long, default_value_t = 1.0)]
    courant: f64,
    #[arg(long, value_enum, default_value_t = End::Mur)]
    left: End,
    #[arg(long, value_enum, default_value_t = End::Mur)]
    right: End,
    /// Ez node index of the source.
    #[arg(long, default_value_t = 100)]
    source_node: usize,
    /// Gaussian width tau, normalized time.
    #[arg(long, default_value_t = 10.0)]
    tau: f64,
    /// Gaussian center t0, normalized time. Defaults to 6 tau, so the turn-on step is below 1e-15.
    #[arg(long)]
    t0: Option<f64>,
    #[arg(long, value_enum, default_value_t = Kind::Soft)]
    kind: Kind,
    /// Output NPZ path. Holds ez (steps + 1 by nx, row n is time step n), courant, and dt.
    #[arg(long, default_value = "out/pulse.npz")]
    out: PathBuf,
}

#[derive(Clone, Copy, ValueEnum)]
enum End {
    Pec,
    Mur,
}

#[derive(Clone, Copy, ValueEnum)]
enum Kind {
    Soft,
    Hard,
}

fn run(a: &RunArgs) -> Result<(), Box<dyn std::error::Error>> {
    let end = |e| match e {
        End::Pec => Boundary1d::Pec,
        End::Mur => Boundary1d::Mur1,
    };
    let rows = a.steps.checked_add(1).ok_or("steps too large")?;
    let total = rows
        .checked_mul(a.nx)
        .filter(|&t| t <= MAX_RECORD)
        .ok_or(format!("record of {rows} x {} values is too large", a.nx))?;

    let mut sim = Sim1d::new(a.nx, a.courant, end(a.left), end(a.right))?;
    sim.add_source(Source1d {
        node: a.source_node,
        waveform: Waveform::Gaussian {
            t0: a.t0.unwrap_or(6.0 * a.tau),
            tau: a.tau,
        },
        amplitude: 1.0,
        kind: match a.kind {
            Kind::Soft => SourceKind::Soft,
            Kind::Hard => SourceKind::Hard,
        },
    })?;

    let mut ez = Vec::with_capacity(total);
    ez.extend_from_slice(sim.ez());
    for _ in 0..a.steps {
        sim.step();
        ez.extend_from_slice(sim.ez());
    }

    let s = sim.courant();
    write_out(
        &a.out,
        &[
            npz::Array {
                name: "ez",
                shape: &[rows, a.nx],
                data: &ez,
            },
            npz::Array {
                name: "courant",
                shape: &[],
                data: &[s],
            },
            npz::Array {
                name: "dt",
                shape: &[],
                data: &[s],
            },
        ],
    )?;
    println!(
        "wrote {} ({rows} x {} Ez samples, S = {s})",
        a.out.display(),
        a.nx
    );
    Ok(())
}

fn write_out(path: &Path, arrays: &[npz::Array]) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    npz::write_npz(path, arrays)?;
    Ok(())
}

/// Runs a 2D scene for `steps` steps and writes the NPZ described on `RunArgs::scene`.
fn run_scene(
    scene_path: &Path,
    steps: usize,
    out: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let scene = Scene::from_json(&std::fs::read_to_string(scene_path)?)?;
    let (nx, ny) = (scene.grid.nx, scene.grid.ny);
    let n_probes = scene.monitors.len();
    let rows = steps.checked_add(1).ok_or("steps too large")?;
    rows.checked_mul(n_probes)
        .filter(|&t| t <= MAX_RECORD)
        .ok_or(format!(
            "probe record of {n_probes} x {rows} values is too large"
        ))?;

    let mut sim = SimTm::from_scene(&scene)?;
    Kernel::step(&mut sim, steps as u64);
    let fields = [Component::Ez, Component::Hx, Component::Hy].map(|c| sim.read_field(c));
    let [ez, hx, hy] = fields;
    let (ez, hx, hy) = (ez?, hx?, hy?);
    let probes: Vec<f64> = sim.probe_history()?.concat();
    let s = sim.courant();
    write_out(
        out,
        &[
            npz::Array {
                name: "ez",
                shape: &[ny, nx],
                data: &ez,
            },
            npz::Array {
                name: "hx",
                shape: &[ny, nx],
                data: &hx,
            },
            npz::Array {
                name: "hy",
                shape: &[ny, nx],
                data: &hy,
            },
            npz::Array {
                name: "probes",
                shape: &[n_probes, rows],
                data: &probes,
            },
            npz::Array {
                name: "courant",
                shape: &[],
                data: &[s],
            },
            npz::Array {
                name: "dt",
                shape: &[],
                data: &[s],
            },
        ],
    )?;
    println!(
        "wrote {} ({nx} x {ny} grid after {steps} steps, {n_probes} probes, S = {s})",
        out.display()
    );
    Ok(())
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    let result = match &cli.cmd {
        Cmd::Run(a) => match &a.scene {
            Some(path) => run_scene(path, a.steps, &a.out),
            None => run(a),
        },
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn run_scene_writes_fields_and_probes() {
        let dir = std::env::temp_dir().join(format!("fdtd_cli_scene_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let scene = dir.join("scene.json");
        std::fs::write(
            &scene,
            r#"{"version": 1, "grid": {"nx": 12, "ny": 9, "dx_phys_m": 0.001, "courant": 0.5},
            "materials": [{"kind": "dielectric", "eps_inf": 1, "sigma": 0}],
            "sources": [{"i": 4, "j": 4, "waveform": {"type": "ricker", "t0": 12, "f_peak": 0.1},
                "amplitude": 1}],
            "monitors": [{"type": "point", "i": 7, "j": 4}, {"type": "point", "i": 4, "j": 4}]}"#,
        )
        .unwrap();
        let out = dir.join("out.npz");
        run_scene(&scene, 30, &out).unwrap();

        let mut zip = zip::ZipArchive::new(std::fs::File::open(&out).unwrap()).unwrap();
        for (name, shape) in [
            ("ez", "(9, 12)"),
            ("hx", "(9, 12)"),
            ("hy", "(9, 12)"),
            ("probes", "(2, 31)"),
            ("courant", "()"),
            ("dt", "()"),
        ] {
            let mut bytes = Vec::new();
            zip.by_name(&format!("{name}.npy"))
                .unwrap()
                .read_to_end(&mut bytes)
                .unwrap();
            let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            let header = std::str::from_utf8(&bytes[10..10 + hlen]).unwrap();
            assert!(
                header.contains(&format!("'shape': {shape}")),
                "{name}: {header}"
            );
        }
        let bad = dir.join("bad.json");
        std::fs::write(&bad, "{}").unwrap();
        assert!(run_scene(&bad, 1, &out).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
