//! `fdtd`: command line front end for the reference solver.

mod npz;

use clap::{Parser, Subcommand, ValueEnum};
use fdtd_core::FdtdError;
use fdtd_core::cpml::PmlSpec;
use fdtd_core::fdtd1d::{Boundary1d, Sim1d, Source1d};
use fdtd_core::grid::Component;
use fdtd_core::kernel::Kernel;
use fdtd_core::reflection::{ReflectionSetup, reflection_db};
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
    /// Measure CPML reflection over a grid of layer parameters with the standard
    /// angle sweep, against one shared reference run, and write a CSV sorted by
    /// the worst reflection from 0 to 70 degrees, best first. Run it in release.
    PmlSweep(SweepArgs),
}

#[derive(clap::Args)]
struct SweepArgs {
    /// Layer thicknesses in cells.
    #[arg(long, value_delimiter = ',', default_values_t = [10, 16, 20])]
    thickness: Vec<usize>,
    /// Grading orders.
    #[arg(long, value_delimiter = ',', default_values_t = [2.0, 3.0, 4.0])]
    m: Vec<f64>,
    /// sigma_max as a multiple of sigma_opt.
    #[arg(long, value_delimiter = ',', default_values_t = [0.6, 0.8, 1.0, 1.2, 1.5])]
    sigma_scale: Vec<f64>,
    #[arg(long, value_delimiter = ',', default_values_t = [1.0, 5.0, 10.0, 15.0])]
    kappa_max: Vec<f64>,
    /// Normalized like a conductivity.
    #[arg(long, value_delimiter = ',', default_values_t = [0.0, 0.05, 0.1, 0.2])]
    alpha_max: Vec<f64>,
    #[arg(long, default_value = "out/pml_sweep.csv")]
    out: PathBuf,
}

/// Every combination of the sweep values, thickness outermost.
fn sweep_specs(a: &SweepArgs) -> Vec<PmlSpec> {
    let mut specs = Vec::new();
    for &thickness in &a.thickness {
        for &m in &a.m {
            for &sigma_scale in &a.sigma_scale {
                for &kappa_max in &a.kappa_max {
                    for &alpha_max in &a.alpha_max {
                        specs.push(PmlSpec {
                            thickness,
                            m,
                            sigma_scale,
                            kappa_max,
                            alpha_max,
                        });
                    }
                }
            }
        }
    }
    specs
}

/// Reflection per probe for each spec, measured in parallel against one
/// reference. Threads only split the list of runs; each run is the plain
/// single-threaded reference kernel.
fn measure_all(
    setup: &ReflectionSetup,
    reference: &[Vec<f64>],
    specs: &[PmlSpec],
) -> Result<Vec<Vec<f64>>, FdtdError> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let chunk = specs.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = specs
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    part.iter()
                        .map(|&p| {
                            let test = setup.test_domain(Some(p))?;
                            Ok(reflection_db(&test, reference))
                        })
                        .collect::<Result<Vec<_>, FdtdError>>()
                })
            })
            .collect();
        let mut all = Vec::with_capacity(specs.len());
        for h in handles {
            all.extend(h.join().map_err(|_| FdtdError::InvalidParameter {
                name: "sweep thread panicked",
                value: f64::NAN,
            })??);
        }
        Ok(all)
    })
}

fn pml_sweep(a: &SweepArgs) -> Result<(), Box<dyn std::error::Error>> {
    let setup = ReflectionSetup::standard();
    let specs = sweep_specs(a);
    for p in &specs {
        p.validate()?;
    }
    eprintln!("reference run, then {} layer runs", specs.len());
    let reference = setup.reference()?;
    let results = measure_all(&setup, &reference, &specs)?;

    let mut rows: Vec<(f64, &PmlSpec, &Vec<f64>)> = specs
        .iter()
        .zip(&results)
        .map(|(p, db)| (setup.worst_db(db, 70.0), p, db))
        .collect();
    rows.sort_by(|x, y| x.0.total_cmp(&y.0));

    let angles: Vec<String> = (0..setup.probes.len())
        .map(|k| format!("db_at_{:.1}deg", setup.angle_deg(k)))
        .collect();
    let mut csv = format!(
        "thickness,m,sigma_scale,kappa_max,alpha_max,worst_0_70_db,{}\n",
        angles.join(",")
    );
    for (worst, p, db) in &rows {
        let cells: Vec<String> = db.iter().map(|v| format!("{v:.2}")).collect();
        csv += &format!(
            "{},{},{},{},{},{worst:.2},{}\n",
            p.thickness,
            p.m,
            p.sigma_scale,
            p.kappa_max,
            p.alpha_max,
            cells.join(",")
        );
    }
    if let Some(dir) = a.out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&a.out, csv)?;
    for &t in &a.thickness {
        if let Some((worst, p, _)) = rows.iter().find(|r| r.1.thickness == t) {
            println!("best at {t} cells: {worst:.2} dB with {p:?}");
        }
    }
    println!("wrote {}", a.out.display());
    Ok(())
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
        Cmd::PmlSweep(a) => pml_sweep(a),
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

    /// Every committed example scene parses and validates.
    #[test]
    fn example_scenes_are_valid() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../validation/scenes");
        let mut n = 0;
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "json") {
                let text = std::fs::read_to_string(&path).unwrap();
                Scene::from_json(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                n += 1;
            }
        }
        assert!(n >= 2, "found {n} scenes");
    }

    #[test]
    fn sweep_covers_every_combination() {
        let cli = Cli::parse_from([
            "fdtd",
            "pml-sweep",
            "--thickness",
            "10,16",
            "--m",
            "3",
            "--sigma-scale",
            "0.8,1",
            "--alpha-max",
            "0,0.1,0.2",
        ]);
        let Cmd::PmlSweep(a) = cli.cmd else {
            panic!("expected pml-sweep")
        };
        let specs = sweep_specs(&a);
        // 2 thicknesses, 1 m, 2 scales, 4 default kappas, 3 alphas.
        assert_eq!(specs.len(), 2 * 2 * 4 * 3);
        assert_eq!(specs[0].thickness, 10);
        assert_eq!(specs[specs.len() - 1].thickness, 16);
        assert_eq!(specs[specs.len() - 1].alpha_max, 0.2);
    }

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
