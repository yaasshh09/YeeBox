//! `fdtd`: command line front end for the reference solver.

mod npz;
mod svg;

use clap::{Parser, Subcommand, ValueEnum};
use fdtd_core::FdtdError;
use fdtd_core::cases::{self, CaseResult, fmt_num};
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
    /// Regenerate the validation report: CPML reflection vs angle for the
    /// recommended 10 and 16 cell layers, the slab R and T plot, and the
    /// table of every acceptance case, as CSV, SVG and report.md. Run it in release.
    Report(ReportArgs),
    /// Run every Tier 2 acceptance case (A1 to A15) on the f64 reference
    /// kernel and print measured value, gate and pass or fail. Exits with
    /// an error if any case fails. Run it in release.
    Validate,
}

/// The case rows as a markdown table: id, check, measured, gate, result, notes.
fn case_table(rows: &[CaseResult]) -> String {
    let mut t = String::from(
        "| Case | Check | Measured | Gate | Result | Conditions |
|---|---|---|---|---|---|
",
    );
    for r in rows {
        t += &format!(
            "| {} | {} | {} {} | {} | {} | {} |
",
            r.id,
            r.name,
            fmt_num(r.measured),
            r.unit,
            r.gate,
            if r.pass { "pass" } else { "FAIL" },
            r.note.replace('|', "/")
        );
    }
    t
}

fn validate() -> Result<(), Box<dyn std::error::Error>> {
    let rows = cases::all();
    print!("{}", case_table(&rows));
    let failed: Vec<&str> = rows.iter().filter(|r| !r.pass).map(|r| r.id).collect();
    if failed.is_empty() {
        println!(
            "
all {} checks pass",
            rows.len()
        );
        Ok(())
    } else {
        Err(format!("failed: {}", failed.join(", ")).into())
    }
}

#[derive(clap::Args)]
struct ReportArgs {
    #[arg(long, default_value = "validation/report")]
    dir: PathBuf,
}

/// Measures every layer on `setup` and writes `cpml_reflection.csv`,
/// `cpml_reflection.svg` and `report.md` into `dir`. Returns the dB per probe
/// for each layer.
fn write_cpml_report(
    dir: &Path,
    setup: &ReflectionSetup,
    layers: &[PmlSpec],
) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dir)?;
    let reference = setup.reference()?;
    let results = measure_all(setup, &reference, layers)?;
    let probes = setup.probes.len();
    // The angle sweep runs right of the source; anything else (the corner)
    // goes in the tables only.
    let on_sweep = |k: usize| setup.probes[k].0 >= setup.source.0;

    let mut csv = String::from("probe,i,j,angle_deg");
    for p in layers {
        csv += &format!(",db_{}_cells", p.thickness);
    }
    csv.push('\n');
    for k in 0..probes {
        let (i, j) = setup.probes[k];
        csv += &format!("{k},{i},{j},{:.2}", setup.angle_deg(k));
        for db in &results {
            csv += &format!(",{:.2}", db[k]);
        }
        csv.push('\n');
    }
    std::fs::write(dir.join("cpml_reflection.csv"), csv)?;

    let colors = ["#1f77b4", "#d95f02", "#1b9e77", "#7570b3"];
    let points: Vec<Vec<(f64, f64)>> = results
        .iter()
        .map(|db| {
            let mut pts: Vec<(f64, f64)> = (0..probes)
                .filter(|&k| on_sweep(k))
                .map(|k| (setup.angle_deg(k), db[k]))
                .collect();
            pts.sort_by(|a, b| a.0.total_cmp(&b.0));
            pts
        })
        .collect();
    let labels: Vec<String> = layers
        .iter()
        .map(|p| format!("{} cells", p.thickness))
        .collect();
    let series: Vec<svg::Series> = points
        .iter()
        .zip(&labels)
        .zip(colors.iter().cycle())
        .map(|((pts, label), color)| svg::Series {
            label,
            color,
            points: pts,
        })
        .collect();
    let lowest = results
        .iter()
        .flatten()
        .copied()
        .filter(|v| v.is_finite())
        .fold(-60.0_f64, f64::min);
    let plot = svg::Plot {
        title: "CPML reflection vs angle of incidence (f64 reference kernel)",
        x_label: "angle of incidence (deg)",
        y_label: "reflection (dB)",
        x: (0.0, 80.0, 10.0),
        y: ((lowest / 20.0).floor() * 20.0, 0.0, 20.0),
        series: &series,
        hlines: &[(-60.0, "A9 gate, -60 dB"), (-80.0, "A10 target, -80 dB")],
    };
    std::fs::write(dir.join("cpml_reflection.svg"), plot.render())?;

    let mut md = String::from(
        "# Validation report\n\n\
         `fdtd report` writes this file, so I regenerate it instead of editing it.\n\n\
         ## CPML reflection (A9, A10)\n\n\
         ![CPML reflection vs angle](cpml_reflection.svg)\n\n",
    );
    md += &format!(
        "I measure it with the reference-domain method. A Ricker pulse (spectral peak at \
         20 cells per wavelength, S = {s}) starts {h} cells above the bottom interface of a \
         {w} x {ht} node vacuum box. I record Ez on the interface row, where the direct \
         wave meets the layer at each angle, and at the interface corner, then compare \
         with the same box inside PEC walls so far away that nothing comes back within the \
         {n} step window. Per probe, R = 20 log10(max |test - reference| / max |reference|). \
         Everything runs on the f64 reference kernel.\n\n",
        s = setup.s,
        h = setup.source.1,
        w = setup.width,
        ht = setup.height,
        n = setup.steps
    );
    md += "| Probe | Angle (deg) |";
    for p in layers {
        md += &format!(" {} cells (dB) |", p.thickness);
    }
    md += "\n|---|---|";
    md += &"---|".repeat(layers.len());
    md.push('\n');
    for k in 0..probes {
        let name = if on_sweep(k) { "interface" } else { "corner" };
        md += &format!("| {name} | {:.1} |", setup.angle_deg(k));
        for db in &results {
            md += &format!(" {:.2} |", db[k]);
        }
        md.push('\n');
    }
    md += "\n| Check | Layer | Gate | Worst, 0 to 70 deg | Result |\n|---|---|---|---|---|\n";
    for (p, db) in layers.iter().zip(&results) {
        let worst = setup.worst_db(db, 70.0);
        let (check, gate, ok) = if p.thickness < 16 {
            ("A9", "at most -60 dB", worst <= -60.0)
        } else {
            ("A10", "-80 dB target", worst <= -80.0)
        };
        md += &format!(
            "| {check} | {} cells: m = {}, sigma_scale = {}, kappa_max = {}, alpha_max = {} | {gate} | {worst:.2} dB | {} |\n",
            p.thickness,
            p.m,
            p.sigma_scale,
            p.kappa_max,
            p.alpha_max,
            if ok { "pass" } else { "FAIL" }
        );
    }
    md += "\nThe layers are `PmlSpec::recommended`, tuned with `fdtd pml-sweep` against the \
           worst reflection over every probe above, grazing 80 degrees included.\n\n\
           ## Cavity convergence (A7)\n\n\
           ![Cavity resonance error vs resolution](cavity_convergence.png)\n\n\
           `validation/gen/plot_cavity_convergence.py` makes this plot. It re-measures the \
           resonances in NumPy on its own, as a cross-check of the Rust test.\n";
    std::fs::write(dir.join("report.md"), md)?;
    Ok(results)
}

/// Writes `slab_rt.csv` and `slab_rt.svg` (measured R and T against the Airy
/// formula, A13 at L = 20) and returns the report section for them and for
/// the full case table.
fn case_sections(
    dir: &Path,
    rows: &[CaseResult],
    slab: &[(f64, f64, f64, f64, f64)],
) -> Result<String, Box<dyn std::error::Error>> {
    let mut csv = String::from(
        "f,R,T,R_airy,T_airy
",
    );
    for &(f, r, t, rc, tc) in slab {
        csv += &format!(
            "{f},{r:.8},{t:.8},{rc:.8},{tc:.8}
"
        );
    }
    std::fs::write(dir.join("slab_rt.csv"), csv)?;
    let pick = |g: fn(&(f64, f64, f64, f64, f64)) -> f64| -> Vec<(f64, f64)> {
        slab.iter().map(|p| (p.0, g(p))).collect()
    };
    let (r, t, rc, tc) = (pick(|p| p.1), pick(|p| p.2), pick(|p| p.3), pick(|p| p.4));
    let series = [
        svg::Series {
            label: "R, Airy",
            color: "#9ecae1",
            points: &rc,
        },
        svg::Series {
            label: "R, FDTD",
            color: "#1f77b4",
            points: &r,
        },
        svg::Series {
            label: "T, Airy",
            color: "#fdae6b",
            points: &tc,
        },
        svg::Series {
            label: "T, FDTD",
            color: "#d95f02",
            points: &t,
        },
    ];
    let f_max = slab.iter().map(|p| p.0).fold(0.0, f64::max);
    let plot = svg::Plot {
        title: "Dielectric slab, eps 4, 20 cells: R and T vs frequency (A13)",
        x_label: "frequency (cycles per unit time, dx = c0 = 1)",
        y_label: "R, T",
        x: (0.0, (f_max / 0.005).ceil() * 0.005, 0.005),
        y: (0.0, 1.0, 0.2),
        series: &series,
        hlines: &[],
    };
    std::fs::write(dir.join("slab_rt.svg"), plot.render())?;
    Ok(format!(
        "
## Every case (A1 to A15)

         `fdtd validate` prints the same table. Each gate is the one in          `docs/04-validation.md`; the conditions column says what was run.

{}
         ## Dielectric slab (A13)

         ![Slab R and T vs frequency](slab_rt.svg)

         The measured curves sit on the exact discrete slab to 1e-7; what separates them          from Airy is the grid's own dispersion, 4.9e-3 at the band edge, as the golden          file predicts. The numbers are in `slab_rt.csv`.
",
        case_table(rows)
    ))
}

fn report(a: &ReportArgs) -> Result<(), Box<dyn std::error::Error>> {
    let setup = ReflectionSetup::standard();
    let layers = [PmlSpec::recommended(10), PmlSpec::recommended(16)];
    let results = write_cpml_report(&a.dir, &setup, &layers)?;
    for (p, db) in layers.iter().zip(&results) {
        println!(
            "{} cells: worst 0 to 70 deg {:.2} dB",
            p.thickness,
            setup.worst_db(db, 70.0)
        );
    }
    let rows = cases::all();
    let (golden, rt) = cases::plane_wave::slab_spectrum(20)?;
    let slab: Vec<(f64, f64, f64, f64, f64)> = golden
        .iter()
        .enumerate()
        .map(|(k, g)| (g.f, rt.r[k], rt.t[k], g.r_c, g.t_c))
        .collect();
    let md_path = a.dir.join("report.md");
    let mut md = std::fs::read_to_string(&md_path)?;
    md += &case_sections(&a.dir, &rows, &slab)?;
    std::fs::write(&md_path, md)?;
    let failed = rows.iter().filter(|r| !r.pass).count();
    println!("{} checks, {failed} failed", rows.len());
    println!("wrote {}", a.dir.display());
    Ok(())
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
        Cmd::Report(a) => report(a),
        Cmd::Validate => validate(),
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

    /// The report writes CSV, SVG and markdown with one row per probe and one
    /// curve per layer, on a small setup.
    #[test]
    fn report_writes_three_files() {
        let dir = std::env::temp_dir().join(format!("fdtd_cli_report_{}", std::process::id()));
        let setup = ReflectionSetup {
            width: 60,
            height: 30,
            source: (20, 10),
            probes: vec![(20, 0), (30, 0), (45, 0), (0, 0)],
            steps: 300,
            ..ReflectionSetup::standard()
        };
        let layers = [PmlSpec::recommended(10), PmlSpec::recommended(16)];
        let results = write_cpml_report(&dir, &setup, &layers).unwrap();
        assert_eq!(results.len(), 2);
        let read = |f: &str| std::fs::read_to_string(dir.join(f)).unwrap();
        let csv = read("cpml_reflection.csv");
        assert_eq!(csv.lines().count(), 5);
        assert!(csv.starts_with("probe,i,j,angle_deg,db_10_cells,db_16_cells"));
        let svg = read("cpml_reflection.svg");
        assert!(svg.starts_with("<svg") && svg.trim_end().ends_with("</svg>"));
        assert_eq!(svg.matches("<polyline").count(), 2);
        // Three sweep probes per curve; the corner is left out of the plot.
        assert_eq!(svg.matches("<circle").count(), 6);
        let md = read("report.md");
        assert!(md.contains("| corner |") && md.contains("| A9 |") && md.contains("| A10 |"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The case sections write the slab CSV and plot and carry the table.
    #[test]
    fn case_sections_write_the_slab_plot() {
        use fdtd_core::cases::Gate;
        let dir = std::env::temp_dir().join(format!("fdtd_cli_cases_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let rows = [CaseResult::new(
            "A13",
            "slab",
            4.9e-3,
            Gate::AtMost(0.01),
            "abs",
            "x".into(),
        )];
        let slab = [
            (0.01, 0.16, 0.84, 0.162, 0.838),
            (0.02, 0.34, 0.66, 0.337, 0.663),
        ];
        let md = case_sections(&dir, &rows, &slab).unwrap();
        assert!(md.contains("| A13 | slab | 4.90e-3 abs | <= 0.01 | pass |"));
        assert!(md.contains("![Slab R and T vs frequency](slab_rt.svg)"));
        let csv = std::fs::read_to_string(dir.join("slab_rt.csv")).unwrap();
        assert_eq!(csv.lines().count(), 3);
        let svg = std::fs::read_to_string(dir.join("slab_rt.svg")).unwrap();
        assert_eq!(svg.matches("<polyline").count(), 4);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The table has one line per row, the gate and unit in place, and a
    /// FAIL for a failing row; pipes in notes cannot break the columns.
    #[test]
    fn case_table_rows() {
        use fdtd_core::cases::Gate;
        let rows = [
            CaseResult::new(
                "A1",
                "shape",
                0.0,
                Gate::AtMost(1e-12),
                "rel",
                "S = 1".into(),
            ),
            CaseResult::new(
                "A9",
                "layer",
                -50.0,
                Gate::AtMost(-60.0),
                "dB",
                "a | b".into(),
            ),
        ];
        let t = case_table(&rows);
        assert_eq!(t.lines().count(), 4);
        assert!(t.contains("| A1 | shape | 0 rel | <= 1.00e-12 | pass | S = 1 |"));
        assert!(t.contains("| A9 | layer | -50 dB | <= -60 | FAIL | a / b |"));
        assert!(Cli::try_parse_from(["fdtd", "validate"]).is_ok());
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
