//! `fdtd`: command line front end for the reference solver.

mod npz;

use clap::{Parser, Subcommand, ValueEnum};
use fdtd_core::fdtd1d::{Boundary1d, Sim1d, Source1d};
use fdtd_core::source::{SourceKind, Waveform};
use std::path::PathBuf;

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
    /// Run the 1D solver with a Gaussian pulse and dump the space-time Ez field to NPZ.
    /// All quantities are normalized: dx = 1, c0 = 1, dt = S.
    Run(RunArgs),
}

#[derive(clap::Args)]
struct RunArgs {
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

    if let Some(dir) = a.out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let s = sim.courant();
    npz::write_npz(
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

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    let result = match &cli.cmd {
        Cmd::Run(a) => run(a),
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
