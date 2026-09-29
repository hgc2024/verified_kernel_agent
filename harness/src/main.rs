//! `kharness` CLI. Every subcommand prints exactly one JSON object to stdout; logs go to stderr.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use kharness::{compile, guards, profile, report::RunReport, task::Task, timing, verify};

#[derive(Debug, Parser)]
#[command(name = "kharness", version, about = "Compile, verify, and time candidate CUDA kernels")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Args)]
struct Target {
    /// Path to the task TOML (e.g. tasks/level1/gemm_fp32.toml).
    #[arg(long)]
    task: PathBuf,
    /// Path to the candidate .cu implementing the C ABI in kernels/include/vka/kernels.h.
    #[arg(long)]
    candidate: PathBuf,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Compile the candidate with nvcc and capture diagnostics.
    Compile {
        #[command(flatten)]
        target: Target,
    },
    /// Check correctness over the nominal shape plus fuzzed shapes and seeds.
    Verify {
        #[command(flatten)]
        target: Target,
        #[arg(long, default_value_t = 20)]
        trials: u32,
        /// Seed trials from the OS RNG instead of a fixed seed.
        #[arg(long)]
        seed_from_os: bool,
    },
    /// Time the candidate against the task baseline.
    Bench {
        #[command(flatten)]
        target: Target,
        #[arg(long, default_value_t = 10)]
        warmup: u32,
        #[arg(long, default_value_t = 100)]
        iters: u32,
    },
    /// Collect Nsight Compute metrics for the candidate.
    Profile {
        #[command(flatten)]
        target: Target,
        #[arg(long, default_value = "default")]
        metrics: String,
    },
    /// compile + verify + guards + bench.
    Run {
        #[command(flatten)]
        target: Target,
    },
}

impl Command {
    fn target(&self) -> &Target {
        match self {
            Command::Compile { target }
            | Command::Verify { target, .. }
            | Command::Bench { target, .. }
            | Command::Profile { target, .. }
            | Command::Run { target } => target,
        }
    }
}

fn dispatch(command: &Command) -> Result<RunReport> {
    let target = command.target();
    let task = Task::load(&target.task)?;
    let candidate = target.candidate.as_path();
    let mut report = RunReport::new(&task.id, &candidate.display().to_string());

    match command {
        Command::Compile { .. } => {
            report.compile = Some(compile::compile(&task, candidate)?);
        }
        Command::Verify {
            trials,
            seed_from_os,
            ..
        } => {
            let opts = verify::VerifyOptions {
                trials: *trials,
                seed_from_os: *seed_from_os,
            };
            report.verify = Some(verify::verify(&task, candidate, &opts)?);
        }
        Command::Bench { warmup, iters, .. } => {
            let opts = timing::BenchOptions {
                warmup: *warmup,
                iters: *iters,
            };
            report.timing = Some(timing::bench(&task, candidate, &opts)?);
        }
        Command::Profile { metrics, .. } => {
            let opts = profile::ProfileOptions {
                metrics: metrics.clone(),
            };
            report.profile = Some(profile::profile(&task, candidate, &opts)?);
        }
        Command::Run { .. } => {
            report.compile = Some(compile::compile(&task, candidate)?);
            let verify_opts = verify::VerifyOptions::default();
            report.verify = Some(verify::verify(&task, candidate, &verify_opts)?);
            report.guards = guards::run_all(&task, candidate)?;
            let bench_opts = timing::BenchOptions::default();
            report.timing = Some(timing::bench(&task, candidate, &bench_opts)?);
        }
    }
    Ok(report)
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match dispatch(&cli.command) {
        Ok(report) => match serde_json::to_string(&report) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("kharness: failed to serialize report: {err}");
                ExitCode::FAILURE
            }
        },
        Err(err) => {
            let message = format!("{err:#}");
            eprintln!("kharness: {message}");
            let target = cli.command.target();
            let payload = serde_json::json!({
                "task": target.task.display().to_string(),
                "candidate": target.candidate.display().to_string(),
                "error": message,
            });
            println!("{payload}");
            ExitCode::FAILURE
        }
    }
}
