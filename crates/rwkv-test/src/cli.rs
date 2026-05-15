use std::{
    io::{self, IsTerminal},
    path::PathBuf,
    process::ExitCode,
};

use clap::{Parser, Subcommand, ValueEnum};

use crate::{compare, compare_rwkv_nn};

#[derive(Parser)]
#[command(
    name = "rwkv-test",
    about = "Compare RWKV trace safetensors against a baseline",
    long_about = "Compare one RWKV trace case directory against one baseline case directory.\n\
The tool recursively scans --actual for .safetensors files, matches each file by \
relative path under --baseline, and reports numeric drift plus missing or extra files."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compare two RWKV trace case directories.
    Compare(CompareArgs),
    /// Run `rwkv-nn`, write an actual trace, and compare it against a baseline.
    CompareRwkvNn(CompareRwkvNnArgs),
}

/// Arguments for comparing two RWKV trace case directories.
#[derive(Parser)]
pub struct CompareArgs {
    /// Trace case directory produced by the implementation under test.
    #[arg(long)]
    pub actual: PathBuf,
    /// Baseline trace case directory with the same relative safetensors layout.
    #[arg(long)]
    pub baseline: PathBuf,
    /// Absolute-error tolerance for floating-point tensors.
    #[arg(long, default_value_t = 0.0)]
    pub atol: f64,
    /// Relative-error tolerance for floating-point tensors.
    #[arg(long, default_value_t = 0.0)]
    pub rtol: f64,
    /// Minimum cosine similarity required for floating-point tensors.
    #[arg(long, default_value_t = 1.0)]
    pub cos_min: f64,
    /// Denominator floor used when computing relative error.
    #[arg(long, default_value_t = 1e-12)]
    pub rel_eps: f64,
    /// Colorize PASS/FAIL table rows.
    #[arg(long, value_enum, default_value_t = ColorMode::Auto)]
    pub color: ColorMode,
    /// Do not fail when baseline contains files absent from actual.
    #[arg(long)]
    pub allow_extra_baseline: bool,
}

/// Arguments for running `rwkv-nn` and comparing the generated trace.
#[derive(Parser)]
pub struct CompareRwkvNnArgs {
    /// Output directory for the generated actual trace.
    #[arg(
        long,
        default_value = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../target/rwkv-test/rwkv_nn_actual/rwkv_lm/bf16/case_000000"
        )
    )]
    pub actual: PathBuf,
    /// Baseline trace case directory produced by `rwkv-lm`.
    #[arg(
        long,
        default_value = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test_data/rwkv_lm/bf16/case_000000"
        )
    )]
    pub baseline: PathBuf,
    /// Fixed RWKV LM safetensors weights.
    #[arg(
        long,
        default_value = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../weights/rwkv-init-0.1b-ctx512-test.st"
        )
    )]
    pub weights: PathBuf,
    /// Absolute-error tolerance for floating-point tensors.
    #[arg(long, default_value_t = 4.0e-2)]
    pub atol: f64,
    /// Relative-error tolerance for floating-point tensors.
    #[arg(long, default_value_t = 1.0e-2)]
    pub rtol: f64,
    /// Minimum cosine similarity required for floating-point tensors.
    #[arg(long, default_value_t = 0.999)]
    pub cos_min: f64,
    /// Denominator floor used when computing relative error.
    #[arg(long, default_value_t = 1e-12)]
    pub rel_eps: f64,
    /// Colorize PASS/FAIL table rows.
    #[arg(long, value_enum, default_value_t = ColorMode::Auto)]
    pub color: ColorMode,
    /// Number of measured forward samples for module timing.
    #[arg(long, default_value_t = 3)]
    pub repeat: usize,
    /// Number of unmeasured warmup forwards before timing.
    #[arg(long, default_value_t = 1)]
    pub warmup: usize,
}

/// Terminal color mode for CLI comparison output.
#[derive(Clone, Copy, ValueEnum)]
pub enum ColorMode {
    /// Colorize when stdout is a terminal.
    Auto,
    /// Always emit ANSI color escapes.
    Always,
    /// Never emit ANSI color escapes.
    Never,
}

/// Runs the command line interface.
pub fn run_cli() -> ExitCode {
    let result = match Cli::parse().command {
        Command::Compare(args) => compare(args),
        Command::CompareRwkvNn(args) => compare_rwkv_nn(args),
    };
    match result {
        Ok(true) => ExitCode::from(1),
        Ok(false) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::from(2)
        }
    }
}

impl ColorMode {
    pub(crate) fn use_color(self) -> bool {
        match self {
            Self::Auto => io::stdout().is_terminal(),
            Self::Always => true,
            Self::Never => false,
        }
    }
}
