use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Subcommand};

use crate::commands;

#[derive(Args, Clone, Debug)]
pub struct CoverageArgs {
    #[command(subcommand)]
    pub command: CoverageSubcommand,
}

#[derive(Clone, Debug, Subcommand)]
pub enum CoverageSubcommand {
    /// Check whether cargo-llvm-cov is available.
    Install,
    /// Generate lcov output using cargo-llvm-cov.
    Generate(CoverageGenerateArgs),
}

#[derive(Args, Clone, Debug)]
pub struct CoverageGenerateArgs {
    #[arg(long, default_value = "debug")]
    pub profile: String,

    #[arg(long)]
    pub ignore: Option<String>,

    #[arg(long, default_value = "target/coverage/lcov.info")]
    pub output: PathBuf,
}

pub(crate) fn handle(args: CoverageArgs) -> Result<()> {
    match args.command {
        CoverageSubcommand::Install => commands::ensure_tool("cargo-llvm-cov"),
        CoverageSubcommand::Generate(generate_args) => generate(generate_args),
    }
}

fn generate(args: CoverageGenerateArgs) -> Result<()> {
    let mut cargo_args = vec![
        "llvm-cov".into(),
        "nextest".into(),
        "--workspace".into(),
        "--lcov".into(),
        "--output-path".into(),
        args.output.display().to_string(),
    ];

    if args.profile == "release" {
        cargo_args.push("--release".into());
    }

    if let Some(ignore) = args.ignore {
        cargo_args.push("--ignore-filename-regex".into());
        cargo_args.push(glob_list_to_regex(&ignore));
    }

    commands::run("cargo", &cargo_args)
}

fn glob_list_to_regex(raw: &str) -> String {
    raw.split(',')
        .map(str::trim)
        .filter(|pattern| !pattern.is_empty())
        .map(glob_to_regex_fragment)
        .collect::<Vec<_>>()
        .join("|")
}

fn glob_to_regex_fragment(pattern: &str) -> String {
    let mut fragment = String::new();
    for ch in pattern.chars() {
        match ch {
            '*' => fragment.push_str(".*"),
            '.' => fragment.push_str("\\."),
            '/' => fragment.push('/'),
            other => fragment.push(other),
        }
    }
    fragment
}
