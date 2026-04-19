use std::{fs, path::PathBuf};

use anyhow::Result;
use clap::Args;

use crate::commands::{self, WorkspacePackage};

#[derive(Args, Clone, Debug, Default)]
pub struct BenchArgs {
    /// Run in CI mode and request bencher-compatible output.
    #[arg(long)]
    pub ci: bool,

    /// Restrict benchmark execution to a single package.
    #[arg(long = "crate")]
    pub package: Option<String>,

    /// Restrict benchmark execution to a single bench target.
    #[arg(long)]
    pub bench: Option<String>,

    /// Write CI output to a file for later workflow processing.
    #[arg(long)]
    pub output: Option<PathBuf>,
}

pub(crate) fn handle(args: BenchArgs) -> Result<()> {
    let packages = commands::workspace_packages()?;
    let discovered = discover_benches(&packages, &args);

    if discovered.is_empty() {
        commands::print_placeholder(
            "benchmark command surface is reserved; no bench targets are present yet",
        );
        return Ok(());
    }

    let mut combined_output = String::new();

    for (package, bench) in discovered {
        let mut cargo_args = vec![
            "bench".into(),
            "-p".into(),
            package.name.clone(),
            "--bench".into(),
            bench.clone(),
        ];

        if args.ci {
            cargo_args.push("--".into());
            cargo_args.push("--output-format".into());
            cargo_args.push("bencher".into());
            combined_output.push_str(&commands::capture("cargo", &cargo_args)?);
        } else {
            commands::run("cargo", &cargo_args)?;
        }
    }

    if let Some(path) = args.output {
        if args.ci {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(path, combined_output)?;
        } else {
            commands::print_placeholder(
                "benchmark output files are only emitted in `--ci` mode; skipping file write",
            );
        }
    }

    Ok(())
}

fn discover_benches<'a>(
    packages: &'a [WorkspacePackage],
    args: &BenchArgs,
) -> Vec<(&'a WorkspacePackage, String)> {
    packages
        .iter()
        .filter(|package| package.name != "xtask")
        .filter(|package| {
            args.package
                .as_ref()
                .map(|selected| &package.name == selected)
                .unwrap_or(true)
        })
        .flat_map(|package| {
            package
                .targets
                .iter()
                .filter(|target| target.kind.iter().any(|kind| kind == "bench"))
                .filter(|target| {
                    args.bench
                        .as_ref()
                        .map(|selected| &target.name == selected)
                        .unwrap_or(true)
                })
                .map(|target| (package, target.name.clone()))
                .collect::<Vec<_>>()
        })
        .collect()
}
