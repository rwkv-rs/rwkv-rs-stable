use anyhow::Result;
use clap::{Args, Subcommand};

use crate::commands;

#[derive(Args, Clone, Debug)]
pub struct CheckArgs {
    #[command(subcommand)]
    pub command: CheckSubcommand,
}

#[derive(Clone, Debug, Subcommand)]
pub enum CheckSubcommand {
    /// Run cargo-audit.
    Audit,
    /// Run rustfmt in check mode.
    Format,
    /// Run clippy with warnings denied.
    Lint,
    /// Run typos.
    Typos,
    /// Run the full check suite.
    All,
}

pub(crate) fn handle(args: CheckArgs) -> Result<()> {
    match args.command {
        CheckSubcommand::Audit => commands::run("cargo", &["audit".into()]),
        CheckSubcommand::Format => commands::run(
            "cargo",
            &[
                "+nightly".into(),
                "fmt".into(),
                "--all".into(),
                "--check".into(),
            ],
        ),
        CheckSubcommand::Lint => commands::run(
            "cargo",
            &[
                "clippy".into(),
                "--workspace".into(),
                "--all-targets".into(),
                "--".into(),
                "-D".into(),
                "warnings".into(),
            ],
        ),
        CheckSubcommand::Typos => commands::run("typos", &[]),
        CheckSubcommand::All => {
            handle(CheckArgs {
                command: CheckSubcommand::Audit,
            })?;
            handle(CheckArgs {
                command: CheckSubcommand::Format,
            })?;
            handle(CheckArgs {
                command: CheckSubcommand::Lint,
            })?;
            handle(CheckArgs {
                command: CheckSubcommand::Typos,
            })
        }
    }
}
