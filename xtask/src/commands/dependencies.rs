use anyhow::Result;
use clap::{Args, Subcommand};

use crate::commands;

#[derive(Args, Clone, Debug)]
pub struct DependenciesArgs {
    #[command(subcommand)]
    pub command: Option<DependenciesSubcommand>,
}

#[derive(Clone, Debug, Subcommand)]
pub enum DependenciesSubcommand {
    /// Run cargo-audit.
    Audit,
    /// Run cargo-deny checks.
    Deny,
    /// Run cargo-udeps.
    Udeps,
    /// Run the full dependency suite.
    All,
}

pub(crate) fn handle(args: DependenciesArgs) -> Result<()> {
    match args.command.unwrap_or(DependenciesSubcommand::All) {
        DependenciesSubcommand::Audit => commands::run("cargo", &["audit".into()]),
        DependenciesSubcommand::Deny => {
            commands::run("cargo", &["deny".into(), "check".into(), "licenses".into()])?;
            commands::run(
                "cargo",
                &[
                    "deny".into(),
                    "check".into(),
                    "bans".into(),
                    "sources".into(),
                ],
            )
        }
        DependenciesSubcommand::Udeps => commands::run(
            "cargo",
            &["+nightly".into(), "udeps".into(), "--all-targets".into()],
        ),
        DependenciesSubcommand::All => {
            handle(DependenciesArgs {
                command: Some(DependenciesSubcommand::Audit),
            })?;
            handle(DependenciesArgs {
                command: Some(DependenciesSubcommand::Deny),
            })?;
            handle(DependenciesArgs {
                command: Some(DependenciesSubcommand::Udeps),
            })
        }
    }
}
