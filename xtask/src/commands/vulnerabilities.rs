use anyhow::Result;
use clap::{Args, Subcommand};

use crate::commands;

#[derive(Args, Clone, Debug)]
pub struct VulnerabilitiesArgs {
    #[command(subcommand)]
    pub command: Option<VulnerabilitiesSubcommand>,
}

#[derive(Clone, Debug, Subcommand)]
pub enum VulnerabilitiesSubcommand {
    /// Run cargo-careful.
    CargoCareful,
    /// Run AddressSanitizer.
    AddressSanitizer,
    /// Run ThreadSanitizer.
    ThreadSanitizer,
    /// Run MemorySanitizer.
    MemorySanitizer,
    /// Run SafeStack.
    SafeStack,
    /// Run all nightly vulnerability checks.
    All,
}

pub(crate) fn handle(args: VulnerabilitiesArgs) -> Result<()> {
    match args.command.unwrap_or(VulnerabilitiesSubcommand::All) {
        VulnerabilitiesSubcommand::CargoCareful => commands::run(
            "cargo",
            &["+nightly".into(), "careful".into(), "test".into()],
        ),
        VulnerabilitiesSubcommand::AddressSanitizer => {
            run_sanitizer("-Zsanitizer=address -Copt-level=3")
        }
        VulnerabilitiesSubcommand::ThreadSanitizer => {
            run_sanitizer("-Zsanitizer=thread -Copt-level=3")
        }
        VulnerabilitiesSubcommand::MemorySanitizer => {
            run_sanitizer("-Zsanitizer=memory -Zsanitizer-memory-track-origins -Copt-level=3")
        }
        VulnerabilitiesSubcommand::SafeStack => {
            run_sanitizer("-Zsanitizer=safestack -Copt-level=3")
        }
        VulnerabilitiesSubcommand::All => {
            handle(VulnerabilitiesArgs {
                command: Some(VulnerabilitiesSubcommand::CargoCareful),
            })?;
            handle(VulnerabilitiesArgs {
                command: Some(VulnerabilitiesSubcommand::AddressSanitizer),
            })?;
            handle(VulnerabilitiesArgs {
                command: Some(VulnerabilitiesSubcommand::ThreadSanitizer),
            })?;
            handle(VulnerabilitiesArgs {
                command: Some(VulnerabilitiesSubcommand::MemorySanitizer),
            })?;
            handle(VulnerabilitiesArgs {
                command: Some(VulnerabilitiesSubcommand::SafeStack),
            })
        }
    }
}

fn run_sanitizer(rustflags: &str) -> Result<()> {
    let envs = vec![
        ("RUSTFLAGS".to_string(), rustflags.to_string()),
        ("RUSTDOCFLAGS".to_string(), rustflags.to_string()),
    ];
    commands::run_with_env(
        "cargo",
        &[
            "+nightly".into(),
            "test".into(),
            "-Zbuild-std".into(),
            "--target".into(),
            "x86_64-unknown-linux-gnu".into(),
            "--".into(),
            "--nocapture".into(),
        ],
        &envs,
    )
}
