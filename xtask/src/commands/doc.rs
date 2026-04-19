use anyhow::Result;
use clap::{Args, Subcommand};

use crate::{commands, Context};

#[derive(Args, Clone, Debug)]
pub struct DocArgs {
    #[command(subcommand)]
    pub command: DocSubcommand,

    #[command(flatten)]
    pub workspace: commands::WorkspaceArgs,
}

#[derive(Clone, Debug, Subcommand, PartialEq, Eq)]
pub enum DocSubcommand {
    /// Build workspace docs.
    Build,
    /// Run doc tests.
    Tests,
}

pub(crate) fn handle(args: DocArgs, context: Context) -> Result<()> {
    match context {
        Context::Std => handle_std(args),
        Context::NoStd => {
            commands::print_placeholder(
                "no-std documentation pipeline is pre-reserved and currently a no-op",
            );
            Ok(())
        }
        Context::All => {
            let args_clone = args.clone();
            handle_std(args_clone)?;
            commands::print_placeholder(
                "no-std documentation pipeline is pre-reserved and currently a no-op",
            );
            Ok(())
        }
    }
}

fn handle_std(args: DocArgs) -> Result<()> {
    match args.command {
        DocSubcommand::Build => {
            let mut cargo_args = vec!["doc".into()];
            commands::append_workspace_args(&mut cargo_args, &args.workspace);
            cargo_args.push("--no-deps".into());
            commands::run("cargo", &cargo_args)
        }
        DocSubcommand::Tests => {
            let mut cargo_args = vec!["test".into()];
            commands::append_workspace_args(&mut cargo_args, &args.workspace);
            cargo_args.push("--doc".into());
            commands::run("cargo", &cargo_args)
        }
    }
}
