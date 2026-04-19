use anyhow::Result;
use clap::Args;

use crate::{commands, Context};

#[derive(Args, Clone, Debug, Default)]
pub struct BuildArgs {
    #[command(flatten)]
    pub workspace: commands::WorkspaceArgs,

    /// Use the CI-oriented build profile.
    #[arg(long)]
    pub ci: bool,
}

pub(crate) fn handle(args: BuildArgs, context: Context) -> Result<()> {
    match context {
        Context::Std => build_std(&args),
        Context::NoStd => {
            commands::print_placeholder(
                "no-std build pipeline is pre-reserved and will be wired when no-std targets land",
            );
            Ok(())
        }
        Context::All => {
            build_std(&args)?;
            commands::print_placeholder(
                "no-std build pipeline is pre-reserved and will be wired when no-std targets land",
            );
            Ok(())
        }
    }
}

fn build_std(args: &BuildArgs) -> Result<()> {
    let mut cargo_args = vec!["build".into()];
    commands::append_workspace_args(&mut cargo_args, &args.workspace);

    if args.ci {
        eprintln!("info: running std build with CI profile");
    }

    commands::run("cargo", &cargo_args)
}
