use anyhow::Result;
use clap::{Args, ValueEnum};

use crate::{commands, Context};

#[derive(Args, Clone, Debug, Default)]
pub struct TestArgs {
    #[command(flatten)]
    pub workspace: commands::WorkspaceArgs,

    /// CI runner profile to emulate.
    #[arg(long, value_enum)]
    pub ci: Option<CiTestType>,
}

#[derive(Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum CiTestType {
    #[value(name = "github-runner")]
    Github,
    #[value(name = "github-mac-runner")]
    GithubMac,
    #[value(name = "gcp-cuda-runner")]
    GcpCuda,
    #[value(name = "gcp-vulkan-runner")]
    GcpVulkan,
    #[value(name = "gcp-wgpu-runner")]
    GcpWgpu,
}

pub(crate) fn handle(args: TestArgs, context: Context) -> Result<()> {
    match context {
        Context::Std => handle_std(args),
        Context::NoStd => {
            commands::print_placeholder(
                "no-std test profile is pre-reserved and will be enabled with real no-std crates",
            );
            Ok(())
        }
        Context::All => {
            let args_clone = args.clone();
            handle_std(args_clone)?;
            commands::print_placeholder(
                "no-std test profile is pre-reserved and will be enabled with real no-std crates",
            );
            Ok(())
        }
    }
}

fn handle_std(args: TestArgs) -> Result<()> {
    if let Some(profile) = &args.ci {
        match profile {
            CiTestType::Github | CiTestType::GithubMac => {
                eprintln!("info: running workspace nextest for CI profile `{profile:?}`");
            }
            CiTestType::GcpCuda | CiTestType::GcpVulkan | CiTestType::GcpWgpu => {
                eprintln!(
                    "info: GPU CI profile `{profile:?}` is reserved; current workspace has no dedicated GPU-only test targets, so this run executes the shared nextest suite on the selected runner"
                );
            }
        }
    }

    let mut cargo_args = vec!["nextest".into(), "run".into()];
    commands::append_workspace_args(&mut cargo_args, &args.workspace);
    commands::run("cargo", &cargo_args)
}
