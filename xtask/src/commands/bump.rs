use anyhow::Result;
use clap::Args;

use crate::commands;

#[derive(Args, Clone, Debug, Default)]
pub struct BumpArgs {
    /// Reserved target version.
    #[arg(long)]
    pub version: Option<String>,
}

pub(crate) fn handle(args: BumpArgs) -> Result<()> {
    let suffix = args
        .version
        .as_deref()
        .map(|version| format!(" for version {version}"))
        .unwrap_or_default();
    commands::print_placeholder(&format!(
        "version bump automation is reserved and not wired yet{suffix}"
    ));
    Ok(())
}
