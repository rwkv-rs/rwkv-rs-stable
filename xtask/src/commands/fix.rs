use anyhow::Result;
use clap::Args;

use crate::commands;

#[derive(Args, Clone, Debug, Default)]
pub struct FixArgs {
    /// Run rustfmt in check mode instead of applying changes.
    #[arg(long)]
    pub check: bool,
}

pub(crate) fn handle(args: FixArgs) -> Result<()> {
    let mut cargo_args = vec!["+nightly".into(), "fmt".into(), "--all".into()];
    if args.check {
        cargo_args.push("--check".into());
    }
    commands::run("cargo", &cargo_args)
}
