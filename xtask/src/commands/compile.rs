use anyhow::Result;
use clap::Args;

use crate::{
    commands::build::{self, BuildArgs},
    Context,
};

#[derive(Args, Clone, Debug, Default)]
pub struct CompileArgs {
    #[command(flatten)]
    pub build: BuildArgs,
}

pub(crate) fn handle(args: CompileArgs, context: Context) -> Result<()> {
    build::handle(args.build, context)
}
