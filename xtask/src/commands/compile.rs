use anyhow::Result;
use clap::Args;

use crate::{
    Context,
    commands::build::{self, BuildArgs},
};

#[derive(Args, Clone, Debug, Default)]
pub struct CompileArgs {
    #[command(flatten)]
    pub build: BuildArgs,
}

pub(crate) fn handle(args: CompileArgs, context: Context) -> Result<()> {
    build::handle(args.build, context)
}
