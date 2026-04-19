use anyhow::Result;
use clap::Args;

use crate::{
    commands::{
        books::{self, BookKind, BookKindArgs, BookSubcommand, BooksArgs},
        build::{self, BuildArgs},
        check::{self, CheckArgs, CheckSubcommand},
        doc::{self, DocArgs, DocSubcommand},
        test::{self, CiTestType, TestArgs},
        WorkspaceArgs,
    },
    Context,
};

#[derive(Args, Clone, Debug, Default)]
pub struct ValidateArgs {
    #[command(flatten)]
    pub workspace: WorkspaceArgs,
}

pub(crate) fn handle(args: ValidateArgs, context: Context) -> Result<()> {
    if matches!(context, Context::Std | Context::All) {
        check::handle(CheckArgs {
            command: CheckSubcommand::All,
        })?;

        build::handle(
            BuildArgs {
                workspace: args.workspace.clone(),
                ci: true,
            },
            Context::Std,
        )?;

        test::handle(
            TestArgs {
                workspace: args.workspace.clone(),
                ci: Some(CiTestType::Github),
            },
            Context::Std,
        )?;

        doc::handle(
            DocArgs {
                command: DocSubcommand::Build,
                workspace: args.workspace.clone(),
            },
            Context::Std,
        )?;
        doc::handle(
            DocArgs {
                command: DocSubcommand::Tests,
                workspace: args.workspace.clone(),
            },
            Context::Std,
        )?;

        books::handle(BooksArgs {
            book: BookKind::Rwkv(BookKindArgs {
                command: BookSubcommand::Build,
            }),
        })?;
    }

    if matches!(context, Context::NoStd | Context::All) {
        build::handle(
            BuildArgs {
                workspace: args.workspace.clone(),
                ci: true,
            },
            Context::NoStd,
        )?;
        test::handle(
            TestArgs {
                workspace: args.workspace,
                ci: Some(CiTestType::Github),
            },
            Context::NoStd,
        )?;
    }

    Ok(())
}
