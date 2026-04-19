mod commands;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};

use crate::commands::{
    bench::{self, BenchArgs},
    books::{self, BooksArgs},
    build::{self, BuildArgs},
    bump::{self, BumpArgs},
    check::{self, CheckArgs},
    compile::{self, CompileArgs},
    coverage::{self, CoverageArgs},
    dependencies::{self, DependenciesArgs},
    doc::{self, DocArgs},
    fix::{self, FixArgs},
    publish::{self, PublishArgs},
    test::{self, TestArgs},
    validate::{self, ValidateArgs},
    vulnerabilities::{self, VulnerabilitiesArgs},
};

#[derive(Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum Context {
    Std,
    NoStd,
    All,
}

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Workspace automation entrypoint for rwkv-rs."
)]
struct Cli {
    #[arg(long, value_enum, default_value_t = Context::Std)]
    context: Context,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run workspace builds.
    Build(BuildArgs),
    /// Run benchmark targets and CI benchmark export.
    Bench(BenchArgs),
    /// Manage mdBook-based books.
    Books(BooksArgs),
    /// Placeholder for future version bump automation.
    Bump(BumpArgs),
    /// Run code-quality checks.
    Check(CheckArgs),
    /// Alias for `build`.
    Compile(CompileArgs),
    /// Manage coverage tooling and reports.
    Coverage(CoverageArgs),
    /// Run dependency-oriented checks.
    Dependencies(DependenciesArgs),
    /// Build docs or run doc tests.
    Doc(DocArgs),
    /// Apply non-destructive local fixes.
    Fix(FixArgs),
    /// Publish crates in dependency order.
    Publish(PublishArgs),
    /// Run tests for local development or CI profiles.
    Test(TestArgs),
    /// Run the default validation pipeline.
    Validate(ValidateArgs),
    /// Run nightly vulnerability-oriented checks.
    Vulnerabilities(VulnerabilitiesArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Bench(args) => bench::handle(args),
        Command::Books(args) => books::handle(args),
        Command::Build(args) => build::handle(args, cli.context),
        Command::Bump(args) => bump::handle(args),
        Command::Check(args) => check::handle(args),
        Command::Compile(args) => compile::handle(args, cli.context),
        Command::Coverage(args) => coverage::handle(args),
        Command::Dependencies(args) => dependencies::handle(args),
        Command::Doc(args) => doc::handle(args, cli.context),
        Command::Fix(args) => fix::handle(args),
        Command::Publish(args) => publish::handle(args),
        Command::Test(args) => test::handle(args, cli.context),
        Command::Validate(args) => validate::handle(args, cli.context),
        Command::Vulnerabilities(args) => vulnerabilities::handle(args),
    }
}
