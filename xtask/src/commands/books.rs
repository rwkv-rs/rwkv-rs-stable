use std::path::Path;

use anyhow::Result;
use clap::{Args, Subcommand};

use crate::commands;

#[derive(Args, Clone, Debug)]
pub struct BooksArgs {
    #[command(subcommand)]
    pub book: BookKind,
}

#[derive(Clone, Debug, Subcommand)]
pub enum BookKind {
    /// RWKV user-facing book.
    Rwkv(BookKindArgs),
    /// Reserved contributor book entrypoint.
    Contributor(BookKindArgs),
}

#[derive(Args, Clone, Debug)]
pub struct BookKindArgs {
    #[command(subcommand)]
    pub command: BookSubcommand,
}

#[derive(Clone, Debug, Subcommand)]
pub enum BookSubcommand {
    /// Build the selected book.
    Build,
    /// Serve the selected book locally.
    Open(OpenArgs),
}

#[derive(Args, Clone, Debug)]
pub struct OpenArgs {
    #[arg(long, default_value_t = 3000)]
    pub port: u16,
}

pub(crate) fn handle(args: BooksArgs) -> Result<()> {
    match args.book {
        BookKind::Rwkv(book_args) => {
            handle_book("rwkv-rs-book", Path::new("./rwkv-rs-book"), book_args)
        }
        BookKind::Contributor(book_args) => handle_book(
            "contributor-book",
            Path::new("./contributor-book"),
            book_args,
        ),
    }
}

fn handle_book(name: &str, path: &Path, args: BookKindArgs) -> Result<()> {
    if !path.exists() {
        commands::print_placeholder(&format!(
            "book `{name}` is reserved but not present in this workspace yet"
        ));
        return Ok(());
    }

    commands::ensure_tool("mdbook")?;

    match args.command {
        BookSubcommand::Build => {
            commands::run_in_dir("mdbook", &[String::from("build")], &[], Some(path))
        }
        BookSubcommand::Open(open_args) => commands::run_in_dir(
            "mdbook",
            &[
                "serve".into(),
                "--open".into(),
                "--port".into(),
                open_args.port.to_string(),
            ],
            &[],
            Some(path),
        ),
    }
}
