use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use rwkv_export::{ConvertOptions, check_st, rwkv_lm_pth2st};

#[derive(Debug, Parser)]
#[command(name = "rwkv-export")]
#[command(about = "RWKV weight export utilities")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Convert a local RWKV7 G1 PyTorch checkpoint into web-rwkv `.safetensors`.
    #[command(name = "pth2st", alias = "pth-to-web")]
    Pth2St {
        /// Input `.pth` checkpoint path.
        #[arg(long)]
        input: PathBuf,
        /// Output `.safetensors` path.
        #[arg(long)]
        output: PathBuf,
        /// Replace the output file when it already exists.
        #[arg(long)]
        overwrite: bool,
    },
    /// Validate a RWKV LM `.safetensors` file and print inferred model shape.
    #[command(name = "check-st", alias = "check-web")]
    CheckSt {
        /// Input `.safetensors` path.
        #[arg(long)]
        input: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Pth2St {
            input,
            output,
            overwrite,
        } => rwkv_lm_pth2st(input, output, ConvertOptions { overwrite })?,
        Command::CheckSt { input } => {
            let info = check_st(input)?;
            println!(
                "rwkv7-g1 tensors={} layers={} vocab={} emb={} heads={} head_size={}",
                info.num_tensors,
                info.num_cells,
                info.vocab_size,
                info.embedded_dim,
                info.num_heads,
                info.head_size
            );
        }
    }

    Ok(())
}
