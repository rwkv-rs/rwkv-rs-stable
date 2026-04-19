use anyhow::Result;
use clap::Args;

use crate::commands;

#[derive(Args, Clone, Debug, Default)]
pub struct PublishArgs {
    /// Print the publish DAG and exit.
    #[arg(long)]
    pub plan: bool,

    /// Use cargo publish --dry-run.
    #[arg(long)]
    pub dry_run: bool,

    /// Allow dirty working trees during publish operations.
    #[arg(long)]
    pub allow_dirty: bool,

    /// Restrict publish execution to a single crate node.
    #[arg(long = "crate")]
    pub package: Option<String>,
}

pub(crate) fn handle(args: PublishArgs) -> Result<()> {
    let packages = commands::current_workspace_package_names()?;

    if args.plan {
        print_plan(&packages);
        return Ok(());
    }

    for step in publish_plan() {
        if args
            .package
            .as_ref()
            .map(|selected| selected != step.name)
            .unwrap_or(false)
        {
            continue;
        }

        if !step.active || !packages.contains(&step.name.to_string()) {
            commands::print_placeholder(&format!(
                "publish node `{}` is reserved with dependencies {:?}",
                step.name, step.depends_on
            ));
            continue;
        }

        let mut cargo_args = vec!["publish".into(), "-p".into(), step.name.into()];
        if args.dry_run {
            cargo_args.push("--dry-run".into());
        }
        if args.allow_dirty {
            cargo_args.push("--allow-dirty".into());
        }

        commands::run("cargo", &cargo_args)?;
    }

    Ok(())
}

struct PublishStep {
    name: &'static str,
    depends_on: &'static [&'static str],
    active: bool,
}

fn publish_plan() -> &'static [PublishStep] {
    &[
        PublishStep {
            name: "rwkv-config",
            depends_on: &[],
            active: false,
        },
        PublishStep {
            name: "rwkv-derive",
            depends_on: &[],
            active: false,
        },
        PublishStep {
            name: "rwkv-data",
            depends_on: &["rwkv-config"],
            active: false,
        },
        PublishStep {
            name: "rwkv-nn",
            depends_on: &["rwkv-config"],
            active: false,
        },
        PublishStep {
            name: "rwkv-train",
            depends_on: &["rwkv-config", "rwkv-data", "rwkv-nn"],
            active: false,
        },
        PublishStep {
            name: "rwkv-infer",
            depends_on: &["rwkv-config", "rwkv-nn"],
            active: false,
        },
        PublishStep {
            name: "rwkv-eval",
            depends_on: &["rwkv-config", "rwkv-infer"],
            active: false,
        },
        PublishStep {
            name: "rwkv-export",
            depends_on: &["rwkv-config", "rwkv-nn"],
            active: false,
        },
        PublishStep {
            name: "rwkv-prompt",
            depends_on: &["rwkv-config", "rwkv-infer"],
            active: false,
        },
        PublishStep {
            name: "rwkv-trace",
            depends_on: &["rwkv-config", "rwkv-train", "rwkv-infer"],
            active: false,
        },
        PublishStep {
            name: "rwkv-agent",
            depends_on: &["rwkv-config", "rwkv-infer"],
            active: false,
        },
        PublishStep {
            name: "rwkv-bench",
            depends_on: &["rwkv-config", "rwkv-infer"],
            active: false,
        },
        PublishStep {
            name: "rwkv",
            depends_on: &[
                "rwkv-config",
                "rwkv-derive",
                "rwkv-data",
                "rwkv-nn",
                "rwkv-train",
                "rwkv-infer",
                "rwkv-eval",
                "rwkv-export",
                "rwkv-prompt",
                "rwkv-trace",
                "rwkv-agent",
                "rwkv-bench",
            ],
            active: true,
        },
    ]
}

fn print_plan(existing_packages: &[String]) {
    for step in publish_plan() {
        let state = if step.active && existing_packages.contains(&step.name.to_string()) {
            "active"
        } else {
            "reserved"
        };
        eprintln!("{state}: {} -> {:?}", step.name, step.depends_on);
    }
}
