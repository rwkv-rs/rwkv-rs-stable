pub(crate) mod bench;
pub(crate) mod books;
pub(crate) mod build;
pub(crate) mod bump;
pub(crate) mod check;
pub(crate) mod compile;
pub(crate) mod coverage;
pub(crate) mod dependencies;
pub(crate) mod doc;
pub(crate) mod fix;
pub(crate) mod publish;
pub(crate) mod test;
pub(crate) mod validate;
pub(crate) mod vulnerabilities;

use std::{
    path::Path,
    process::{Command, Stdio},
};

use anyhow::{bail, Context as _, Result};
use clap::Args;
use serde::Deserialize;

#[derive(Args, Clone, Debug, Default)]
pub(crate) struct WorkspaceArgs {
    /// Restrict execution to selected packages.
    #[arg(long = "only", value_delimiter = ',', num_args = 1..)]
    pub(crate) only: Vec<String>,

    /// Exclude packages from workspace-wide execution.
    #[arg(long = "exclude", value_delimiter = ',', num_args = 1..)]
    pub(crate) exclude: Vec<String>,

    /// Enable comma-separated feature flags.
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    pub(crate) features: Vec<String>,

    /// Disable default features.
    #[arg(long)]
    pub(crate) no_default_features: bool,

    /// Run cargo commands in release mode.
    #[arg(long)]
    pub(crate) release: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct WorkspacePackage {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) targets: Vec<WorkspaceTarget>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct WorkspaceTarget {
    pub(crate) name: String,
    pub(crate) kind: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct CargoMetadata {
    packages: Vec<WorkspacePackage>,
    workspace_members: Vec<String>,
}

pub(crate) fn run(program: &str, args: &[String]) -> Result<()> {
    run_with_env(program, args, &[])
}

pub(crate) fn run_with_env(
    program: &str,
    args: &[String],
    envs: &[(String, String)],
) -> Result<()> {
    run_in_dir(program, args, envs, None)
}

pub(crate) fn run_in_dir(
    program: &str,
    args: &[String],
    envs: &[(String, String)],
    dir: Option<&Path>,
) -> Result<()> {
    let mut command = Command::new(program);
    command.args(args);
    command.stdin(Stdio::inherit());
    command.stdout(Stdio::inherit());
    command.stderr(Stdio::inherit());

    for (key, value) in envs {
        command.env(key, value);
    }

    if let Some(dir) = dir {
        command.current_dir(dir);
    }

    print_command(program, args, envs, dir);

    let status = command
        .status()
        .with_context(|| format!("failed to execute `{program}`"))?;
    if !status.success() {
        bail!("command `{program}` exited with status {status}");
    }

    Ok(())
}

pub(crate) fn capture(program: &str, args: &[String]) -> Result<String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .with_context(|| format!("failed to execute `{program}`"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("command `{program}` failed: {stderr}");
    }

    String::from_utf8(output.stdout).context("command output was not valid utf-8")
}

pub(crate) fn ensure_tool(tool: &str) -> Result<()> {
    let status = Command::new(tool)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .with_context(|| format!("failed to probe tool `{tool}`"))?;

    if !status.success() {
        bail!("required tool `{tool}` is not available in PATH");
    }

    Ok(())
}

pub(crate) fn print_placeholder(message: &str) {
    eprintln!("reserved: {message}");
}

pub(crate) fn workspace_packages() -> Result<Vec<WorkspacePackage>> {
    let output = capture(
        "cargo",
        &[
            "metadata".into(),
            "--format-version".into(),
            "1".into(),
            "--no-deps".into(),
        ],
    )?;

    let metadata: CargoMetadata = sonic_rs::from_str(&output).context("parse cargo metadata")?;
    let packages = metadata
        .packages
        .into_iter()
        .filter(|package| metadata.workspace_members.contains(&package.id))
        .collect();

    Ok(packages)
}

pub(crate) fn current_workspace_package_names() -> Result<Vec<String>> {
    let mut names = workspace_packages()?
        .into_iter()
        .map(|package| package.name)
        .collect::<Vec<_>>();
    names.sort();
    Ok(names)
}

pub(crate) fn append_workspace_args(args: &mut Vec<String>, workspace: &WorkspaceArgs) {
    if workspace.only.is_empty() {
        args.push("--workspace".into());
    } else {
        for package in &workspace.only {
            args.push("-p".into());
            args.push(package.clone());
        }
    }

    for package in &workspace.exclude {
        args.push("--exclude".into());
        args.push(package.clone());
    }

    if !workspace.features.is_empty() {
        args.push("--features".into());
        args.push(workspace.features.join(","));
    }

    if workspace.no_default_features {
        args.push("--no-default-features".into());
    }

    if workspace.release {
        args.push("--release".into());
    }
}

fn print_command(program: &str, args: &[String], envs: &[(String, String)], dir: Option<&Path>) {
    let rendered_env = envs
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>();

    let rendered_args = args.join(" ");
    let rendered_dir = dir
        .map(|dir| format!(" (cwd: {})", dir.display()))
        .unwrap_or_default();

    if rendered_env.is_empty() {
        eprintln!("+ {program} {rendered_args}{rendered_dir}");
    } else {
        eprintln!(
            "+ {} {program} {rendered_args}{rendered_dir}",
            rendered_env.join(" ")
        );
    }
}
