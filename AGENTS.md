# Repository Guidelines

## Usage
This file keeps only high-priority constraints and navigation. For a concrete task, first decide whether to read the corresponding skill. Do not load all instructions at once.

Read as needed:
- Project structure and crate responsibilities: `$project-structure`
- Build, test, benchmark, and common commands: `$development-workflow`
- Test and benchmark mirrored-directory conventions: `$testing-layout`
- Compile, test, benchmark, and regression-check standards: `$verification-standards`
- Coding style, naming, and implementation constraints: `$coding-conventions`
- Code review constraints and reviewer behavior: `$code-review`
- Commit and PR conventions: `$commit-and-pr`
- Source comments and documentation workflow: `$documentation-workflow`

## High-Priority Rules
- Align on the goal before expanding into solution design or implementation. If the user emphasizes "official/community common practice", do not invent repository-private workflows, tools, or gates first.
- Prefer the minimum sufficient approach from official Rust guidance and mainstream community consensus. Consider additional mechanisms only when current mainstream practice clearly cannot satisfy the goal, and explain the gap first.
- This repository is a Rust workspace. Dependency versions are managed by the workspace. When adding or adjusting dependencies, prefer editing the root `Cargo.toml`.
- Do not commit secrets or sensitive information, such as `HF_TOKEN`, proxy configuration, or deployment environment variables.
- Before modifying code, read the minimum necessary skill instructions. After modifying code, run verification commands that match the change scope. For test/benchmark placement and naming, follow `$testing-layout`. For acceptance thresholds and regression checks, follow `$verification-standards`.
- Before committing, run `cargo run-checks` by default. If a local change clearly uses narrower verification, write the commands actually run in the commit note or PR description.
- Use `cargo +nightly fmt --all` or `cargo xtask check format` for formatting and format checks. Do not use stable `cargo fmt --all` as the final formatting authority.
- For code review, PR review, or pre-merge checks, use `$pragmatic-clean-code-reviewer` by default and also follow `$code-review` for this Rust repository's supplemental constraints.
- If `$pragmatic-clean-code-reviewer` is unavailable, use `$skill-installer` to install `https://github.com/Zhen-Bo/pragmatic-clean-code-reviewer` as `pragmatic-clean-code-reviewer` before continuing the review.
- Use Conventional Commits. Keep commit messages short and locatable, for example `fix(rwkv-infer): ...`.

## Additional Notes
The skills under `.agents/skills/` are the detailed agent-facing specifications for this repository. If this file conflicts with a more specific skill, the more specific and task-local skill wins.

## Conversation Style
- Do not use "not ... but ..." as the default expression pattern.
- Use that contrast only when the user is genuinely likely to confuse A for B. Put the correct content first and add "rather than ..." as a supplement.
- If the "not" part is obvious, common sense, or unlikely to be misunderstood, delete it and state only the correct content.
- Do not perform rigor by first stating a worthless wrong interpretation, especially when the wrong interpretation is longer than the correct conclusion.
- State facts directly when possible. Do not walk down a wrong branch before returning to the point.
- Avoid performative clarification.

## Behavior Style
- Define correctness as: simpler and more intuitive logic, better performance, clear file and code structure, fully separated responsibilities, key comments, smaller future change sets, and templates that are easy to copy.
- Replace old content with the correct logic in one pass.
- Documentation and comments should not mention old implementations.
