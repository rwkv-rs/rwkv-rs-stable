---
name: verification-standards
description: Verification standards for compile, test, benchmark, documentation, and regression checks in this Rust workspace. Use when deciding minimum acceptance commands by risk level, recording validation results, or checking whether tests and benchmarks are sufficient.
---

# Verification Standards

## Basic Requirements

- Any change that affects source comment contracts, module terminology, exception strategy, or evidence references must run at least:
  - `cargo doc --workspace --no-deps`
  - `mdbook build rwkv-rs-book`
- Any code change still needs a scope-matched `cargo check`, `cargo test`, or `cargo nextest`.
- Test and benchmark placement paths and naming must follow `$testing-layout`. Unit tests should follow the tested function in the corresponding source file.
- Workspace crate/example roots default to `#![warn(missing_docs)]` and rustdoc `warn` settings to expose documentation debt.

## Risk Levels

- Low risk:
  - Documentation-only or comment-only changes with no behavior change.
  - Minimum requirement: `cargo doc --workspace --no-deps`, `mdbook build rwkv-rs-book`.
- Medium risk:
  - Changes to public APIs, data structures, configuration loading, or error propagation.
  - Minimum requirement: low-risk commands plus relevant crate `cargo check` / `cargo test`.
- High risk:
  - Changes to kernels, inference hot paths, scheduling logic, benchmark harnesses, or training optimizers.
  - Minimum requirement: medium-risk commands plus corresponding benchmark runs and result confirmation.

## Result Retention

- When a change involves performance conclusions, prefer updating the corresponding Criterion benchmark and use the `github-action-benchmark` CI history page as long-term retention.
- When a change involves concept or history explanations, update `rwkv-rs-book` at the same time.
