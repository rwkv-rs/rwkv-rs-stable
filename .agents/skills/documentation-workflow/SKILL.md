---
name: documentation-workflow
description: Source comment, rustdoc, rwkv-rs-book, and documentation gate guidance for this Rust workspace. Use when changing public APIs, comments, terminology, error behavior, design explanations, algorithms, scheduling, performance conclusions, or documentation checks.
---

# Comment and Documentation Workflow

## Goals

- Continuously complete public API rustdoc and expose documentation debt through compiler warnings.
- Keep complex implementation design intent, key constraints, and current background information in source code or `rwkv-rs-book` for long-term maintenance.
- Avoid custom documentation gates that diverge from common Rust community practice.

## Source Comments

- Prefer standard rustdoc style for public API documentation.
- For internal implementation, add comments only when non-obvious constraints, algorithm background, or easy-to-misuse paths exist.
- Documentation and comments focus on the current design, current contract, and current implementation reasons.
- When error behavior must be described, prefer common rustdoc sections such as `# Errors` or `# Panics` instead of custom tag formats.
- `unwrap` / `expect` / `panic!` are allowed in this repository. Whether they are appropriate is judged by concrete semantics and code review, with no extra tool ban.

## `rwkv-rs-book`

Use `rwkv-rs-book` to organize user-visible capabilities, design background, and longer implementation explanations.

Update `rwkv-rs-book` when a change satisfies any of the following:
- It adjusts public APIs or user-visible behavior.
- It modifies core concept names or design explanations.
- It introduces a new algorithm variant, scheduling strategy, or performance conclusion.
- It corrects assumptions in current design documentation.

## Review and Gates

- Workspace crate/example roots uniformly enable `#![warn(missing_docs)]` and rustdoc-related `warn` settings to expose missing docs, broken links, and invalid HTML tags.
- CI and daily verification mainly rely on:
  - `cargo check --workspace`
  - `cargo doc --workspace --no-deps`
  - `mdbook build rwkv-rs-book`
- Comment sufficiency, error-message clarity, and whether design background is needed are mainly guaranteed through code review.
