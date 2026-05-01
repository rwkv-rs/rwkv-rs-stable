---
name: coding-conventions
description: Coding style, naming, implementation, Result boundary, custom rwkv-nn operator, and configuration constraints for this Rust workspace. Use when writing or reviewing Rust code, deciding error handling, adding comments, implementing kernels, or choosing configuration mechanisms.
---

# Coding Style and Implementation Constraints

## Basic Style

- Use ASCII punctuation consistently.
- Follow `rustfmt.toml`: maximum line width `100`, 4-space indentation, grouped imports, and no automatic import reordering.
- Put all `use` items near the top of the file when possible. Avoid writing absolute paths inside code.

## Naming Conventions

- Modules and functions: `snake_case`
- Types and traits: `PascalCase`
- Constants: `SCREAMING_SNAKE_CASE`
- Count prefixes: `num_*`
- Index names: `*_index`
- Trainable parameters: `params_*`
- Boolean states: `is_*`
- Boolean controls: `need_*`

## Implementation Preferences

- Confirm the task goal and boundary before choosing an implementation method.
- If the user or context requires official or community common practice, first choose capabilities already provided by official Rust documentation, compiler lints, `rustdoc`, `clippy`, the standard library, and mainstream crates.
- Before designing a new mechanism, verify whether it is compatible with the repository's current baseline, whether it truly solves the target problem, and whether the maintenance cost is worthwhile.
- Prefer avoiding `unsafe`; use it only when necessary and the benefit is clear.
- Prefer rejecting `Result`. Before introducing `Result`, think deeply about why this error must be handled by the caller, instead of using `panic!`, `expect`, or `unwrap` directly.
- When judging exception boundaries, prioritize user cognitive load. If requiring the caller to handle an exception only increases usage complexity without real recoverability, do not design it as `Result`.
- If the core goal is for the system to run stably under correct use, do not wrap internal logic errors, broken state invariants, broken scheduling contracts, invalid shape contracts, or implementation defects as external `Err` values.
- Use `Result` only for user input errors, external I/O, missing configuration, network/storage dependency failures, and other exception paths that genuinely need interface-side or caller-side decisions.
- For `Result::Err` caused by code logic problems, the default conclusion should be that the implementation has a defect to fix, rather than continuing to propagate the exception to the interface side.
- When adding comments, prefer explaining design intent, background constraints, and implementation reasons rather than restating surface code behavior.
- Do not add helper functions that are called once and only forward parameters or hide a small local sequence.
- Extract helpers only when they provide reuse, remove meaningful duplication, reduce real complexity, or name a non-obvious domain concept.
- Before finishing a change, scan the current diff for newly added single-call wrappers and inline them unless they meet the extraction rule above.

## `rwkv-nn` Custom Operators

- Backend traits express only public primitive capabilities, such as forward `fused_<kernel>`. Do not add `*BackwardBackend` traits for a single custom backward path.
- Autodiff backward is an internal operator implementation detail. In `backward.rs`, place the Burn `Backward` node near the `Autodiff<...>` backend impl, and write gradient logic directly in `Backward::backward`.
- If backward can be expressed with ordinary Burn tensor ops, write tensor ops directly following the template style.
- If backward requires a CubeCL kernel or autotune, make the `Autodiff<CubeBackend<...>, C>` impl specific enough to call the kernel directly. Keep autotune and launch logic local to the corresponding `Backward::backward`.
- Do not create helpers that only forward parameters or name a single call site, such as `*_backward`, `*_autodiff`, or `*_variant`. Extract helpers only when they provide real reuse value or significantly reduce complexity.
- If a tuner candidate serves only one backward node, inline it inside the `Tunable::new(...).ok()` closure. Keep safety comments for specific kernel launches next to the corresponding `unsafe` block.
- Keep all `use` items at the top of the file. Use code order to express "autodiff impl first, kernel details later"; do not place imports in the middle of a file.

## Configuration Constraints

This project prefers loading configuration from TOML through `rwkv-config`. Do not default to introducing `.env`, environment variables, or command-line-argument style configuration as a general solution unless the task explicitly requires it and it is consistent with the existing design.
