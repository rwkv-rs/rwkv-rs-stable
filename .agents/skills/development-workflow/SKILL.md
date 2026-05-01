---
name: development-workflow
description: Build, test, benchmark, documentation, and development workflow guidance for this Rust workspace. Use when choosing commands, deciding verification scope, running tests or benchmarks, handling performance-sensitive changes, or checking repository development conventions.
---

# Build, Test, and Development Workflow

## Common Commands

- `cargo check --workspace`: Quickly check the whole workspace.
- `cargo build --workspace --release`: Build release artifacts.
- `cargo +nightly fmt --all`: Format code with the repository rustfmt configuration.
- `cargo clippy --workspace --all-targets -- -D warnings`: Check code with zero warnings as the goal.
- `cargo nextest run --workspace`: Preferred test entrypoint.
- `cargo llvm-cov nextest --workspace --lcov --output-path target/coverage/lcov.info`: Run workspace tests and generate a coverage report.
- `cargo bench -p <crate> --bench <subsystem>`: Run a Criterion benchmark target. When output must be retained for `github-action-benchmark`, append `-- --output-format bencher`.
- `cargo doc --workspace --no-deps`: Generate documentation and check export layers. `missing_docs` and rustdoc lints on crate roots expose workspace documentation issues as warnings.
- `mdbook build rwkv-rs-book`: Build the book and check chapter structure.

## Execution Strategy

Prefer the smallest command that matches the change scope, such as `cargo test -p rwkv-infer` or `cargo check -p rwkv-eval`. Add workspace-level checks for cross-crate or public-interface changes.

For whether unit tests or benchmarks are needed, whether performance regressed, or which commands complete acceptance, follow `$verification-standards`. That skill defines minimum verification thresholds by risk level and result-recording expectations.

For where tests or benchmarks belong and how to find tests/benchmarks from source paths, follow `$testing-layout`.

## Testing Conventions

- Unit tests follow the file containing the tested function and live in that source file's `#[cfg(test)] mod tests`.
- `src/test_utils/**` is for test utilities and shared logic.
- Integration, end-to-end, or cross-crate boundary tests may live under `tests/**`.
- Use `#[tokio::test]` for async logic.
- `rwkv-eval` benchmark dataset tests live under `crates/rwkv-eval/tests/cores/datasets/**`, are ignored by default, and must be run with `cargo nextest -p rwkv-eval --test benchmark_datasets --run-ignored only ...`.
- `examples/rwkv-lm-eval/scripts/unit_test.sh` shows a test entrypoint that requires `HF_TOKEN`.
- CI coverage uses `cargo-llvm-cov` plus `cargo nextest`. Coverage below the `codecov.yml` target produces only a summary or PR hint and does not block deployment after successful checks.

## Performance-Related Changes

For performance optimizations, do not rely only on subjective judgment. Use benchmark tests or key metrics to state before/after differences and bottleneck locations.

If a change touches `rwkv-nn` kernels, `rwkv-lm` inference paths, or other hot code, prefer reusing the `benches/` benchmark paths corresponding to the source structure. If no benchmark exists, add a new benchmark file according to `$testing-layout`. Use Criterion and the standard `github-action-benchmark` integration for microbenchmarks that need long-term comparison.

For benchmark-only validation, use the smallest subsystem target, such as `cargo bench -p rwkv-data --bench tokenizer -- --test` or `cargo bench -p rwkv-data --bench tokenizer --no-run`, when a full benchmark run is not needed.

CI benchmark history tracks `rwkv-nn` Criterion microbenchmarks. The benchmark workflow compares PRs with the `run-benchmarks` label. Manual workflow runs update history curves when the target is the default branch. The workflow is fixed to the GPU runner `szx-pro6000x4` and should not be scheduled on GPU-less cloud hosts. `examples/rwkv-lm/benches/inferring.rs` is a manual stress-test entrypoint with external model configuration and is not part of that history curve.

For changes involving source comments, terminology, or exception strategy, also follow `$documentation-workflow`.
