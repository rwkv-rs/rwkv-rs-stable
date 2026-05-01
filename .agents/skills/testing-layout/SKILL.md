---
name: testing-layout
description: Test and benchmark directory layout, path mapping, naming, and execution conventions for this Rust workspace. Use when adding or moving tests, benchmarks, testdata, test generators, or source-adjacent test utilities.
---

# Test and Benchmark Directory Conventions

## Core Rules

- Unit tests follow the file containing the tested function. If a function is defined in a source file, put the corresponding unit tests in that source file's `#[cfg(test)] mod tests`.
- `src/test_utils/**` is for test utilities and shared logic, such as backend types, fixtures, random input construction, and assertion helpers.
- Tests that cross modules, integrate components, are end-to-end, or need an independent test crate boundary may live under `tests/**`.
- Performance benchmarks live under `benches/**`, and their directory structure should mirror the tested source structure.
- Reference data lives in `testdata/`. Reference data generation scripts live in `testgen/`.

## Path Mapping

- Source: `crates/<crate>/src/foo/bar.rs`
- Unit tests: `#[cfg(test)] mod tests` inside `crates/<crate>/src/foo/bar.rs`
- Integration tests: `crates/<crate>/tests/foo/bar.rs`
- Benchmarks: `crates/<crate>/benches/foo/bar.rs`

Examples:
- `crates/rwkv-nn/src/kernels/addcmul/mod.rs`
  - Unit test functions: `forward`, `backward`
- `crates/rwkv-nn/benches/kernels/addcmul/forward.rs`
  - Benchmark function for a single operator: `forward`
  - Benchmark function in a multi-operator file: `<kernel_name>_forward`
- `crates/rwkv-nn/benches/kernels/addcmul/backward.rs`
  - Benchmark function for a single operator: `backward`
  - Benchmark function in a multi-operator file: `<kernel_name>_backward`
- `crates/rwkv-eval/src/cores/datasets/maths/gsm8k.rs`
  - Integration or dataset tests may live in `crates/rwkv-eval/tests/cores/datasets/maths/gsm8k.rs`
- `examples/rwkv-lm/src/inferring.rs`
  - Benchmarks may live in `examples/rwkv-lm/benches/inferring.rs`

## Module Organization

- Put unit tests in the tested source file's `#[cfg(test)] mod tests`.
- Align directory structures under `tests/**` and `benches/**` with the source structure, so tests or benchmarks can be found from a source path.
- Put shared test utilities under `src/test_utils/**`; put shared benchmark utilities in the corresponding `benches/**/mod.rs`.
- Benchmark entrypoints use `#[path = "../mod.rs"] mod common;` to import same-level shared utilities.
- If one test or benchmark covers multiple input sizes, scenarios, or baseline/custom comparisons, organize them inside the same test function.

## Benchmark Target Organization

- Register only crate-level or subsystem-level Criterion targets in `Cargo.toml`, such as `kernels`, `tokenizer`, `mmap`, or `processor`.
- Do not add a new `[[bench]]` for each operator, step, scenario, or forward/backward case.
- Put concrete benchmark files under `benches/<target>/**`, mirroring the source tree, and aggregate them from `benches/<target>.rs` with Rust modules.
- When adding a benchmark case under an existing subsystem, update the Rust benchmark entrypoint or child module instead of expanding `Cargo.toml`.

## Test Design

- Prefer tests that exercise real business logic and observable behavior. Choose the test shape from the behavior being protected.
- For format conversion, serialization, import/export, encoding/decoding, and persistence paths, prefer roundtrip, equivalence, or invariant tests; preserving business meaning is the baseline requirement.
- Add tests for internal metadata, panic branches, or private helpers only when they represent a business contract or cover a key risk missed by the main workflow.
- When old and new formats or compatibility paths coexist, verify both paths with the same business sample semantics where possible.

## Naming Conventions

- "Test function" includes unit test functions and benchmark test functions.
- Name ordinary test functions `<tested_function>`.
- One test function corresponds to one tested function. The test function body may call the tested function multiple times to cover different inputs, boundaries, expected outputs, or benchmark input sizes.
- Custom rule: tests for all neural-network `Module` and `Kernel` types are named `forward` / `backward`, corresponding to forward numerical accuracy/performance and backward gradient accuracy/performance.
- For custom operator benchmarks, if a kernel module exposes only one operator, name functions `forward` / `backward`. If a kernel module exposes multiple operators, name functions `<kernel_name>_forward` / `<kernel_name>_backward`.
- Benchmark groups, Criterion `BenchmarkId`, and variables inside the test body may express scenario, scale, or baseline/custom distinctions. The test function name itself should not carry the scenario.
- Test function names should semantically locate the tested object directly.

## Execution Entrypoints

- Run unit tests with `cargo test -p <crate> <test_function_name>` or `cargo nextest run -p <crate> <test_function_name>`.
- Enter `rwkv-eval` benchmark dataset tests through `cargo nextest run -p rwkv-eval --test benchmark_datasets ...`.
- Run a single crate's Criterion benchmark with `cargo bench -p <crate> --bench <subsystem>`. Add `-- --output-format bencher` when CI benchmark history integration is needed.

## Constraints

- Python does not participate in formal test orchestration. It is allowed only in `testgen/` to generate `testdata/`.
- When changing source code, also maintain unit tests in the same file as the tested function. When touching hot or performance-sensitive paths, also maintain the corresponding `benches/**` benchmarks.
- `src/test_utils/**` provides only test helper capabilities.
