# rwkv-nn Test Ownership Migration

## Goal

Move real `rwkv-nn` tests and benchmarks that need model weights, trace data, or
`rwkv-export` mapping into `crates/rwkv-test`. `rwkv-nn` keeps only
`kernels/template/**` local tests and template microbenchmarks.

This avoids a dependency cycle:

```text
rwkv-test -> rwkv-export -> rwkv-nn
```

`rwkv-nn` must not depend on `rwkv-export`.

## Current Contract

- `weights/rwkv-init-0.1b-ctx512-test.st` is the default real model fixture.
- `crates/rwkv-test/test_data/rwkv_lm/bf16/case_000000` is the default trace
  baseline.
- `crates/rwkv-test` owns real `rwkv-nn` correctness and benchmark entrypoints.
- `crates/rwkv-nn` owns template-kernel local tests and the template-only
  `kernels` Criterion target.
- Fixture loading, safetensors reading, and report writing stay outside measured
  benchmark sections.

## Completed In This Migration

- [x] Moved the trace baseline from `crates/rwkv-nn/test_data/**` to
  `crates/rwkv-test/test_data/**`.
- [x] Removed non-template `rwkv-nn` local tests.
- [x] Removed `rwkv-nn` train trace fixture helpers.
- [x] Removed non-template `rwkv-nn` benchmark targets and files.
- [x] Added a `rwkv-test` library boundary for compare/stat helpers.
- [x] Kept the existing `rwkv-test compare` CLI through a thin binary shim.
- [x] Added `rwkv-test` CUDA integration coverage for `RwkvLM::forward` loss
  against the trace baseline.
- [x] Added a `rwkv-test` Criterion benchmark target for real `rwkv-nn` forward
  loss.
- [x] Updated repo-local skills so future work keeps the same ownership rule.

## Acceptance Commands

```bash
cargo metadata --no-deps
cargo test -p rwkv-nn --lib --features cuda kernels::template
cargo test -p rwkv-test --features cuda
cargo run -p rwkv-test -- compare \
  --actual crates/rwkv-test/test_data/rwkv_lm/bf16/case_000000 \
  --baseline crates/rwkv-test/test_data/rwkv_lm/bf16/case_000000
cargo bench -p rwkv-test --bench rwkv_nn --features cuda -- --quick
cargo +nightly fmt --all -- --check
```
