# rwkv-test

`rwkv-test` compares RWKV trace activation dumps produced by the companion
[`rwkv-rs-test`](https://github.com/rwkv-rs/rwkv-rs-test) trace workflow. It also
owns real `rwkv-nn` validation that needs model weights, trace fixtures, or
`rwkv-export` weight mapping. Keeping that validation here avoids making
`rwkv-nn` depend on `rwkv-export`.

## Usage

```bash
cargo run -p rwkv-test -- compare \
  --actual /path/to/test_gen/albatross/fp16/case_000000 \
  --baseline /path/to/test_gen/rwkv_lm/bf16/case_000000 \
  --atol 1e-3 --rtol 1e-2 --cos-min 0.999
```

## rwkv-nn Validation

`rwkv-nn` keeps only template-kernel local tests. Run real model validation
through this crate:

```bash
cargo test -p rwkv-test --features cuda
cargo bench -p rwkv-test --bench rwkv_nn --features cuda -- --quick
```

The default fixture uses `weights/rwkv-init-0.1b-ctx512-test.st` and
`crates/rwkv-test/test_data/rwkv_lm/bf16/case_000000`.

Generate and compare a `rwkv-nn` trace against the default `rwkv-lm` training
trace fixture with:

```bash
cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn
```

The `compare-rwkv-nn` timing profile is `train-forward-steady-state`: the
`rwkv-nn` actual trace runs an in-process warmup before collecting
`timing/<module>.time.json` samples. A speedup is only reported when actual and
baseline timing files have compatible `repeat` and `warmup` metadata, and
`warmup > 0`. The speedup is not an inference speedup, and it is not a full
training-step speedup because backward and the optimizer step are not part of
these module timing files.

`--actual` and `--baseline` must point at matching case directories. The
`compare` command prints two separate result surfaces: activation comparison
for `.safetensors` files, and module timing comparison for
`timing/**/*.time.json`. Activation rows are keyed by relative safetensors path.
Timing rows are keyed by the JSON `module` field.

Module timing uses the `canonical_compute` scope for speedup summaries. Timing
files live under `timing/<module>.time.json`, and the canonical key comes from
the JSON `module` field. Each timing JSON must include `module`, positive
`elapsed_ns`, positive `repeat`, non-negative `warmup`, and `samples_ns` whose
rounded average equals `elapsed_ns`. The summary includes modules such as embedding, layer
norm, time mixer, channel mixer, residual adds, LM-head hidden state, and loss.
It ignores input metadata and auxiliary activations such as
`embedding/token_ids.safetensors`, `cell_0000/time_mixer/value_from_first_cell.safetensors`,
and loss helper tensors because those files either do not measure compute or
are outputs of a module that already has one timing record. Missing canonical
timing on either side fails the timing comparison.

## Trace File Contract

The expected trace layout follows the `rwkv-rs-test` case contract:

```text
case_000000/
├── embedding/
│   ├── token_ids.safetensors
│   └── embedded_context.safetensors
├── layer_norm0/
│   └── embedded_context.safetensors
├── cells/
│   ├── cell_0000/
│   │   ├── time_mixer/value_from_first_cell.safetensors
│   │   ├── time_mixer/embedded_context.safetensors
│   │   ├── embedded_context_after_time_mixer.safetensors
│   │   ├── channel_mixer/embedded_context.safetensors
│   │   └── embedded_context_after_channel_mixer.safetensors
│   └── cell_<n>/
│       └── ...
├── lm_head/
│   ├── embedded_context.safetensors
│   └── logits.safetensors
└── timing/
    ├── embedding.time.json
    ├── layer_norm0.time.json
    ├── cells/cell_0000/time_mixer.time.json
    ├── cells/cell_0000/channel_mixer.time.json
    └── loss/l2wrap_cross_entropy.time.json
```

Every `.safetensors` file must contain exactly one tensor, and the tensor name
must match the file stem. For example, `logits.safetensors` must contain a
single tensor named `logits`.

Every compute module that participates in speedup summaries should have one
`timing/<module>.time.json` file whose `elapsed_ns` covers only that module's
compute region. Safetensors export, JSON writing, filesystem sync, and CPU
copies must stay outside `elapsed_ns`. `rwkv-peft` unfused internals should be
aggregated at the export side into the matching `rwkv-lm` canonical module
before comparison; internal temporary tensors are not new comparison outputs.

The actual and baseline tensors must have the same dtype, shape, and element
count. Dtype or shape mismatches fail before numeric tolerances are applied.

## Comparison Rules

Floating-point tensors are decoded to `f64` for statistics. Supported numeric
dtypes are `F64`, `F32`, `F16`, `BF16`, signed and unsigned integer dtypes, and
`BOOL`. Packed, FP8, MX, and complex dtypes are rejected with an error.

For each tensor, the output includes:

- `count`
- `max_abs` and `mean_abs`
- `max_rel = max(abs(actual - baseline) / max(abs(baseline), rel_eps))`
- `mean_rel`
- `cosine`

Floating-point tensors fail when `max_abs > atol && max_rel > rtol`, or when
`cosine < cos-min`. Integer and bool tensors are compared byte-for-byte because
they usually carry ids or masks.

Defaults are strict: `--atol 0`, `--rtol 0`, and `--cos-min 1`. Pass explicit
tolerances when comparing different engines or dtypes.

## Exit Codes

- `0`: all compared files passed.
- `1`: at least one tensor failed, or a file was missing/extra.
- `2`: CLI arguments, filesystem access, or safetensors parsing failed.
