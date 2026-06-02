# Lm Head Projection Timing Boundary

- Date: 2026-05-16
- Branch/worktree: `kernel-tuning-lm-head-projection-timing-boundary-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this branch inherits broad unrelated workspace changes and the kept value-residual-gate vector-axis fix. This attempt is read-only unless the evidence proves the timing contract is wrong and a separate implementation note is written before editing.
- Prior-note/source search command:
  - `rg -n "lm_head|l2wrap|loss|timing_scope|canonical_compute|time.json|TraceTiming|record|perf_counter|timing" crates/rwkv-test/src crates/rwkv-nn/src crates/rwkv-test/examples .agents/notes/kernel-tuning -S`
- Matched prior evidence:
  - `2026-05-16-local-current-slowrow-audit.md` says `lm_head.time.json` is the final LayerNorm timing row, not the unembed matmul.
  - `2026-05-16-lm-head-forward-online-softmax-gb10.md` rejected changing the `lm_head_l2wrap_ce` row kernel after `nsys` showed the row kernel itself regressed.
  - `2026-05-16-remote-time-mixer-post-gates-nsys.md` and the latest `nsys-value-residual-gate-vector-axis` profile both show a large `matmul_entry...` group with `3` launches around `47ms` total in the clean profile, likely the lm_head projection matmul.
- Changed boundary: this is not another lm-head loss-kernel algorithm attempt. It checks whether the trace timing artifact used by `compare-rwkv-nn` includes the actual lm_head projection matmul before choosing more kernel work from `.time.json`.
- Machine/GPU: source inspection locally; if profiler data is needed, use remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`; no local GPU runs.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, vocab `65536`, rows `8192`, regenerated remote baseline under `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Command to run: inspect `crates/rwkv-test/src/rwkv_nn_trace/writer.rs`, `crates/rwkv-test/src/timing.rs`, and model/lm_head call paths to map `lm_head.time.json`, `lm_head/embedded_context.safetensors`, `lm_head/logits.safetensors`, and `loss/l2wrap_cross_entropy.time.json` to real compute boundaries.
- Expected decision boundary: if `lm_head.time.json` excludes the projection matmul, record the timing artifact as incomplete for kernel selection and open a separate timing-contract branch before changing trace output. If it does include projection, continue kernel tuning from profiler attribution instead.

## 2026-05-16 Source Inspection

- `crates/rwkv-test/src/rwkv_nn_trace/writer.rs` maps:
  - `lm_head.time.json` to `self.trace_layer_norm("lm_head", "lm_head/embedded_context", &model.layer_norm_for_unembed, embedded_context)`.
  - `model.unembed.forward(embedded_context)` runs immediately after that and has no `write_time(...)` call.
  - `loss/l2wrap_cross_entropy.time.json` wraps only `lm_head_l2wrap_ce(logits, targets)` after logits have already been produced.
- `crates/rwkv-nn/src/models/lm.rs` maps the real model boundary as final layer norm, then `self.unembed.forward(embedded_context_normalized)`, then `lm_head_l2wrap_ce(logits, targets)`.
- `crates/rwkv-test/src/timing.rs` treats `"lm_head"` and `"loss/l2wrap_cross_entropy"` as canonical timing rows, but there is no canonical timing row for the unembed projection or full lm-head projection+loss boundary.
- Interpretation: current `lm_head.time.json` excludes the largest lm_head projection matmul shown by `nsys`. The `.time.json` compare is therefore incomplete for choosing next kernel targets: it reports `lm_head` as a tiny row while profiler evidence shows the unembed projection is one of the largest GPU surfaces.
- Decision: close this branch as evidence. Do not pick next kernels from canonical `.time.json` totals alone until the trace timing contract is fixed or the missing projection is separately profiled. The implementation fix should be a new timing-contract branch that adds an explicit canonical timing row for the unembed projection or renames/splits the existing `lm_head` row without breaking output trace compatibility.

## 2026-05-16 Remote Baseline Artifact Check

- Remote baseline timing files under `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000/timing` contain only:
  - `lm_head.time.json`
  - `loss/l2wrap_cross_entropy.time.json`
- Remote actual timing files under `target/rwkv-test/rwkv_nn_actual/.../timing` also contain only those two lm-head/loss rows.
- Baseline `lm_head.time.json`: `elapsed_ns=573215`, samples `[532176, 392693, 794777]`, far below the `nsys` unembed projection matmul scale.
- Updated decision: this is a trace contract gap shared by actual and regenerated baseline. Adding a new canonical projection timing row only on the Rust actual path would make compare fail with a missing baseline row. A real fix needs the baseline generator/contract to emit the same row, or the Rust compare must keep the new projection row non-canonical until the baseline side is updated.
