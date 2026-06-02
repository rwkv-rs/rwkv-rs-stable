# Local Short-Kernel Provenance Diagnosis

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-local-short-kernel-provenance-20260516` in the existing dirty workspace.
- Dirty-tree constraint: this checkout carries broad unrelated uncommitted workspace changes plus the current remote-clean tuning tree. This branch is read-only unless a later note explicitly opens a narrowly scoped code branch.
- Prior-note/source search command:
  - `rg -n "local.*0\\.92|short-kernel|pre_layer_norm|channel_mixer|baseline provenance|LocalTuner|direct.*channel mixer|custom channel mixer|lm_head|l2wrap" .agents/notes/kernel-tuning /root/.codex/memories/MEMORY.md -S`
- Matched prior evidence:
  - `2026-05-16-local-acceptance-after-idle-check.md` records local correctness pass but timing failure: `actual_total_ms=38.884`, `baseline_total_ms=35.957`, `speedup=0.92x`.
  - `2026-05-16-local-post-gates-audit.md` already attributes local post-gates failure mostly to Cubek/Burn BF16 matmul, WKV7, LayerNorm, and lm-head surfaces; it explicitly says not to rerun residual-add, channel-mixer forced Cube/Burn-reference, LocalTuner bypass, target-logit, atomic, online-softmax, or prune-256 attempts.
  - `2026-05-16-channel-mixer-ncu.md` shows channel-mixer custom mix/ReLU-square kernels are memory-bound but not the main module bottleneck; the two channel-mixer matmuls dispatch to the scalar-load `lhs_size_1/rhs_size_1` family.
  - `2026-05-16-layernorm-ordered256-post-gates.md` rejected ordered-256 because activation failed in the known `value_from_first_cell` / `lm_head/embedded_context` drift family.
  - Memory says previous local compare baselines below `1.0` require checking baseline provenance, device, and toolchain drift before accepting a remote conclusion as local.
- Machine/GPU: local `NVIDIA GeForce RTX 5090`, compute capability `12.0`.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, local baseline under `crates/rwkv-test/test_data/rwkv_lm/bf16/case_000000`.
- Kernel/stage: local failing short-kernel/module rows: `pre_layer_norm_for_*`, `embedded_context_after_*`, `channel_mixer`, `lm_head`, and `loss/l2wrap_cross_entropy`.
- Hypothesis: the local `0.92x` result is not an activation or ordered-256 issue. It is either a baseline provenance mismatch or an implementation-boundary issue dominated by short module timings and Cubek matmul dispatch choices that prior negative branches already constrain.
- Candidate parameters: none in this branch.
- Expected keep/revert boundary: keep only source/provenance attribution. Do not run another broad local compare or rerun duplicate implementation attempts from the matched evidence.
- Next command: inspect the timing contract, trace writer boundaries, and selected module source paths to identify a non-duplicate next code target.

## 2026-05-16 Timing Boundary Inspection

- `crates/rwkv-test/src/rwkv_nn_trace/writer.rs` times all relevant rows with an explicit `B::sync(device) -> work -> B::sync(device)` boundary.
- `pre_layer_norm_for_*` calls the project-owned `layer_norm(input, gamma, beta, 1e-5)` wrapper and records only timing unless the activation snapshot is needed.
- `embedded_context_after_*` remains plain Burn `lhs + rhs` inside `trace_residual_add`; previous custom residual call-site wiring is already rejected and should not be repeated here.
- `channel_mixer` uses the project-owned custom wrapper only on the stateless train path: custom `channel_mixer_mix`, Cubek `float_matmul`, custom `channel_mixer_relu_square`, Cubek `float_matmul`.
- `lm_head` timing currently means final LayerNorm only; `lm_head/projection` is emitted as noncanonical timing-only, and loss timing covers `lm_head_l2wrap_ce(logits, targets)`.
- Interpretation:
  - The local failure is not caused by timing JSON including safetensors writes; the measured boundaries sync only the module work.
  - The next non-duplicate target is Cubek matmul dispatch/provenance for channel_mixer and lm_head, plus timing-baseline provenance. Do not reopen residual or ordered-256.
- Next command:
  - Inspect local autotune cache entries for matmul/channel_mixer/layer_norm dispatch choices around the failing shapes.

## 2026-05-16 Autotune Cache Inspection

- LayerNorm local active key:
  - `runtime=cuda`, `dtype=BF16`, `d_model=768`, `rows=8192`, `num_streaming_multiprocessors=170`, `deterministic=true`, `deterministic_min_block_size=1024`.
  - Selected `block_1024`; smaller blocks are skipped. This matches the clean activation result.
- Channel mixer elementwise cache:
  - `channel-mixer-mix-forward` has current hardware-rich key for `rows=8192`, `innermost_dim=768`, `max_line_size=8`; selected `line_size_2`.
  - `channel-mixer-relu-square-forward` has current hardware-rich key for `rows=8192`, `innermost_dim=3072`, `max_line_size=8`, `is_in_place=true`; selected `line_size_8`.
  - Older short keys without full hardware fields are still present in the cache logs, but the current hardware-rich keys exist and are used by the present code boundary.
- Cubek matmul cache:
  - Cache includes the large `m=512,n=65536,k=1024` lm-head projection key, selecting `matmul_specialized_tma_mma` with median about `4.607ms`.
  - Cache includes medium channel/time projection families such as `m=512,n=1024,k=1024`, `m=512,n=64,k=1024`, `m=512,n=128,k=1024`, and smaller `k=64/32` variants.
  - This confirms the remaining local channel/lm-head surfaces are mostly Cubek matmul selection and module-boundary behavior, not missing project-owned elementwise autotune keys.
- Interpretation:
  - The local `0.92x` result is not fixed by reopening channel mixer line-size tuning or LayerNorm block-size tuning.
  - The only plausible local code targets left under the current constraints are broader Cubek matmul-shape dispatch/provenance work or changing the benchmark contract/baseline provenance. Both are bigger than a small kernel knob.
- Next command:
  - Inspect local baseline timing file history/provenance for representative failing rows.

## 2026-05-16 Baseline Provenance Inspection

- Representative local baseline timing files are clean in git status and were introduced by `c2d3482 refactor(rwkv-test): split trace comparison modules`.
- File mtimes are `2026-05-15 10:43:53..56 +0800`, with no local modifications for the inspected rows.
- `crates/rwkv-test/README.md` says the default baseline is the repo fixture under `crates/rwkv-test/test_data/rwkv_lm/bf16/case_000000`, produced by the companion `rwkv-rs-test` trace workflow.
- Interpretation:
  - The local baseline is a checked-in fixture, not a fresh run on the current code path.
  - Treating the local `0.92x` as a real acceptance failure is correct for this repository contract, but it should not be confused with remote regenerated-baseline evidence.
- Decision:
  - Close this branch as read-only provenance evidence. The only non-duplicate implementation path with enough upside is a corrected LayerNorm ordered-256 candidate, because prior CPU diagnostics showed the intended ordered math is exact versus `1024` while the rejected post-gates kernel likely implemented the device path incorrectly or used a stale boundary.
