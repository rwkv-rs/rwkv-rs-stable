# Gated Readout Combine Analysis

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-gated-readout-combine-analysis-20260516` in the existing dirty workspace.
- Dirty-tree constraint: this checkout carries broad unrelated uncommitted changes, the kept TimeMixer gate wiring, the temporary LayerNorm diagnostic example, and the evidence-only WKV7 note. This attempt is analysis first and must not edit live kernels until the forward/backward boundary is proven.
- Prior-note search command:
  - `rg -n "gated_readout|GatedReadout|gate_input|output_gate|projection_output|group_norm|bonus|combine|full backward" .agents/notes/kernel-tuning crates/rwkv-nn/src/modules/time_mixer crates/rwkv-nn/src/kernels/train -S`
- Matched prior evidence:
  - `2026-05-16-wire-time-mixer-gates-gb10.md` is the kept remote high point: activation passed, all timing rows passed, `actual_total_ms=91.595`, baseline `175.887`, speedup `1.92x`.
  - `2026-05-16-local-post-gates-audit.md` says a possible next non-duplicate branch is a `GatedReadout` combine kernel with full backward coverage.
  - `2026-05-16-wkv7-remote-launch-design.md` closed WKV7 as evidence-only and identified `GatedReadout::forward` as a clearer project-owned fusion boundary.
  - Do not repeat residual-add A/B, channel-mixer forced Cube/Burn-reference/fusion, LocalTuner bypass, WKV7 forced `row_tile=16`, or lm-head row-kernel variants.
- Machine/GPU: remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1` for acceptance; no local GPU runs while the user is using the machine.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, `num_heads=12`, `head_size=64`, regenerated remote baseline under `~/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Kernel/stage: TimeMixer `GatedReadout::forward` after WKV7 and output-gate LoRA, before output projection.
- Hypothesis: keep LoRA and output projection matmuls in Burn/Cubek, but fuse the project-owned bonus/out-gated chain:
  - `bonus = sum(receptance * replacement_key * bonus_param, dim=head_size) * value`
  - `out_gated = (group_norm_output + bonus) * gate`
  This may reduce generic BF16 elementwise/reduction/broadcast launches without touching tensor-core matmul paths.
- Candidate parameters: none yet. First inspect backward requirements, whether `GroupNorm` backward/output projection gradients require intermediate tensors, and whether a custom op can save enough state without expanding memory.
- Expected keep/revert boundary: keep this note as analysis evidence unless a complete forward+backward candidate is identified. Do not wire a forward-only custom op into train code if it would break autodiff or require unverified duplicate backward logic.
- Next command: inspect existing custom train-kernel forward/backward integration patterns and the `GatedReadout` source-level gradient dependencies.

## 2026-05-16 continuation

- User constraint: debug first on remote `10.100.1.253`; do not run local GPU because the local GPU is reserved for other work and timing may be inaccurate.
- Ran local/remote preflight:
  - `git status --short --branch`
  - `sed -n '1,260p' crates/rwkv-nn/src/modules/time_mixer/gated_readout.rs`
  - `ssh -i ~/.ssh/id_ed25519_dgx_spark_windows caizus@10.100.1.253 'ls -la ~/Projects ...; find ...'`
- Remote path finding result: `/home/caizus/Projects/Packages/rwkv-rs-stable` and `/home/caizus/Projects/Packages/rwkv-rs-test` exist, but the remote stable copy does not contain `.git`. Treat remote as a synchronized run directory and keep branch/ledger ownership in the local checkout.
- Source inspection result:
  - `GatedReadout::forward` computes output-gate LoRA first, Burn group norm, then `bonus = sum(receptance * replacement_key * bonus_param, dim=3) * value`, then `out_gated = (group_norm_output + bonus) * gate`, then output projection.
  - A forward-only custom op is insufficient for training: backward must return gradients for `receptance`, `replacement_key`, `value`, `param_receptance_key_bonus`, `group_norm_output`, and `gate`. That is materially larger than the existing value-residual-gate elementwise backward.
- Current decision boundary: query the remote post-gates nsys artifacts before implementing. If this chain is not a measurable hotspot, close this as evidence-only rather than adding a high-risk duplicate backward.
- Next command: locate and query remote `nsys` SQLite reports for current post-gates kernel/API time distribution.

## Remote nsys evidence

- Remote artifact located:
  - `/home/caizus/Projects/Packages/rwkv-rs-stable/target/rwkv-test/nsys-time-mixer-gates-gb10.sqlite`
- Top post-gates CUDA kernel groups from that artifact:
  - `matmul_entry_lhs_bf16_lhs_size_1...`: 327 launches, `120.828 ms` total.
  - `kernel_binop_c_bf16_n_8`: 360 launches, `31.778 ms` total.
  - `wkv7_pretrain_forward_output_kernel_f_bf16`: 36 launches, `22.523 ms` total.
  - `mix6_forward_kernel_f__n_1`: 36 launches, `14.427 ms` total.
  - `channel_mixer_relu_square_forward_kernel_f__n_4`: 36 launches, `14.065 ms` total.
  - `layer_norm_forward_kernel_f_`: 78 launches, `6.393 ms` total.
- Ordered sequence after a representative `wkv7_pretrain_forward_output_kernel` shows a likely `GatedReadout` region:
  - output-gate LoRA and sigmoid: matmul/cast/scalar/unary/matmul, about `0.23 ms`.
  - group norm and bonus/out-gated generic kernels: reduce/binop/scalar/unary sequence through the output projection, about `1.13 ms` before the output projection matmul.
  - output projection matmul: about `0.16 ms` for that cell in the representative slice.
- Analysis decision: the chain is measurable, but a full custom backward is large. A lower-risk implementation branch can fuse only the non-autodiff forward path used by `compare-rwkv-nn`, while `Autodiff<...>` keeps the reference Burn graph. This avoids an unverified duplicate backward while testing whether launch/memory-traffic reduction helps the remote forward benchmark.
- Keep/revert decision for this branch: close as evidence-only. New implementation needs a fresh branch and note because the hypothesis changes from analysis to forward-only dispatch with autodiff reference fallback.
