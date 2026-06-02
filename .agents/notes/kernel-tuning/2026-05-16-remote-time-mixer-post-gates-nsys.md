# Remote Time Mixer Post-Gates Nsys

- Date: 2026-05-16
- Branch/worktree: `kernel-tuning-remote-time-mixer-post-gates-nsys-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this branch inherits broad unrelated workspace changes and the kept 64-thread/two-warp GatedReadout combine code. This attempt is read-only profiler attribution first; it must not edit kernels before the profiler/source evidence identifies a non-duplicate implementation boundary.
- Prior-note search command:
  - `rg -n "time_mixer|post-gates|nsys|mix6|key_prepare|value_residual|gated_readout|wkv7|learning_rate" .agents/notes/kernel-tuning | head -200`
- Matched prior evidence:
  - `2026-05-16-gated-readout-forward-combine-gb10.md` has the clean remote `nsys-gated-readout-forward-combine-clean.sqlite` profile after autotune cache cleanup. Top surfaces were `wkv7_pretrain_forward_output_kernel_f_bf16` `22.732ms / 36`, `mix6_forward_kernel_f__n_1` `14.198ms / 36`, `channel_mixer_relu_square_forward_kernel_f__n_4` `13.666ms / 36`, and `gated_readout_combine_forward_kernel_f_` `11.793ms / 36`.
  - `2026-05-16-wkv7-remote-launch-design.md` rejected a row-tile-only WKV7 retry: the source keeps row state in registers for the full context, and forced `row_tile=16` was already worse.
  - `2026-05-16-mix6-forward-hardware-key.md` restored mix6 forward LocalTuner/hardware key locally; this is a key-design fix, not proof of a new math or launch candidate.
  - `2026-05-16-lm-head-forward-online-softmax-gb10.md` rejected the online-softmax loss row on GB10 because `nsys` showed the changed row kernel was slower.
  - `2026-05-16-remote-channel-mixer-edge-samples.md` shows the latest current JSON has `actual_total_ms=84.876`, `baseline_total_ms=175.887`, `speedup=2.072`, one narrow `cell_0000/channel_mixer` fail, and the largest remaining actual timing rows are `cells/*/time_mixer` plus `loss/l2wrap_cross_entropy`.
- Machine/GPU: remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`; no local GPU runs.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, regenerated remote baseline under `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Command to run first: query existing remote `target/rwkv-test/nsys-gated-readout-forward-combine-clean.sqlite` and, if needed, current remote `rwkv_nn_actual` timing JSON. Do not launch `nsys` again unless existing profiler data is stale for the current binary.
- Expected decision boundary: pick the next implementation branch only if the profiler points to a project-owned kernel/dispatch boundary not already covered by the WKV7 row-tile, GatedReadout warp32, residual-add, channel-mixer, or lm-head negative notes. If the top remaining cost is a known-hard recurrence or third-party matmul, close this as attribution and choose a different documented gap.

## 2026-05-16 Existing Nsys And Source Attribution

- Existing profile queried: `/home/caizus/Projects/Packages/rwkv-rs-stable/target/rwkv-test/nsys-gated-readout-forward-combine-clean.sqlite`.
- Top kernel groups:
  - `matmul_entry_lhs_bf16_lhs_size_1_rhs_bf16_rhs_size_1_acc_bf16_acc_size_8`, `3` launches, `47.232ms`, grid `4096x16x1`, block `32x12x1`.
  - `matmul_entry_lhs_bf16_lhs_size_1_rhs_bf16_rhs_size_1_acc_bf16_acc_size_8`, `144` launches, `24.880ms`, grid `16x48x1`, block `32x12x1`.
  - `wkv7_pretrain_forward_output_kernel_f_bf16`, `36` launches, `22.732ms`, grid `12x16x1`, block `64x1x1`, `108` registers/thread.
  - `mix6_forward_kernel_f__n_1`, `36` launches, `14.198ms`, grid `24576x1x1`, block `32x8x1`, `32` registers/thread.
  - `channel_mixer_relu_square_forward_kernel_f__n_2`, `36` launches, `13.818ms`.
  - `lm_head_l2wrap_ce_forward_row_kernel_f__i_i32`, `3` launches, `13.502ms`.
  - `gated_readout_combine_forward_kernel_f_`, `36` launches, `11.793ms`, grid `8192x12x1`, block `64x1x1`, `21` registers/thread.
  - `key_prepare_forward_64_kernel_f_`, `36` launches, `9.437ms`.
  - `value_residual_gate_forward_kernel_f__n_1`, `33` launches, `6.620ms`.
- Mix6 autotune log: remote `rwkv7-time-mixer-mix6-forward` key has `max_line_size=8`, but selected `line_size_1`. Candidate medians are close: `line_size_1` `411.554us`, `line_size_2` `416.770us`, `line_size_4` `417.617us`, `line_size_8` `416.817us`. The `n_1` suffix is a real selected candidate, but not an obvious bug because the tuner measured it slightly fastest on GB10.
- Value residual gate autotune log: remote `value-residual-gate-forward` key has `max_line_size=1`, so every vectorized candidate was skipped and only `line_size_1` ran (`220.081us` median).
- Source inspection result: `value_residual_gate/forward.rs` computes `max_line_size_many(&[value, value_from_first_cell, gate_base, gate_input], shape.num_dims() - 1)`. For the 3D tensors this axis is embedded dimension `2`, but `gate_base` is 1D `[embedded_dim]`, so including it in the same axis calculation collapses `max_line_size` to `1`. This is a plausible implementation bug in vector-width eligibility, not a numerical-algorithm change.
- Decision: close this branch as read-only attribution. Do not retry WKV7 row tiling, GatedReadout warp32, Mix6 line-size forcing, channel mixer, or lm-head online softmax from this evidence. The next implementation branch should fix only `value_residual_gate` forward vector-width eligibility by computing the base tensor vector width on its own embedded axis, then let the existing LocalTuner choose among candidates on GB10.
