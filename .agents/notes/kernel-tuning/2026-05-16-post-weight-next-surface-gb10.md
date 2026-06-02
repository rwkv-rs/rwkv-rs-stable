# Post Weight-Decay Next Surface GB10

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-post-weight-next-surface-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this branch inherits broad unrelated workspace changes plus the kept `weight_decay_transform` code and prior kernel notes. This attempt is read-only attribution first and must not edit kernels.
- Prior-note search commands:
  - `rg -n "wkv7|WKV7|mix6|channel_mixer|relu_square|gated_readout|key_prepare|weight_decay|ncu|occupancy|LocalTuner|autotune|row-pack|rowpack|warp|line_size|GB10|10\\.100\\.1\\.253" .agents/notes/kernel-tuning -S`
  - `rg -n "GatedReadout|gated_readout|warp32|32-thread|one-warp|row-pack|rowpack|two lanes|64-thread|combine" .agents/notes/kernel-tuning crates/rwkv-nn/src/kernels/train/time_mixer/gated_readout_combine crates/rwkv-nn/src/modules/time_mixer/gated_readout.rs -S`
  - `rg -n "WKV7|wkv7|row_tile|time split|chunk|state handoff|scan|row-tile" .agents/notes/kernel-tuning crates/rwkv-nn/src/kernels/train/time_mixer/wkv7 -S`
  - `rg -n "mix6|line_size|n_1|vector|coalesc|memory|matmul|scale" .agents/notes/kernel-tuning crates/rwkv-nn/src/kernels/train/time_mixer/mix6 -S`
- Matched prior evidence:
  - `2026-05-16-weight-decay-transform-gb10.md` kept the fused weight-decay transform after remote activation `54/54`, timing `76/76`, and total `2.18x`; ncu remained blocked by `ERR_NVGPUCTRPERM`.
  - `2026-05-16-remote-current-profile-gb10.md` identified WKV7, mix6, channel-mixer ReLU-square, lm-head row, GatedReadout combine, and key-prepare as large remaining surfaces, with several duplicate-guarded.
  - `2026-05-16-wkv7-remote-launch-design.md` and `2026-05-16-wkv7-row-tile16-forced.md` close row-tile-only WKV7 tuning; a real WKV7 split would need a separate algorithmic state-handoff/scan design.
  - `2026-05-16-gated-readout-warp32-gb10.md` rejects and reverts warp32; `2026-05-16-gated-readout-rowpack-gb10.md` is also closed.
  - `2026-05-16-key-prepare-warps-gb10.md` rejects the key-prepare warps-per-cube tuner and reverts to fixed 4 warps.
  - Channel-mixer Burn-reference/forced-Cube/fusion/LocalTuner-bypass directions are duplicate-guarded negative; mix6 line-size/axis forcing has already been inspected.
- Machine/GPU: remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`; do not use the local GPU.
- Remote run directory: `/home/caizus/Projects/Packages/rwkv-rs-stable`, synchronized non-git copy.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, regenerated remote baseline under `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Hypothesis: after `weight_decay_transform`, the remaining `kernel_binop_c_bf16_n_8` and adjacent generic kernels may point to another project-owned unfused expression. Use the existing post-weight remote nsys sqlite to locate their sequence context before choosing a new implementation branch.
- Candidate parameters: none in this read-only attempt.
- Expected keep/revert boundary: keep this note as attribution evidence. If the remaining surface is duplicate-closed or Cubek matmul internals, do not edit. If it identifies a new project-owned transform/fusion boundary, open a fresh implementation branch and note before code changes.
- Next command: query remote `target/rwkv-test/nsys-weight-decay-transform-gb10.sqlite` for ordered kernel windows around `kernel_binop_c_bf16_n_8`, `kernel_binop_c_bf16_n_1`, `reduce_kernel_in_bf16...`, and adjacent custom kernels.

## Ordered Kernel Query Attempt 1

- Command: remote Python `sqlite3` ordered-window query over `target/rwkv-test/nsys-weight-decay-transform-gb10.sqlite`.
- Result: invalid query. The remote `zsh` / heredoc quoting stripped Python string quotes, producing `SyntaxError: invalid syntax` at `mark=* if j==center else`.
- Next command: rerun the same read-only query with a double-quoted remote command and quoted heredoc delimiter.

## Ordered Kernel Query Attempt 2

- Command: remote Python `sqlite3` ordered-window query over `target/rwkv-test/nsys-weight-decay-transform-gb10.sqlite` with corrected heredoc quoting.
- Result: partial success. The ordered windows showed repeated sequences around `reduce_kernel_in_bf16*`, `kernel_binop_c_bf16_n_1`, `kernel_binop_c_bf16_n_8`, `gated_readout_combine_forward`, CubeCL matmul launches, `layer_norm_forward_kernel`, `channel_mixer_mix`, and `channel_mixer_relu_square_forward_kernel`. This confirms the remaining generic launches are spread across the time-mixer tail and channel-mixer entry/exit chain rather than a single obvious duplicate of the already kept `weight_decay_transform`.
- Invalid tail: the final grouped summary query failed with `sqlite3.OperationalError: no such column: total` because the SQL ordered by an aggregate alias that this sqlite expression did not accept.
- Decision: keep the ordered-window evidence as attribution only. Do not edit kernels from this partial result.
- Next command: rerun only the grouped summary with `ORDER BY 3 DESC` on remote `10.100.1.253`, using the same nsys sqlite file, to identify the largest remaining non-duplicate launch groups.

## Grouped Summary Attempt 1

- Command: remote Python grouped summary query over `target/rwkv-test/nsys-weight-decay-transform-gb10.sqlite`, using an embedded script inside a double-quoted `ssh` command.
- Result: invalid query. The remote `zsh` quoting again stripped Python string quotes, producing `SyntaxError` at `cur.execute(select name from sqlite_master...)`.
- Decision: no profiling conclusion from this run.
- Next command: rerun the grouped summary by piping a locally quoted heredoc into `ssh ... 'cd ... && python3 -'`, so the remote shell does not parse the Python body.

## Grouped Summary Attempt 2

- Command: remote Python grouped summary query over `target/rwkv-test/nsys-weight-decay-transform-gb10.sqlite`, piped through `ssh ... 'cd ... && python3 -'`.
- Result: query succeeded, but the `CUPTI_ACTIVITY_KIND_KERNEL.demangledName` column is a string-table id in this export. The top groups printed numeric ids such as `1230`, `1243`, `1251`, `1229`, and `1255`; id `1230` totaled `81.895 ms / 218 launches`, including one long `16.188 ms` launch. This is not yet actionable because names must be resolved through the Nsight string table.
- Decision: keep this as schema evidence only. Do not choose a kernel from numeric string ids.
- Next command: inspect string table schema and rerun the grouped summary joined against the string table to print demangled kernel names, grid/block, register, and shared-memory metadata.

## Resolved Grouped Summary

- Command: remote Python query over `target/rwkv-test/nsys-weight-decay-transform-gb10.sqlite`, joining `CUPTI_ACTIVITY_KIND_KERNEL.demangledName` to `StringIds`.
- Result: resolved top kernel groups. Largest totals:
  - CubeCL bf16 matmul variants: `31.777 ms / 2`, `17.906 ms / 96`, `13.906 ms / 24`, `12.526 ms / 24`, plus smaller groups. These are shared matmul surfaces, not a narrow train-kernel edit.
  - `wkv7_pretrain_forward_output_kernel_f_bf16`: `15.252 ms / 24`, duplicate-guarded unless using a new state-handoff/scan design.
  - `mix6_forward_kernel_f__n_1`: `9.666 ms / 24`, duplicate-guarded for line-size/axis retry on GB10.
  - `channel_mixer_relu_square_forward_kernel_f__n_2`: `9.345 ms / 24`, current GB10 tuner winner; duplicate-guarded for forced Cube/Burn/reference retries.
  - `lm_head_l2wrap_ce_forward_row_kernel_f__i_i32`: `8.511 ms / 2`, duplicate-guarded for previous atomic/target-logit/online-softmax/prune attempts unless the reduction design changes.
  - `kernel_binop_c_bf16_n_8`: `8.484 ms / 96` plus `1.527 ms / 24`; `kernel_binop_c_bf16_n_1`: `4.047 ms / 48`; `reduce_kernel_in_bf16_in_size_8_out_bf16_out_size_8_acc_f32`: `3.781 ms / 48`.
  - `gated_readout_combine_forward_kernel_f_`: `7.832 ms / 24`, duplicate-guarded for warp32/rowpack.
  - `key_prepare_forward_64_kernel_f_`: `6.405 ms / 24`, duplicate-guarded for warps-per-cube tuning.
  - `layer_norm_forward_kernel_f_`: `4.348 ms / 52`, accuracy-sensitive; ordered-256 and split-tail smaller reductions are closed as drift failures.
- Decision: do not open branches for the already closed top custom kernels from this grouped view. The next non-duplicate investigation is to map the remaining generic `kernel_binop*` and `reduce_kernel*` launch windows to model-stage code.
- Next command: query ordered, resolved kernel windows around `kernel_binop_c_bf16_n_8`, `kernel_binop_c_bf16_n_1`, and `reduce_kernel_in_bf16*` to identify whether they are LoRA/weight-prepare/channel-mixer glue, CubeCL matmul epilogue work, or another project-owned fusion boundary.

## Generic Kernel Window Query

- Command: remote ordered-window query over resolved kernel names around `kernel_binop_c_bf16_n_8`, `kernel_binop_c_bf16_n_1`, and `reduce_kernel_in_bf16*`.
- Result: `216` matching centers collapsed into `10` unique repeated window signatures. The repeated per-cell sequence includes LoRA/matmul-adjacent kernels, then `gated_readout_combine_forward_kernel_f_`, projection matmul, `layer_norm_forward_kernel_f_`, `channel_mixer_mix_forward_kernel_f__n_8`, channel-mixer matmul, `channel_mixer_relu_square_forward_kernel_f__n_2`, and another matmul. Representative windows show:
  - `reduce_kernel_in_bf16* -> kernel_scalar_binop_c_bf16_n_1 -> kernel_binop_c_bf16_n_1 -> kernel_binop_c_bf16_n_8 -> reduce_kernel_in_bf16*`.
  - `kernel_binop_c_bf16_n_1 -> kernel_binop_c_bf16_n_8 -> kernel_binop_c_bf16_n_8 -> gated_readout_combine_forward_kernel_f_ -> matmul`.
  - `gated_readout_combine_forward_kernel_f_ -> matmul -> kernel_binop_c_bf16_n_8 -> layer_norm_forward_kernel_f_ -> channel_mixer_mix_forward_kernel_f__n_8`.
  - `channel_mixer` tail `matmul -> channel_mixer_relu_square_forward_kernel_f__n_2 -> matmul -> kernel_binop_c_bf16_n_8 -> layer_norm_forward_kernel_f_`, with two final lm-head windows.
- Source cross-check: `weight_prepare.rs` uses three LoRA paths around `projection_*`, `learning_rate_gate`, `value_residual_gate`, `weight_decay_transform`, and `key_prepare`; `gated_readout.rs` uses output-gate LoRA, group norm, `gated_readout_combine`, and output projection; `channel_mixer` already has a custom primitive but still relies on backend matmul between elementwise stages.
- Current inference: the generic launches are likely LoRA/gate/projection glue spanning `weight_prepare`, `gated_readout`, and channel-mixer residual/layer-norm boundaries, not a single already named custom kernel.
- Decision: still no code edits. Need NVTX/range correlation before choosing a new branch, otherwise this risks becoming another duplicated or overly broad fusion attempt.
- Next command: inspect the nsys sqlite for NVTX/range tables and, if present, correlate the generic kernels to `rwkv.infer.model.*` ranges.

## Range Correlation Check

- Command: inspect range-like tables in `target/rwkv-test/nsys-weight-decay-transform-gb10.sqlite`.
- Result: this export has no NVTX/tracing range table. Range-like tables are limited to CUDA events, diagnostic events, and enum metadata.
- Source-order attribution:
  - The `reduce_kernel_in_bf16* -> scalar/binop -> binop` repeated group immediately before `gated_readout_combine_forward_kernel_f_` maps to `GatedReadout::forward`: Burn `GroupNorm` over the WKV7 output, then the custom gated-readout combine.
  - The `kernel_binop_c_bf16_n_8` after gated-readout projection and after channel-mixer output maps to residual/tensor-add boundaries in the trace writer and surrounding module flow. This is duplicate-guarded by the existing residual-add notes and must not be rerun as another plain residual experiment.
  - The remaining LoRA/matmul-adjacent generic kernels are broad low-rank projection glue; fusing them would be a larger LoRA design, not the smallest next kernel step.
- Decision: close this read-only attribution attempt with a concrete next boundary: a new branch for fusing gated-readout GroupNorm with the existing gated-readout combine. This is materially different from the closed `gated_readout` warp32/rowpack attempts because it changes the operation boundary and targets the preceding Burn GroupNorm generic reductions.
- Keep/revert state: no code edits were made on this branch.
