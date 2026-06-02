# Binop Attribution GB10

- Date: 2026-05-16 20:50 +0800.
- Branch/worktree: `kernel-tuning-binop-attribution-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits plus retained/reverted kernel tuning changes. This attempt owns only this attribution note unless a later entry explicitly opens an implementation branch.
- User constraint: tune/debug first on remote `10.100.1.253`; do not use local GPU timing.
- Prior-note search commands:
  - `rtk rg -n "kernel_binop_c|binop_c_bf16|binop|binary|add_scalar|mul_scalar|bonus|receptance|embedded_context_after|after_time_mixer|after_channel_mixer" .agents/notes/kernel-tuning/2026-05-16-*.md .agents/skills/kernel-tuning/SKILL.md`
  - `rtk rg -n "kernel_binop_c_bf16_n_8|kernel_binop_c|binop_c" target -S`
  - `rtk rg -n "bonus|receptance|group_norm|gated_readout|value_residual|learning_rate|embedded_context_after|time_mixer.*\\+|\\* .*gate|sigmoid|square|powf|exp\\(" crates/rwkv-nn/src/crates crates/rwkv-nn/src`
- Matched prior evidence:
  - Current remote `nsys` rankings repeatedly show `kernel_binop_c_bf16_n_8` as a remaining generic Burn kernel family, around `8.5ms / 72` before the latest WKV7 direct-value attempt and `5.892ms / 48` after the kept GatedReadout GroupNorm+combine fusion.
  - `2026-05-16-gated-readout-groupnorm-combine-gb10.md` records that the fused GatedReadout kernel removed the targeted Burn GroupNorm reduce/scalar/binop groups, but `kernel_binop_c_bf16_n_8` remained in the grouped output.
  - Existing duplicate guards close residual-add/Burn-add A/B, GatedReadout row-pack/warp32/sumsq variants, WKV7 row-tile/shared-lanes/segment/factored/direct-value attempts, key_prepare warps, lm_head row variants, and channel-mixer forced Cube/Burn/fusion/line-size retries.
- Machine/GPU: remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`. Remote `ncu` counters are blocked by `ERR_NVGPUCTRPERM`, so this branch uses existing `nsys` sqlite timeline metadata only.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`.
- Hypothesis: the remaining `kernel_binop_c_bf16_n_8` launches may come from a still-unfused project expression, or from generic Burn elementwise work that is too broad/noisy to target. Before changing code, map these launches to adjacent kernels/timeline context and source expressions.
- Candidate parameters: none. This is read-only attribution against existing remote profiler artifacts.
- Expected keep/revert boundary: keep this note as evidence. If timeline adjacency identifies a non-duplicate, project-owned expression with enough repeated time to matter, open a fresh implementation branch and note before editing. If the launches are broad generic Burn residual/module glue or already-covered boundaries, do not edit kernels from this evidence.
- Next command: query existing remote `target/rwkv-test/*.sqlite` artifacts on `10.100.1.253` for `kernel_binop_c_bf16_n_8` launch timestamps and neighboring kernel names; do not run a new benchmark/profile workload.

## Remote Artifact Listing

- Command: `ssh -i ~/.ssh/id_ed25519_dgx_spark_windows ... 'cd ~/Projects/Packages/rwkv-rs-stable && find target/rwkv-test -maxdepth 1 -name "*.sqlite" ... | tail -20'`.
- Result: explicit-key SSH succeeded on `spark-35ac`. Recent existing sqlite artifacts include `nsys-current-after-gb10-results.sqlite` from `2026-05-16 07:30` and `nsys-wkv7-direct-value-gb10.sqlite` from `2026-05-16 07:39`.
- Artifact choice: use `target/rwkv-test/nsys-current-after-gb10-results.sqlite` as the current accepted/reverted source attribution. Avoid `nsys-wkv7-direct-value-gb10.sqlite` because that profile contains the rejected direct-value WKV7 implementation.
- Next command: query `nsys-current-after-gb10-results.sqlite` for every `kernel_binop_c_bf16_n_8` launch with neighboring kernels in timestamp order.

## Invalid Query Attempt

- Invalid command: nested `ssh 'python3 - <<PY ... printf('<%s>', ...) ... PY'`.
- Failure: local shell quoting broke on the SQL string containing `<%s>` and exited with `/bin/bash: line 1: %s: No such file or directory`.
- Validity: no remote query result was produced and no profiler conclusion is drawn.
- Next command: rerun the same read-only sqlite query through `ssh ... 'bash -s' <<'REMOTE'` to avoid nested quoting.

## Timeline Query Result

- Command: remote `python3` sqlite query over `target/rwkv-test/nsys-current-after-gb10-results.sqlite`, grouped and sampled around `kernel_binop_c_bf16_n_8`.
- Result:
  - `total_kernels=1320`.
  - `kernel_binop_c_bf16_n_8`: `72` launches, `8.579ms` total, `119.160us` average.
  - Every one of the `72` launches has the same immediate neighbors:
    - previous: `matmul_entry_lhs_bf16_lhs_size_1_rhs_bf16_rhs_size_1_acc_bf16_acc_size_8`.
    - next: `layer_norm_forward_kernel_f_`.
  - Launch geometry: `grid=(3072,1,1)`, `block=(32,8,1)`, `16` registers/thread, no dynamic shared memory.
- Interpretation:
  - The count `72` equals `2 * 36`, so this is very likely the two residual additions per cell per profiled iteration: after time-mixer output projection and after channel-mixer output projection.
  - Timeline examples support that: one pattern is `gated_readout_combine -> output projection matmul -> binop -> layer_norm -> channel_mixer`, and the other is `channel_mixer_relu_square -> value projection matmul -> binop -> layer_norm -> next time_mixer`.
  - This is not a leftover GatedReadout GroupNorm binop; it is residual module glue adjacent to projection matmuls and the next LayerNorm.
- Next command: inspect `crates/rwkv-nn/src/cells/causal.rs`, current residual-add notes/source, and the timing rows for `embedded_context_after_time_mixer` / `embedded_context_after_channel_mixer` to decide whether this exact residual boundary is a duplicate of the rejected custom residual wiring or a new fused-projection/add boundary.

## Source And Duplicate Check

- Commands:
  - `rtk sed -n '180,260p' crates/rwkv-nn/src/cells/causal.rs`.
  - `rtk sed -n '150,230p' crates/rwkv-nn/src/modules/time_mixer/mod.rs`.
  - `rtk sed -n '1,120p' .agents/notes/kernel-tuning/2026-05-16-wire-custom-residual-add.md` and `2026-05-16-residual-add-hardware-key.md`.
  - `rtk rg -n "embedded_context_after_time_mixer|embedded_context_after_channel_mixer|residual_add|kernel_binop_c_bf16_n_8|cells/\\*/embedded_context_after" ...`.
- Source mapping:
  - `CausalCell::forward` uses plain Burn tensor addition for both residuals:
    - `embedded_context + time_mixer_output.embedded_context`.
    - `embedded_context + channel_mixer_output.embedded_context`.
  - The preceding matmul in the timeline maps to TimeMixer output projection or ChannelMixer value projection, then the residual add, then the next LayerNorm.
- Duplicate evidence:
  - `2026-05-16-wire-custom-residual-add.md` already wired a custom `residual_add(...)` kernel into these exact model and trace-writer call sites. It passed activation but regressed timing and was reverted.
  - `2026-05-16-residual-add-hardware-key.md` already added runtime/hardware/shape/vector/alias/deterministic keying to the custom residual path. The missing piece is not a key-field gap.
  - `2026-05-16-post-weight-next-surface-gb10.md` already partially mapped `kernel_binop_c_bf16_n_8` to residual/tensor-add boundaries, but this branch refreshes the result on the current post-target-logit accepted source and shows the immediate-neighbor pattern is now exactly `72/72`.
- Decision:
  - Do not open another plain residual-add or trace-writer residual branch. That would repeat a recorded negative result.
  - A materially different future attempt would have to fuse the residual add into the preceding Cubek/TMA matmul epilogue, or otherwise eliminate the standalone output tensor/write in a GEMM-quality path. That is a broad matmul-epilogue boundary, similar in risk to the documented projection+loss fusion blocker.
  - Close this branch as read-only attribution and duplicate-guard evidence.
- Next edit: update `.agents/skills/kernel-tuning/SKILL.md` with a concise guard for `kernel_binop_c_bf16_n_8` so future agents do not treat it as a fresh generic-binop optimization target.

## Skill Guard Update

- Edited `.agents/skills/kernel-tuning/SKILL.md`.
- Added a known GB10 attribution guard:
  - current `kernel_binop_c_bf16_n_8` is the pair of residual adds after TimeMixer and ChannelMixer projection matmuls;
  - do not repeat custom residual wiring or treat it as a fresh generic-binop target;
  - a materially different attempt must change the projection/matmul epilogue boundary.
- Keep/revert state: keep the note and skill update. No kernel source changed.
- Next command: run `git diff --check` for the touched note and skill, then sync both files to `10.100.1.253`.

## Validation And Sync

- Command: `git diff --check -- .agents/skills/kernel-tuning/SKILL.md .agents/notes/kernel-tuning/2026-05-16-binop-attribution-gb10.md`.
- Result: passed.
- Remote sync command: scoped `rsync -azR` of the updated skill and this note to `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable/`.
- Result: passed.
- Final decision: close this attribution attempt. `kernel_binop_c_bf16_n_8` is residual-add glue on the current GB10 source, and the plain residual-add implementation path is duplicate-guarded by prior negative evidence. No kernel code was changed.
