# Forward Current ncu After Key Fixes

- Date: 2026-05-16
- Branch/worktree: `kernel-tuning-forward-current-ncu-after-keys-20260516` in the existing dirty workspace.
- Dirty-tree constraint: this profiling branch runs on the accumulated local key-design tree after `mix6` forward and custom `residual_add` line-size key changes. It is read-only except for this note and profiler output files.
- Prior-note search command: `rg -n "forward slow|slowgroup|channel_mixer|layer_norm|lm_head|l2wrap|ncu|SpeedOfLight|Occupancy|MemoryWorkload|SchedulerStats|WarpState" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md /root/.codex/memories/MEMORY.md`.
- Matched prior evidence:
  - `2026-05-16-forward-slowgroup-current-ncu.md` already captured channel mixer, layer norm, lm_head, and loss kernels before the latest key fixes.
  - `2026-05-16-channel-mixer-matmul-fusion-analysis.md` concluded that local channel-mixer matmul fusion is not a good next implementation patch without extending Cubek matmul.
  - `2026-05-16-lm-head-forward-atomic-loss.md` records the atomic-loss forward candidate as activation-safe but timing-negative and reverted.
  - `2026-05-16-layernorm-safe-path-analysis.md` records combined LayerNorm reductions as activation-safe but timing-negative and reverted.
- Scope: current remaining local forward slow groups after key fixes: `channel_mixer`, pre-layer-norm, `lm_head`, and `loss/l2wrap_cross_entropy`.
- Command planned: `ncu --target-processes all --kernel-name regex:'.*(channel_mixer|layer_norm|lm_head|l2wrap|matmul).*' --launch-count 80 --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats --csv --log-file target/rwkv-test/ncu-forward-current-after-keys.csv target/release/rwkv-test compare-rwkv-nn --color never --repeat 1 --warmup 1`.
- Expected keep/revert boundary: keep profiler output and note as evidence only. Compare `.time.json` rows from an ncu run are invalid because profiler instrumentation changes timing.

## 2026-05-16 Continuation

- Prior evidence checked before continuation: `rg -n "residual|layer_norm|BLOCK_SIZE|256|1024|Burn add|ncu-forward|forward-current" .agents/notes/kernel-tuning` found the residual duplicate guard, local LayerNorm block-size drift notes, and earlier forward slowgroup ncu summaries.
- First ncu result: `target/rwkv-test/ncu-forward-current-after-keys.csv` captured 80 launches but did not reach `lm_head/l2wrap` because LayerNorm, ChannelMixer, and CubeCL matmul launches consumed the count first. The compare timing emitted during this profiler run is invalid for speedup conclusions because ncu instrumentation changes runtime and the command intentionally used `repeat=1`.
- Current captured medians:
  - `matmul_entry_lhs_bf16_lhs_size_1_rhs_bf16_rhs_size_1_acc_bf16_acc_size_8`, block `(32,8,1)`, grid `(1536,2,1)`: about `229.7us`, SM throughput `89%`, achieved occupancy `20%`, eligible warps/scheduler `0.16`, no spilling. This is Tensor Core compute-heavy with low scheduler eligibility and low occupancy, not a DRAM-saturated scalar kernel.
  - Same matmul shape with grid `(384,2,1)`: about `228.2us`, similar SM throughput and occupancy. The near-equal duration despite 4x smaller grid suggests fixed scheduling/tiling overhead dominates this small shape.
  - `matmul_entry_lhs_bf16_lhs_size_8_rhs_bf16_rhs_size_8_acc_bf16_acc_size_8`, block `(32,4,1)`, grid `(384,2,1)`: about `106.2us`, memory throughput `72%`, SM throughput `56%`, achieved occupancy `21%`, eligible warps/scheduler `0.20`, no spilling.
  - `layer_norm_forward_kernel_f_`, block `(1024,1,1)`, grid `(8192,1,1)`: median about `84.0us`, memory/SM throughput about `32.5%`, DRAM about `16.7%`, achieved occupancy `63.3%`, theoretical occupancy `66.7%`, eligible warps/scheduler `0.85`, no spilling. This remains latency/eligibility limited under the locally-safe 1024 deterministic boundary.
  - `channel_mixer_relu_square_forward_kernel_f__n_8`, block `(32,8,1)`, grid `(12288,1,1)`: about `52.7us`, DRAM throughput `84%`, SM throughput `7.5%`, achieved occupancy `75.7%`, eligible warps/scheduler `0.08`, no spilling. This is bandwidth/latency dominated and is not the largest single captured cost.
- Measurement continuation before running: run a focused loss-only ncu because the mixed forward capture did not reach `lm_head/l2wrap`.
- Focused loss-only command planned: `ncu --target-processes all --kernel-name regex:'.*(lm_head|l2wrap).*' --launch-count 32 --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats --csv --log-file target/rwkv-test/ncu-forward-current-loss-after-keys.csv target/release/rwkv-test compare-rwkv-nn --color never --repeat 1 --warmup 1`.
- Focused loss-only result: `target/rwkv-test/ncu-forward-current-loss-after-keys.csv` was produced. The process exited with code `1` only because profiler-instrumented compare timings intentionally mismatch the repeat profile; activation still passed `54/54`.
- Focused loss medians:
  - `lm_head_l2wrap_ce_forward_row_kernel_f__i_i32`, block `(256,1,1)`, grid `(8192,1,1)`: `1258.2us`, memory/DRAM throughput `87.2%`, SM throughput `23.2%`, achieved occupancy `98.9%`, eligible warps/scheduler `0.41`, issued warp/scheduler `0.21`, no spilling.
  - `lm_head_l2wrap_ce_forward_row_kernel_f__i_i32`, block `(512,1,1)`, grid `(8192,1,1)`: `714.5us`, memory/DRAM throughput `87.2%`, SM throughput `40.8%`, achieved occupancy `98.5%`, eligible warps/scheduler `1.20`, issued warp/scheduler `0.36`, no spilling.
  - `lm_head_l2wrap_ce_forward_finalize_kernel_f_`, block `(256,1,1)`, grid `(1,1,1)`: `13.4us`; not a material bottleneck.
- Evidence interpretation: the row kernel is bandwidth-bound at high occupancy, but the current candidate space still exposes a clearly worse `256` block on this local CC 12.0 run. The next implementation branch should inspect whether `LocalTuner` is paying for the `256` candidate during measured warmup/steady state, whether the cache key permits a stale winner, or whether the candidate set should be hardware/shape narrowed for `rows=8192`, `vocab=65536`, BF16, deterministic forward.
