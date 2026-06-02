# Remote Channel Mixer Edge ncu

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-remote-channel-mixer-edge-ncu-20260516` in the existing dirty workspace.
- Dirty-tree constraint: this branch starts after the ordered-256 LayerNorm candidate was reverted. It inherits broad unrelated workspace changes and uses the remote `10.100.1.253` code copy for measurement.
- Prior-note search command:
  - `rg -n "channel_mixer|channel mixer|matmul|TMA|LocalTuner|relu_square|Burn reference|forced Cube|fusion|lhs_size|rhs_size|ncu|speedup|GB10|remote" .agents/notes/kernel-tuning /root/.codex/memories/MEMORY.md -S`
- Matched prior evidence:
  - `2026-05-16-channel-mixer-cube-matmul.md`: forcing `MatmulStrategy::Cube` was much slower; do not repeat forced Cube.
  - `2026-05-16-channel-mixer-matmul-fusion-analysis.md`: current Burn/Cubek versions do not expose a small public TMA epilogue hook for fusing `relu_square`; do not repeat Burn-reference/fusion without a new implementation boundary.
  - `2026-05-16-channel-mixer-current-ncu.md`: local ncu showed channel-mixer matmuls dominate the module, but local GPU is no longer the decision source while the user is using the machine.
  - `2026-05-16-channel-mixer-remote-repeat.md`: remote repeated marginal channel-mixer failures, so this is not fully resolved by LayerNorm dispatch.
  - `2026-05-16-layernorm-d768-ordered256.md`: after reverting ordered-256, remote compare still had one failure, `cell_0000/channel_mixer`, while total speedup was `1.49x`.
- Machine/GPU: remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`.
- Kernel/stage: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, channel mixer forward, especially `cell_0000/channel_mixer`.
- Hypothesis: the remaining remote failure is a marginal channel-mixer edge row. Before changing code, collect GB10 ncu evidence for the selected TMA matmuls and surrounding custom elementwise kernels, then choose a candidate that can plausibly increase the remote speedup beyond the current accepted range.
- Candidate parameters: no code candidate in this branch. Profiler fields: kernel names, selected matmul family, duration, achieved occupancy, scheduler efficiency, warp state, memory throughput, launch configuration, registers/shared-memory limits.
- Command to run: `ssh ... 'cd ~/Projects/Packages/rwkv-rs-stable && ncu --target-processes all --kernel-name regex:".*(channel_mixer|matmul_entry_lhs_bf16_lhs_size_1_rhs_bf16_rhs_size_1_acc_bf16_acc_size_8).*" --launch-count 48 --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats --section LaunchStats --section ComputeWorkloadAnalysis --csv --log-file target/rwkv-test/ncu-remote-channel-mixer-edge.csv target/release/rwkv-test compare-rwkv-nn --color never --baseline ~/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 1 --warmup 1'`.
- Expected keep/revert boundary: keep profiler evidence only. Compare timing under ncu is invalid. Do not implement forced Cube, Burn reference/fusion, or LocalTuner bypass from this branch.

## 2026-05-16 Remote ncu Path Fix

- Invalid command result: remote `ncu ...` failed with `zsh:1: command not found: ncu`. No profiler data was collected.
- Next step: locate the remote Nsight Compute binary and rerun with an absolute path or a command-scoped PATH.

## 2026-05-16 Profiler CSV Analysis Continuation

- Branch/worktree: `kernel-tuning-remote-channel-mixer-ncu-analysis-20260516` in the existing dirty workspace.
- Dirty-tree constraint: this branch inherits broad unrelated workspace changes plus the process-rule skill edits. This continuation is read-only profiler evidence analysis and must not edit kernel code.
- Prior-note search command:
  - `rg -n "channel_mixer|channel mixer|matmul|TMA|LocalTuner|relu_square|Burn reference|forced Cube|fusion|lhs_size|rhs_size|ncu|speedup|GB10|remote" .agents/notes/kernel-tuning /root/.codex/memories/MEMORY.md -S`
- Matched prior evidence:
  - `2026-05-16-channel-mixer-cube-matmul.md`: forced Cube matmul was slower; do not repeat that implementation.
  - `2026-05-16-channel-mixer-matmul-fusion-analysis.md`: Burn/Cubek fusion does not expose a small public TMA epilogue hook for `relu_square`; do not repeat Burn-reference/fusion without a new implementation boundary.
  - `2026-05-16-channel-mixer-remote-repeat.md`: remote GB10 repeated marginal channel-mixer timing failures after LayerNorm dispatch.
  - This note already records that `/usr/local/cuda/bin/ncu` was needed on the remote host, and that compare timing under ncu/repeat=1 is invalid for speedup acceptance.
- Machine/GPU: remote `caizus@10.100.1.253`, expected `NVIDIA GB10`, compute capability `12.1`; recheck command output before using the data.
- Kernel/stage: `cells/cell_0000/channel_mixer` edge failure and related `matmul_entry_lhs_bf16_lhs_size_1_rhs_bf16_rhs_size_1_acc_bf16_acc_size_8` / custom elementwise kernels.
- Shape/dtype: BF16 `rwkv_lm` trace, rows `B*T=8192`, `d_model=768`, hidden `4*d_model=3072`.
- Command to run:
  - Pull/inspect `target/rwkv-test/ncu-remote-channel-mixer-edge.csv` from `~/Projects/Packages/rwkv-rs-stable` on `10.100.1.253`.
  - Parse kernel rows for duration, achieved occupancy, warp execution/scheduler signals, memory throughput, launch configuration, registers, and shared memory.
- Expected keep/revert boundary: keep profiler evidence only. If the CSV is missing, stale, or from the wrong binary/branch, mark it invalid and rerun ncu from a rebuilt remote binary before choosing an implementation branch.

### Analysis Result

- Remote hardware check: `nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader` returned `NVIDIA GB10, 12.1`.
- Remote provenance caveat: `~/Projects/Packages/rwkv-rs-stable` is not a git checkout on the remote host, so branch provenance must come from the local branch/note plus remote path and binary timestamp. `target/release/rwkv-test` existed with mtime `May 15 17:45`.
- Pulled CSV: `target/rwkv-test/ncu-remote-channel-mixer-edge.csv`, size `454` bytes, remote mtime `2026-05-15 17:48`.
- Invalid profiler result: the CSV only contains `ERR_NVGPUCTRPERM`; Nsight Compute did not collect kernel metrics because the remote user lacks permission to access NVIDIA GPU performance counters.
- Next command before declaring an ncu blocker: try a minimal `/usr/local/cuda/bin/ncu --section LaunchStats --launch-count 1` run to see whether launch configuration can be collected without performance counters. If this also fails with `ERR_NVGPUCTRPERM`, record the remote ncu permission blocker and use non-counter evidence only until profiling permissions are changed.
- Minimal ncu command: `/usr/local/cuda/bin/ncu --target-processes all --kernel-name regex:".*(channel_mixer|matmul_entry_lhs_bf16_lhs_size_1_rhs_bf16_rhs_size_1_acc_bf16_acc_size_8).*" --launch-count 1 --section LaunchStats --csv --log-file target/rwkv-test/ncu-remote-channel-mixer-launchstats.csv target/release/rwkv-test compare-rwkv-nn --color never --baseline ~/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 1 --warmup 1`.
- Minimal ncu result: the compare process ran and activation stayed valid (`54/54 PASS`), but the CSV again contains only `ERR_NVGPUCTRPERM`. Even `LaunchStats` is blocked on the remote host.
- Decision: remote ncu is currently blocked by GPU performance-counter permissions. Do not claim occupancy, warp efficiency, memory throughput, register-pressure, or shared-memory conclusions from this run. Continue with standard remote compare and source/static evidence until profiling permissions are changed or an allowed profiler path is available.
- Next non-counter command: run the standard remote compare with `repeat=3,warmup=1` from `~/Projects/Packages/rwkv-rs-stable` against the regenerated baseline at `~/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000` to identify the current stable failing rows before opening an implementation branch.
- Standard remote compare command: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline ~/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1`.
- Standard remote compare result: activation stayed valid (`activation_summary compared=54 passed=54 failed=0`), timing was `75/76 PASS`, `actual_total_ms=120.800`, `baseline_total_ms=175.887`, `speedup=1.46x`.
- Remaining failing row: only `timing/cells/cell_0000/channel_mixer.time.json`, `actual_ms=1.845`, `baseline_ms=1.668`, `speedup=0.90x`.
- Interpretation: current non-counter evidence points to a first-cell/channel-mixer timing boundary rather than a broad channel-mixer regression: aggregate `cells/*/channel_mixer` is `19.992ms` vs `25.759ms`, `1.29x`, and all later channel-mixer rows pass. Before opening an implementation branch, inspect whether first measured channel-mixer timing includes one-time dispatch, layout, cache, or synchronization cost.
- Sample inspection command: read remote actual and baseline `timing/cells/cell_*/channel_mixer.time.json` sample arrays with Python.
- Sample inspection result: actual cell_0000 samples were `[1846263, 1571780, 2117450]ns`; baseline cell_0000 samples were `[1553827, 1494197, 1956200]ns`. Later actual cells mostly have a fast steady sample near `1.56ms` plus occasional outliers. This reinforces that the single failing row is sensitive to timing variance and first-cell boundary effects.
- Next evidence command: inspect remote autotune cache/logs for channel-mixer matmul and elementwise selections to confirm the first cell is not using a different candidate before opening an implementation branch.
- Autotune log result: channel-mixer mix has the hardware/shape key `runtime=cuda`, BF16, anchored `num_elements=8388608`, `rows=8192`, `innermost_dim=768`, `max_line_size=8`, `is_in_place=false`, `deterministic=true`, and selected `line_size_2` (`~84us` in tuner). ReLU-square has BF16, anchored `num_elements=33554432`, `rows=8192`, `innermost_dim=3072`, `max_line_size=8`, `is_in_place=true`, `deterministic=true`, and selected `line_size_4` (`~428us` in tuner).
- Matmul autotune log result: the relevant rounded channel-mixer matmul shapes are `m=8192,n=4096,k=1024` and `m=8192,n=1024,k=4096`; both selected `matmul_specialized_tma_mma` (`~630us` and `~550us` in tuner). This matches the prior local TMA observation and does not support repeating forced Cube, Burn reference, or tuner bypass.
- Decision for this branch: keep the ncu permission blocker plus non-counter evidence. Do not open a channel-mixer implementation branch from this data alone; the remaining row failure is too narrow and the candidate logs already pick the expected TMA/vector choices. The next implementation attempt should target a larger remote time surface, especially `cells/*/time_mixer`, with a fresh branch and note.
