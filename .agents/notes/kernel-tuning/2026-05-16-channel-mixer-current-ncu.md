# Channel Mixer Current ncu

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-channel-mixer-current-ncu-20260516` in the existing dirty workspace.
- Dirty-tree constraint: this branch inherits the live LayerNorm runtime-dispatch code and broad unrelated dirty workspace changes. This attempt is read-only profiling for channel mixer; no kernel code will be edited here.
- Prior-note search command: `rg -n "channel_mixer|channel mixer|matmul|TMA|LocalTuner|relu_square|Burn reference|forced Cube|fusion|lhs_size|rhs_size|ncu|speedup" .agents/notes/kernel-tuning /root/.codex/memories/MEMORY.md -S`.
- Matched prior evidence:
  - `2026-05-16-channel-mixer-ncu.md` found local channel mixer custom mix and ReLU-square kernels are memory-bound but not the main module bottleneck; the two TMA matmuls were about `230us` each.
  - `2026-05-16-channel-mixer-cube-matmul.md` found forced Cube matmul was slower than the current TMA path.
  - `2026-05-16-channel-mixer-tma-matmul-ncu.md` planned a deeper TMA ncu rerun, but its first run used a stale release binary and remained pending.
  - `2026-05-16-channel-mixer-remote-repeat.md` showed remote GB10 total speedup stays above `1.0x`, but marginal channel mixer row failures repeat.
- Machine/GPU: local RTX 5090 compute capability `12.0`.
- Kernel/stage: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, channel mixer forward kernels, especially TMA matmuls and surrounding custom elementwise kernels.
- Changed boundary: this is a current-binary ncu capture after the LayerNorm runtime-dispatch branch, not the stale split-tail LayerNorm binary from the previous ncu attempt.
- Command to run: `ncu --target-processes all --kernel-name regex:'.*(channel_mixer|matmul_entry_lhs_bf16_lhs_size_1_rhs_bf16_rhs_size_1_acc_bf16_acc_size_8).*' --launch-count 48 --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats --section LaunchStats --section ComputeWorkloadAnalysis --csv --log-file target/rwkv-test/ncu-channel-mixer-current.csv target/release/rwkv-test compare-rwkv-nn --color never --repeat 1 --warmup 1`.
- Expected keep/revert boundary: keep profiler evidence only. Compare timing from ncu is invalid because profiler instrumentation changes timing. The next implementation branch must be based on specific ncu metrics rather than repeating forced Cube, Burn reference, or tuner bypass attempts.
