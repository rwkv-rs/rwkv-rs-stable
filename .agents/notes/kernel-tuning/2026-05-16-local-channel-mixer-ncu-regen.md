# Local Channel Mixer NCU Regenerated Baseline

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-local-channel-mixer-ncu-regen-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace changes and retained kernel-tuning edits. This attempt is read-only except for this note and profiler artifacts under `target/rwkv-test/`.
- User constraint: the user said local GPU may be used for other work, so this branch first checked GPU state and runs only one targeted ncu if idle. Do not run broad local compares in this branch.
- Prior search command:
  - `rg -n "channel_mixer.*local.*regen|local regenerated.*channel|channel_mixer.*ncu|forced Cube|Burn reference|LocalTuner bypass|line-size|matmul fusion|cube-matmul" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md -S`
- Matched prior evidence:
  - `2026-05-16-local-regenerated-baseline.md` found local regenerated-baseline total speedup `2.44x`, but `cells/*/channel_mixer` remained slower: `9.907ms` actual vs `7.976ms` baseline, `0.81x`.
  - `2026-05-16-channel-mixer-local-regenerated-baseline.md` classifies the remaining non-duplicate local question as Cubek matmul dispatch/provenance inside channel mixer.
  - Duplicate-closed implementation attempts: Burn reference, forced Cube matmul, direct `LocalTuner` bypass, line-size retry, and local matmul fusion under current Cubek/Burn scope.
  - Prior local ncu notes show channel-mixer matmuls using the TMA `lhs_size_1/rhs_size_1` family, with forced Cube slower. This run changes the baseline/provenance boundary to local regenerated Python CUDA baseline and collects current profiler counters only.
- Machine/GPU: local `NVIDIA GeForce RTX 5090`, compute capability previously observed as `12.0`.
- Preflight result: `nvidia-smi` reported `4%` utilization, `3088/32607 MiB` memory, and no compute apps. Local `ncu` path is `/usr/local/cuda/bin/ncu`.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`.
- Baseline path: `/mnt/g/Projects/Packages/rwkv-rs-test/test_gen_local_regen_20260516/rwkv_lm/bf16/case_000000`.
- Kernel/stage: channel mixer forward module, especially `channel_mixer_mix_forward`, `channel_mixer_relu_square_forward`, and adjacent BF16 Cubek matmul entries.
- Hypothesis: under the local regenerated baseline, the remaining channel-mixer gap is likely matmul dispatch/provenance or memory/occupancy behavior, not the already rejected line-size/Burn-reference/forced-Cube paths. ncu should identify whether the current TMA matmul family is occupancy-, memory-, or scheduler-limited before any new implementation branch.
- Expected keep/revert boundary:
  - Keep profiler evidence only.
  - Treat compare timing under ncu as invalid.
  - Do not edit channel mixer kernels in this branch.
  - If ncu is blocked, stale, or the GPU becomes busy, mark the run invalid and stop.
- Next command: build `target/release/rwkv-test` from the current branch before profiling, then run the targeted ncu command below.
- NCU command:

```bash
/usr/local/cuda/bin/ncu \
  --target-processes all \
  --kernel-name regex:'.*(channel_mixer|matmul_entry_lhs_bf16_lhs_size_1_rhs_bf16_rhs_size_1_acc_bf16_acc_size_8).*' \
  --launch-count 80 \
  --section SpeedOfLight \
  --section Occupancy \
  --section MemoryWorkloadAnalysis \
  --section SchedulerStats \
  --section WarpStateStats \
  --csv \
  --log-file target/rwkv-test/ncu-local-channel-mixer-regen.csv \
  target/release/rwkv-test compare-rwkv-nn \
    --color never \
    --baseline /mnt/g/Projects/Packages/rwkv-rs-test/test_gen_local_regen_20260516/rwkv_lm/bf16/case_000000 \
    --repeat 1 \
    --warmup 1
```

## Build Result

- Command: `cargo build --release -p rwkv-test --features cuda`.
- Result: passed; release build finished in `3m 27s`.
- Next command: re-check local GPU state. If it is still idle and has no compute apps, run the targeted ncu command once.
