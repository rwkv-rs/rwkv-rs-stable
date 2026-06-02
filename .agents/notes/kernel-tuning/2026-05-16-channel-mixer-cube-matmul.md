# Channel Mixer Cube Matmul Probe

- Date: 2026-05-16
- Branch/worktree: `kernel-tuning-channel-mixer-cube-matmul-20260516` in the existing dirty workspace.
- Prior-note search terms: `channel_mixer`, `matmul`, `lhs_size_1`, `rhs_size_1`, `vector`, `TMA`, `residual_add`.
- Matched prior evidence: `2026-05-16-channel-mixer-ncu.md` shows the custom mix and ReLU-square kernels are not the module bottleneck; both channel mixer matmuls dispatch to `matmul_entry_lhs_bf16_lhs_size_1_rhs_bf16_rhs_size_1_acc_bf16_acc_size_8`. `2026-05-16-skill-guardrails.md` records that replacing the custom channel mixer path with the Burn reference path was slower, so this attempt keeps the custom path and only changes matmul strategy.
- Machine/GPU: local CUDA machine, prior ncu reports CC 12.0.
- Scope: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, channel mixer forward matmuls.
- Hypothesis: the `lhs_size_1/rhs_size_1` ncu family comes from the Burn/Cubek autotuner selecting TMA strategies; Cubek TMA launch restricts lhs/rhs vector sizes to `1`. For this skewed `[8192,768]x[768,3072]` and `[8192,3072]x[3072,768]` shape, forcing the non-autotune Cube default strategy may avoid the TMA scalar-input path and improve the channel mixer timing.
- Candidate parameters: matmul strategy `Cube` for the two channel mixer matmuls; no change to mix/relu-square line-size candidates.
- Correctness result: local compare passed activation, `activation_summary compared=54 passed=54 failed=0`.
- Timing/profiler result:
  - Compile check: `rtk cargo check -p rwkv-nn --features cuda` passed.
  - Compare command: `rtk cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --repeat 3 --warmup 1`.
  - Compare result: `timing_summary compared=76 passed=7 failed=69 ... actual_total_ms=50.348 baseline_total_ms=35.957 speedup=0.71x`; `cells/*/channel_mixer` worsened to `14.433ms` vs baseline `5.527ms`, `0.38x`.
  - ncu command: `rtk ncu --target-processes all --kernel-name regex:'.*matmul.*' --launch-count 40 --section SpeedOfLight --section Occupancy --section SchedulerStats --section WarpStateStats --csv --log-file target/rwkv-test/ncu-channel-mixer-cube-matmul.csv target/release/rwkv-test compare-rwkv-nn --color never --repeat 1 --warmup 1`.
  - ncu output file: `target/rwkv-test/ncu-channel-mixer-cube-matmul.csv`.
  - ncu result: forcing `MatmulStrategy::Cube` changed the large channel mixer matmuls to `matmul_entry_lhs_bf16_lhs_size_8_rhs_bf16_rhs_size_8_acc_bf16_acc_size_8`, but the slow captures were about `727us` and `371us`. The previous autotuned/TMA path in `target/rwkv-test/ncu-channel-mixer-kernels.csv` used `lhs_size_1/rhs_size_1` but the two likely channel mixer matmuls were about `230us` and `228us`.
  - Root cause learned: `lhs_size_1/rhs_size_1` is expected for Cubek TMA launches because TMA restricts input vector size to `1`; for these channel mixer shapes, the TMA scalar-input path is still faster than the non-autotune Cube default path.
- Decision: negative result. Reverted the code change on this branch and keep the branch/note as evidence. Do not force the channel mixer matmuls to `MatmulStrategy::Cube` for the local CUDA BF16 `B=16,T=512,D=768` fixture.
