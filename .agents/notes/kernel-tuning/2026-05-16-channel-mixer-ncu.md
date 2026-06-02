# Channel Mixer ncu Probe

- Date: 2026-05-16
- Branch/worktree: `kernel-tuning-channel-mixer-ncu-20260516` in the existing dirty workspace.
- Machine/GPU: local CUDA machine, prior ncu reports CC 12.0.
- Command: `rtk ncu --target-processes all --kernel-name regex:'.*(channel_mixer|relu_square|matmul|gemm|Gemm|sgemm|hgemm|cutlass|mma).*' --launch-count 120 --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats --csv --log-file target/rwkv-test/ncu-channel-mixer-kernels.csv target/release/rwkv-test compare-rwkv-nn --color never --repeat 1 --warmup 1`
- Output file: `target/rwkv-test/ncu-channel-mixer-kernels.csv`.
- Scope: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, channel mixer forward path including custom mix/ReLU-square kernels and nearby matmul kernels.
- Reason: clean compare still shows `cells/*/channel_mixer` around `0.51x-0.58x`; previous ncu only established the custom mix kernel is memory-heavy and did not explain whether the two matmuls or autotune/cache behavior dominate the module timing.
- Profiler caveat: compare timing is invalid under ncu and `repeat=1`; use the CSV only for kernel-level metrics.
- Profiler result:
  - `channel_mixer_mix_forward_kernel_f__n_8`: median duration about `11.65us`, block `(32,8,1)`, grid `(3072,1,1)`, DRAM throughput about `76%`, SM throughput about `14%`, achieved occupancy about `77.5%`, average active threads per warp `32`. This is bandwidth-heavy and not the module bottleneck.
  - `channel_mixer_relu_square_forward_kernel_f__n_4`: median duration about `55.87us`, block `(32,8,1)`, grid `(24576,1,1)`, DRAM throughput about `83.6%`, SM throughput about `14%`, achieved occupancy about `71.7%`, average active threads per warp `32`. This is also memory-bound.
  - First channel mixer matmul, likely `key_input @ key_weight`: `matmul_entry_lhs_bf16_lhs_size_1_rhs_bf16_rhs_size_1_acc_bf16_acc_size_8`, block `(32,8,1)`, grid `(1536,2,1)`, median duration about `230.40us`, memory throughput about `67%`, DRAM about `17%`, SM throughput about `90%`, achieved occupancy about `20.1%`, active threads per warp about `25.2`.
  - Second channel mixer matmul, likely `activated_key @ value_weight`: same `lhs_size_1/rhs_size_1` kernel family, block `(32,8,1)`, grid `(384,2,1)`, median duration about `227.55us`, memory throughput about `63%`, DRAM about `19%`, SM throughput about `85%`, achieved occupancy about `19.2%`, active threads per warp about `24.9`.
- Decision: do not spend the next channel mixer attempt on `mix` line-size tuning. The actionable question is why these two channel mixer matmuls dispatch to the scalar-load `lhs_size_1/rhs_size_1` CubeCL matmul family while other model matmuls in the same profile use `lhs_size_8/rhs_size_8`.
