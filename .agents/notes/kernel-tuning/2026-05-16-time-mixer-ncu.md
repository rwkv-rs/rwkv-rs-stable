# Time Mixer ncu Probe

- Date: 2026-05-16
- Branch/worktree: `kernel-tuning-time-mixer-ncu-20260516` in the existing dirty workspace.
- Machine/GPU: local CUDA machine, ncu reports CC 12.0.
- Command: `rtk ncu --target-processes all --kernel-name regex:'.*(mix6|key_prepare|wkv7|learning_rate|value_residual).*' --launch-count 80 --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis --csv --log-file target/rwkv-test/ncu-time-mixer-kernels.csv target/release/rwkv-test compare-rwkv-nn --color never --repeat 1 --warmup 1`
- Output file: `target/rwkv-test/ncu-time-mixer-kernels.csv`.
- Profiler caveat: compare timing is invalid under ncu; the profiled run reported huge `time_mixer` times because ncu serialized and instrumented launches.
- Scope: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, `num_heads=12`, `head_size=64`.
- `mix6_forward_kernel_f__n_8`: block `(32,8,1)`, grid `(3072,1,1)`, duration around `43-48us` for normal captured launches, DRAM/memory throughput around `80%`, achieved occupancy around `78%`, no spilling. This looks memory-bound; line-size retuning is unlikely to produce a large win by itself.
- `key_prepare_forward_64_kernel_f_`: block `(128,1,1)`, grid `(24576,1,1)`, duration around `24-31us`, memory throughput around `83-88%`, achieved occupancy around `82-87%`, no spilling. The specialized head-64 path is already bandwidth heavy.
- `wkv7_pretrain_forward_output_kernel_f_bf16`: block `(64,1,1)`, grid `(12,16,1)`, duration around `0.50ms`, memory throughput around `18%`, achieved occupancy around `4.7%`, theoretical occupancy `33.3%`. ncu flags the grid as too small to fill the GPU.
- Decision: do not spend the next implementation attempt on mix6/key_prepare line-size microtuning. The next meaningful local time-mixer attempt should investigate WKV7 pretrain parallelism or launch geometry: the current recurrence mapping exposes only `batch * heads = 192` blocks and under-fills this GPU.
