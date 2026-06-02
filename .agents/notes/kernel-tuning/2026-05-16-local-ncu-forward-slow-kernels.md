# Local ncu Forward Slow-Kernel Probe

- Date: 2026-05-16
- Branch/worktree: `kernel-tuning-hw-shape-dispatch`
- Machine/GPU: local CUDA machine, ncu reports CC 12.0
- Command: `rtk ncu --target-processes all --kernel-name regex:'.*(lm_head|l2wrap|layer_norm|channel_mixer).*' --launch-count 20 --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis --csv --log-file target/rwkv-test/ncu-forward-slow-kernels.csv target/release/rwkv-test compare-rwkv-nn --color never --repeat 1 --warmup 1`
- Command: `rtk ncu --target-processes all --kernel-name regex:'.*(lm_head|l2wrap).*' --launch-count 10 --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis --csv --log-file target/rwkv-test/ncu-loss-kernels.csv target/release/rwkv-test compare-rwkv-nn --color never --repeat 1 --warmup 1`
- Output files: `target/rwkv-test/ncu-forward-slow-kernels.csv`, `target/rwkv-test/ncu-loss-kernels.csv`.
- Profiler caveat: ncu profiling changes `.time.json` timings heavily; use the CSV metrics for kernel diagnosis, not the compare timing summary from profiled runs.
- LayerNorm observation: the current accepted BF16 D768 path uses `layer_norm_forward_kernel_f_` with block `(1024,1,1)` and grid `(8192,1,1)`. Achieved occupancy is about `63%`, theoretical occupancy `66.67%`, with register and warp limits; memory and compute throughput are both about `32%`, suggesting latency/occupancy limits rather than pure bandwidth saturation.
- Channel mixer observation: `channel_mixer_mix_forward_kernel_f__n_8` is memory heavy. The captured launch had DRAM throughput about `74%`, compute about `14%`, achieved occupancy about `77%`, and no local/shared spilling.
- Loss observation: `lm_head_l2wrap_ce_forward_row_kernel_f__i_i32` is memory-bound and already high occupancy. The accepted block is `(512,1,1)` with grid `(8192,1,1)`, DRAM throughput about `87%`, achieved occupancy about `98%`, and no spilling. The row kernel duration was about `0.73ms` in the later profiled samples.
- Loss finalize observation: `lm_head_l2wrap_ce_forward_finalize_kernel_f_` is a single-block reduction, about `13us`, with low achieved occupancy because the grid has only one block; it is not the main loss cost.
- Decision: do not spend the next iteration on loss finalize. For LayerNorm, 512-block was tested separately and failed accuracy, so a future win needs either a different deterministic reduction strategy or hardware-specific dispatch with an accuracy guard. For channel mixer, look for memory-traffic/fusion wins rather than occupancy tuning alone.
