# WKV7 Pretrain Output Forced Row Tile 16

- Date: 2026-05-16
- Branch/worktree: `kernel-tuning-wkv7-row-tile16-forced-20260516` in the existing dirty workspace.
- Machine/GPU: local CUDA machine, prior ncu reports CC 12.0.
- Kernel/stage: `wkv7_pretrain_forward_output_kernel`, surfaced through `cells/*/time_mixer`.
- Shape/dtype target: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, `num_heads=12`, `head_size=64`, `chunk_len=16`.
- Candidate parameters: force `row_tile=16` by temporarily restricting `PRETRAIN_OUTPUT_ROW_TILE_CANDIDATES` to `[16]`.
- Reason: prior ncu showed `row_tile=64` launches only grid `(12,16,1)` with achieved occupancy around `4.7%`. The previous autotune attempt still selected `64`; this run checks whether end-to-end module timing benefits from higher block count despite repeated shared input loads.
- Correctness result: passed local trace compare, `54/54 PASS`.
- Timing/profiler result: worse. The compare reported `actual_total_ms=54.328`, baseline `35.957`, total `0.66x`; `cells/*/time_mixer` was `36.734ms` vs baseline `27.836ms`, speedup `0.76x`.
- Decision: negative result. Reverted the forced candidate restriction back to `[16, 32, 64]`. The previous tuner winner `row_tile=64` is not just a local measurement artifact; increasing row-block count repeats input loads enough to lose end-to-end despite higher theoretical block count.
