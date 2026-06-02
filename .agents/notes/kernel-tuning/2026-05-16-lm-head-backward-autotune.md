# LM Head L2Wrap Backward Autotune

- Date: 2026-05-16
- Branch/worktree: `kernel-tuning-hw-shape-dispatch`
- Machine/GPU: local CUDA machine
- Kernel/stage: `lm_head_l2wrap_ce` backward.
- Shape/dtype target: CUDA BF16 `rwkv_lm` trace family, `B=16,T=512,V=65536`.
- Change: replaced the fixed backward block size `512` with `LocalTuner` candidates `256`, `512`, and `1024`.
- Autotune key: runtime, dtype, rows as `num_tokens`, vocab size, load width, plane size, max units per cube, max shared memory size, max vector size, SM count, and deterministic flag.
- Candidate parameters: block size determines `num_warps = block_size / 32`; candidates over the hardware max units per cube are filtered out by the tune group.
- Correctness result: local `compare-rwkv-nn` passed activation comparison twice, `54/54 PASS`.
- Timing/profiler result: local first run `actual_total_ms=48.706`, baseline `35.957`, total `0.74x`; second warm-cache run `actual_total_ms=46.339`, baseline `35.957`, total `0.78x`. `loss/l2wrap_cross_entropy` was `0.62x` then `0.71x`; `lm_head` was `0.28x` then `0.11x`.
- Decision: keep the backward candidate expansion as correctness-safe but not accepted as a performance win. The current compare profile is `train-forward-steady-state`, so this experiment does not explain the forward `lm_head`/layer timing regressions; use ncu or a backward-specific benchmark before claiming backward speedup.
