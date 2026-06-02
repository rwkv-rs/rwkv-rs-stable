# LayerNorm D768 BF16 Block 512 Experiment

- Date: 2026-05-16
- Branch/worktree: `kernel-tuning-hw-shape-dispatch`
- Machine/GPU: local CUDA machine, ncu reports CC 12.0
- Kernel/stage: `layer_norm_forward_kernel`, used by `layer_norm0`, pre-time-mix norms, pre-channel-mix norms, and the traced `lm_head/embedded_context` norm.
- Shape/dtype target: CUDA BF16 `rwkv_lm` trace family, `B=16,T=512,D=768`, rows `8192`.
- Prior evidence: ncu on the current 1024-block path showed achieved occupancy around `63%`, theoretical occupancy `66.67%`, and a register/warps occupancy limit. Local history says 256-block LayerNorm caused activation drift, so this experiment does not retry 256.
- Change: for `d_model == 768`, lower the BF16 deterministic minimum block size from `1024` to `512`, leaving other `d_model` values unchanged.
- Candidate parameters: existing candidates `64,128,256,512,768,1024`; deterministic filter should now allow `512,768,1024` for D768 BF16.
- Correctness result: failed local trace compare. Activation comparison was `52/54 PASS`; `cells/cell_0000/time_mixer/value_from_first_cell.safetensors` failed with `max_abs=1.035156e-1`, and `lm_head/embedded_context.safetensors` failed with `max_abs=1.406250e-1`.
- Timing/profiler result: total timing was `actual_total_ms=47.047`, baseline `35.957`, speedup `0.76x`, so there was no usable timing win.
- Revert validation: after restoring the 1024 deterministic boundary, local `compare-rwkv-nn` returned to `54/54 PASS`; timing remained failed at `actual_total_ms=48.747`, baseline `35.957`, speedup `0.74x`.
- Decision: reverted the code change. Treat D768 BF16 deterministic minimum `512` as a local negative result, alongside the older 256 drift; keep the 1024 deterministic boundary on this machine unless a different accuracy guard or hardware-specific dispatch path is added.
