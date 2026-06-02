# LayerNorm D768 BF16 Block 768

- Date: 2026-05-16
- Branch/worktree: `kernel-tuning-layernorm-block768-20260516` in the existing dirty workspace.
- Machine/GPU: local CUDA machine, prior ncu reports CC 12.0.
- Kernel/stage: `layer_norm_forward_kernel`, especially per-cell pre-time-mix and pre-channel-mix norms.
- Shape/dtype target: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`.
- Candidate parameters: change the deterministic BF16 minimum block size from `d_model.next_power_of_two()` to warp-aligned `d_model`, so `D=768` can choose `block_size=768` while still rejecting the known-drifting `256` and `512` candidates.
- Prior negative evidence: local `256` and `512` block sizes drifted activation values. This attempt is distinct because all `D=768` columns fit in one warp-aligned block, avoiding the split-row accumulation shape of the smaller blocks.
- Correctness result: failed local trace compare. Activation comparison was `52/54 PASS`; `cells/cell_0000/time_mixer/value_from_first_cell.safetensors` failed with `max_abs=6.250000e-2`, and `lm_head/embedded_context.safetensors` failed with `max_abs=1.015625e-1`.
- Timing/profiler result: faster but unusable. The compare reported `actual_total_ms=45.296`, baseline `35.957`, total `0.79x`; per-cell pre-layer norms improved relative to the 1024 run, but accuracy failed.
- Decision: negative result. Reverted the code change on this branch and keep the branch/note as evidence that BF16 `D=768` block `768` is not a valid local deterministic candidate.
