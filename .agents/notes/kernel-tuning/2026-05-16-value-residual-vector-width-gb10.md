# 2026-05-16 value_residual_gate vector width on GB10

- Branch/worktree: `kernel-tuning-value-residual-vector-width-gb10-20260516` in `/mnt/g/Projects/Packages/rwkv-rs-stable`.
- Dirty-tree constraint: this branch is created from the current kernel-tuning checkout, which already carries the kept `gated_readout` GroupNorm-combine changes plus broad unrelated workspace edits. This attempt is scoped to `crates/rwkv-nn/src/kernels/train/time_mixer/value_residual_gate/**`.
- Prior-note search command: `rg -n "lm_head|projection|timing contract|gated_readout|groupnorm|ordered-256|residual" .agents/notes/kernel-tuning /root/.codex/memories/MEMORY.md`.
- Matched prior evidence:
  - `2026-05-16-remote-time-mixer-post-gates-nsys.md` records that the remote `value-residual-gate-forward` autotune key had `max_line_size=1` because `max_line_size_many(&[value, value_from_first_cell, gate_base, gate_input], shape.num_dims() - 1)` included 1D `gate_base` with 3D tensors.
  - The same note says the next implementation branch should fix only `value_residual_gate` forward vector-width eligibility and let existing `LocalTuner` choose among candidates on GB10.
  - Closed duplicate guards remain: residual-add/Burn-add A/B, channel-mixer forced Cube/Burn-reference/fusion/line-size, LocalTuner bypass, WKV7 row-tile restriction, lm-head atomic/target-logit/online-softmax/prune-256, GatedReadout warp32/row-pack, key-prepare warps, and LayerNorm ordered-256.
- Machine/GPU: performance decision source is remote `10.100.1.253` / GB10. Local GPU must not be used for timing.
- Backend/runtime: CUDA through Burn/CubeCL.
- Shape/dtype: `rwkv_lm/bf16/case_000000`, `B=16`, `T=512`, `rows=8192`, `d_model=768`.
- Kernel/stage: `time_mixer/value_residual_gate` forward.
- Hypothesis: the current forward key underestimates vector width by mixing 1D `gate_base` into a 3D last-axis calculation. Computing `gate_base` eligibility on its own axis should allow existing `line_size_{2,4,8}` candidates where alignment supports them, without changing math or deterministic boundaries.
- Expected keep/revert boundary: keep only if activation still passes and remote standard compare improves or at least does not regress the relevant time-mixer timing. Revert if activation drifts, selected candidate is invalid/stale, or timing regresses materially.
- Next command: inspect `value_residual_gate` forward IO/key/kernel source, then apply the smallest eligibility-key change.
- Source inspection result: this checkout already contains `max_line_size_value_residual(...)`, separating `[value, value_from_first_cell, gate_input]` embedded-axis vector width from `gate_base` axis `0`.
- Duplicate evidence: `.agents/notes/kernel-tuning/2026-05-16-value-residual-gate-vector-axis-gb10.md` already records this exact implementation boundary, including replacing the old mixed `max_line_size_many(...)` expression and clearing the remote autotune cache.
- Decision: close this branch as duplicate before any remote run or kernel edit. Do not rerun this experiment.
