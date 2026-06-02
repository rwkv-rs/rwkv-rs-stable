# Current Top Surface Guards GB10

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-current-top-surface-guards-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits plus prior retained kernel-tuning changes. This attempt owns only `.agents/skills/kernel-tuning/SKILL.md` and this note.
- User constraint: continue remote-first on `10.100.1.253`; do not use local GPU timing while the local GPU may be used elsewhere.
- Prior search commands:
  - `rg -n "gated_readout|gated-readout|gated readout|combine|row-pack|warp32|GroupNorm|groupnorm|block=\\(64|block size|CubeDim|num_warps" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md -S`
  - `rg -n "value_residual_gate|value-residual|vector-axis|vector axis|line_size|n_2|eligibility|base tensor vector" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md crates/rwkv-nn/src/kernels/train/time_mixer/value_residual_gate -S`
  - Remote read-only sqlite/autotune queries against `target/rwkv-test/nsys-current-r9-gb10.sqlite` and `target/autotune/0.10.0/device-0-0-cuda/rwkv_nn-*.json.log`.
- Matched evidence:
  - GatedReadout GroupNorm+combine is the kept positive branch. Warp32, row-pack, and sumsq-variance variants are closed; the skill already says to retain the 64-thread/two-warp combine unless total compare improves.
  - Value-residual vector-axis fix is already implemented and validated in `2026-05-16-value-residual-gate-vector-axis-gb10.md`. It fixed `max_line_size` eligibility and changed remote dispatch from `n_1` to `n_2`; activation passed and total speedup stayed above `1.0`, but nsys was neutral/slightly negative (`6.741ms / 33` vs previous `6.620ms / 33`).
  - Current remote autotune logs show Mix6 selects `line_size_1` on GB10 while wider candidates are close but slower; channel-mixer relu-square selects `line_size_2`; value-residual selects `line_size_2`.
- Machine/GPU: remote `10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`.
- Scope: durable project-skill update only. No kernel source edit, cache clear, compile, benchmark, or profiler run in this branch.
- Hypothesis: the next correct action is to encode the current top-surface duplicate guards in the project skill so future tuning does not repeat value-residual vector-axis, Mix6 line-size forcing, or channel-mixer line-size/cache forcing without materially new evidence.
- Expected keep/revert boundary: keep the skill update if it only records already validated facts and does not claim speedups where the evidence was neutral. Revert if it overstates performance or hides that these were remote GB10-specific measurements.
- Next edit: add concise GB10 guards for value-residual vector-axis, Mix6 line-size selection, and current channel-mixer line-size/cache forcing to `.agents/skills/kernel-tuning/SKILL.md`.

## Skill Edit

- Edited `.agents/skills/kernel-tuning/SKILL.md`.
- Added `value_residual_gate` guard: vector-axis eligibility fix is kept as dispatch-correctness, not claimed speedup; do not rerun unless fresh clean profiles show a repeatable regression.
- Added Mix6 guard: GB10 selects `line_size_1`; do not force wider line sizes or bypass `LocalTuner` without new profiler evidence or a different kernel body.
- Added channel-mixer guard: current relu-square/mix line-size choices are valid remote autotune outcomes; do not repeat cache-clearing, line-size forcing, Burn-reference, or forced-Cube branches from the known narrow edge unless fresh evidence shows stale candidate or a different implementation boundary.
- Next command: sync this note and `.agents/skills/kernel-tuning/SKILL.md` to `10.100.1.253`, then verify the remote skill contains all three new guards.

## Remote Skill Sync

- Command: scoped `rsync -azR` of `.agents/skills/kernel-tuning/SKILL.md` and this note to `10.100.1.253`, then remote `grep` for the three new guard phrases.
- Remote verification:
  - `.agents/skills/kernel-tuning/SKILL.md:53` contains the `value_residual_gate` vector-axis guard.
  - `.agents/skills/kernel-tuning/SKILL.md:54` contains the Mix6 `line_size_1` guard.
  - `.agents/skills/kernel-tuning/SKILL.md:55` contains the channel-mixer line-size/cache-forcing guard.
- Decision: close this skill-guard attempt. No kernel source edits, compile, benchmark, cache clear, or profiler run were performed in this branch.
