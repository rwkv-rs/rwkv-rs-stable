# Kernel Tuning Skill GB10 Results

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-skill-gb10-results-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits and accumulated kernel tuning changes. This attempt owns only `.agents/skills/kernel-tuning/SKILL.md` and this note.
- User constraint: keep project skills updated, use branch-based kernel tuning, and record every change in notes.
- Prior-note search commands:
  - `rg -n "target-logit|target logit|key-prepare|warps_per_cube|warps-per-cube|row-pack|gated_readout|weight_decay_transform" .agents/skills/kernel-tuning/SKILL.md .agents/notes/kernel-tuning/2026-05-16-*.md`
  - `rg -n "local_target|target_logit|row_start \\+ target|HEAD64_WARPS_PER_CUBE|gated_readout_combine_row_pack|ROW_PACK|weight_decay_transform|row_tile_64_output_factored" crates/rwkv-nn/src/kernels/train .agents/skills/kernel-tuning/SKILL.md`
- Matched prior evidence:
  - `2026-05-16-lm-head-target-logit-gb10.md`: direct target-logit loading in `lm_head_l2wrap_ce` was kept on GB10. Activation passed, total speedup stayed above `1.0`, and nsys row-kernel time improved from `13.879ms / 3` to `12.851ms / 3` while registers/thread dropped from `53` to `40`.
  - `2026-05-16-key-prepare-warps-gb10.md`: key-prepare `warps_per_cube` autotune selected `2` on GB10 but doubled block count and slightly regressed targeted nsys time versus fixed `4`; the source was reverted to `HEAD64_WARPS_PER_CUBE = 4`.
  - `2026-05-16-gated-readout-rowpack-gb10.md`: GatedReadout row-pack improved the combine kernel in nsys but hurt end-to-end standard compare versus the retained 64-thread/two-warp combine; it was reverted.
  - `2026-05-16-weight-decay-transform-gb10.md`: custom weight-decay transform was kept; remote standard compare repeated cleanly (`54/54`, `76/76`, `2.18x`) and nsys showed the prior softplus scalar/unary chain was replaced by the project kernel.
  - Existing skill already records LayerNorm ordered-256, WKV7 segment recompute, WKV7 output factorization, projection timing, projection/loss design, and ncu permission blocker.
- Machine/GPU scope: remote `10.100.1.253`, GB10, CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, regenerated remote baseline.
- Kernel/stage: skill guardrails only; no kernel source or GPU command in this attempt.
- Hypothesis: these kept/rejected GB10 outcomes must be promoted from notes into the skill so later tuning branches do not repeat completed candidates.
- Expected keep/revert boundary: keep concise skill bullets that name the exact boundary and decision. Do not add broad claims beyond the measured GB10 evidence.
- Next edit: add GB10 keep/negative bullets for lm-head target-logit, key-prepare warps, GatedReadout row-pack, and weight-decay transform to `.agents/skills/kernel-tuning/SKILL.md`.

## Skill Edit

- Edited `.agents/skills/kernel-tuning/SKILL.md`.
- Added concise GB10 result guards:
  - keep `lm_head_l2wrap_ce` direct target-logit loading on the measured GB10 boundary;
  - keep fixed `HEAD64_WARPS_PER_CUBE = 4` and do not repeat key-prepare warps-per-cube autotune unchanged;
  - keep the retained 64-thread/two-warp GatedReadout combine and do not repeat row-pack unless total compare improves;
  - keep the custom `weight_decay_transform` kernel and `TrainBackend` wiring unless a new branch shows a regression.
- Next command: run local format check for the skill/note-only change, then sync the updated skill and note to `10.100.1.253`.

## Local Check

- Command: `git diff --check -- .agents/skills/kernel-tuning/SKILL.md .agents/notes/kernel-tuning/2026-05-16-skill-gb10-results.md`.
- Result: passed.
- Next command: scoped rsync of the updated skill and this note to `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable/`.

## Remote Sync

- Command: scoped `rsync -azR` of `.agents/skills/kernel-tuning/SKILL.md` and this note to `10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable/`.
- Result: sync completed.
- Keep/revert state: keep the skill updates as duplicate-experiment guards for the completed GB10 attempts.
- Next action candidate: refresh remote current-source attribution before opening another implementation branch, because the remaining obvious small candidates are now either kept, rejected, or blocked by `ncu` permissions.
