# WKV7 State Time Split Design GB10

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-wkv7-state-time-split-design-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits plus retained/reverted kernel tuning changes. This branch starts as source/design inspection only and must not edit WKV7 code unless a concrete non-duplicate candidate is documented first.
- User constraint: tune/debug first on remote `10.100.1.253`; do not use local GPU timing.
- Prior-note and memory search command:
  - `rtk rg -n "WKV7|wkv7|state-scan|segment|segment_len|row_tile|direct value|output_factored|shared-value|shared state|time-split|scan|recompute|register" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md /root/.codex/memories/MEMORY.md`
- Matched prior evidence:
  - `kernel-tuning` skill records three GB10 WKV7 negatives: f32 `P/Q` segment-transform/scan/recompute was ~`13.2x` slower due global f32 intermediates and `255` regs/thread; `row_tile_64_output_factored` was slower because extra block reductions/sync outweighed saved arithmetic; direct-loading `value[row]` instead of shared staging reduced resources but worsened output time.
  - `2026-05-16-remote-post-target-logit-attribution-gb10.md` says WKV7 output remains the largest project-owned custom kernel at roughly `22ms / 36` launches, but row-tile-only and shared-lanes attempts are rejected.
  - `2026-05-16-goal-gap-audit-gb10.md` routes the next non-duplicate work here because other small kernels are duplicate-closed and broad lm-head projection/loss fusion is a separate operator/API design.
- Machine/GPU: remote target `10.100.1.253`, GB10, CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768,head_size=64`, rows `8192`, heads `12`.
- Hypothesis boundary: inspect whether WKV7 can be split across time/state in a materially different way that avoids global f32 segment tensors, avoids the rejected row-tile-only changes, and reduces the output kernel's serial recurrence or register pressure. This branch does not assume an implementation exists.
- Expected keep/revert boundary: keep only a design candidate if it names the exact state representation, memory traffic, register/shared-memory pressure, and correctness boundary. If the only ideas repeat segment-recompute or row-tile variants, close as blocked and update the skill.
- Next command: inspect the current WKV7 forward/kernel source plus prior WKV7 notes before proposing any edit.

## Duplicate Boundary Result

- Source inspection confirmed the same current structure already recorded in `2026-05-16-wkv7-remote-launch-design.md`:
  - Current pretrain output path keeps one `Array<f32>(head_size)` state per active row in registers.
  - One cube covers `(batch, head, row_tile)` and loops serially over all `context_len=512` time steps.
  - `row_tile=64` means only `12 * 16 = 192` cubes per launch, with high register pressure from the per-row state.
  - Smaller row tiles increase cube count but repeat shared input loads; that exact direction already failed.
- Prior-note collision:
  - `2026-05-16-wkv7-remote-launch-design.md` already concluded that a chunk/time split requires state-handoff or scan-like composition and is too broad for a small launch-geometry tweak.
  - `2026-05-16-wkv7-segment-recompute-gb10.md` then implemented the mathematically safe state transform path with f32 global `P/Q` segment tensors and proved it far slower on GB10.
  - `2026-05-16-wkv7-direct-value-gb10.md` and the skill already close the narrower shared-memory/value-load variation.
- Decision: close this branch as duplicate design evidence. Do not edit WKV7 code here. A future WKV7 attempt must first name a design that avoids both the prior global f32 segment tensors and the row-tile/shared-value/output-factor families; otherwise it is duplicate.
- Keep/revert state: no source changes in this branch.
