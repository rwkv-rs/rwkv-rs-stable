# Completion Audit Remote Only

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-completion-audit-remote-only-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits plus retained/reverted kernel tuning changes. This attempt owns only this audit note unless later entries explicitly open a profiling or implementation branch.
- User constraint: debug first on `10.100.1.253`; do not use the local GPU because the user is running other work there and local timing may be inaccurate.
- Prior-note/source search commands already run before opening this audit:
  - `rg -n "completion|goal|speedup|activation_summary|timing_summary|ncu|ERR_NVGPUCTRPERM|LocalTuner|autotune key|hardware|deterministic|ordered-256|segment" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md -S`
  - `git status --short --branch`
- Matched evidence:
  - `kernel-tuning` skill now requires branch + prewritten note, duplicate guards, remote GB10 primary timing, LocalTuner cache-hit/miss semantics, trace-backed accuracy guards, and hardware/shape/deterministic key dimensions.
  - Recent post-segment audit confirms remote `10.100.1.253` still has the kept direct target-logit row kernel and WKV7 `row_tile_{16,32,64}` only, with no segment/shared-lanes residue.
  - Remote post-target-logit compare is clean at about `2.43x`, but local latest accepted check remains below `1.0x` or stale because the local GPU is currently not usable for timing.
  - Remote `ncu` counters remain blocked by `ERR_NVGPUCTRPERM`; remote evidence can include `nsys` launch/register/shared-memory metadata but not achieved occupancy, warp execution efficiency, or memory throughput until permissions change.
- Objective restated as concrete deliverables:
  1. Every new tuning attempt uses a fresh branch/worktree and prewritten note; failed attempts are reverted but branches/notes remain.
  2. Kernel implementation selection/autotune keys cover backend/runtime, GPU hardware/capability as exposed by CubeCL, dtype, `d_model`, rows, candidate block/warp/vector parameters, alias/in-place state, and deterministic numeric boundary.
  3. LayerNorm is not a hard-coded constant; GB10 can select the fast `256` path while local deterministic policy avoids known `256` drift.
  4. Burn/CubeCL `LocalTuner::execute` key/cache/warmup/measurement behavior and accuracy-guard limitations are documented, and host-side tuner lookup is not incorrectly counted as kernel hot path.
  5. CUDA diagnosis includes warp/block configuration, register/shared-memory/launch geometry, memory/coalescing reasoning, BF16 reduction error analysis, and `ncu` metrics where permissions allow.
  6. Poor results are diagnosed for design versus implementation mistakes before revert.
  7. Final acceptance requires activation passing and speedup `>1.0` on both local and `10.100.1.253`.
- Audit checklist to inspect next:
  - current source key fields for LayerNorm, lm-head, WKV7, channel-mixer, mix6, value-residual, learning-rate, weight-decay, GatedReadout, and key-prepare;
  - current skill duplicate guards;
  - latest local/remote acceptance notes;
  - remote source/log provenance on `10.100.1.253`;
  - whether any missing requirement can be addressed without local GPU or privileged remote `ncu`.
- Expected decision boundary: do not call `update_goal` unless every deliverable is currently covered by real evidence. If any requirement is missing, record it and identify the next non-duplicate action.
- Next command: inspect current source autotune keys and latest acceptance/profiler notes with local read-only commands only.

## Audit Evidence

- Branch/worktree discipline:
  - Current `git status --short --branch` shows this audit is on `kernel-tuning-completion-audit-remote-only-20260516`.
  - The checkout is broadly dirty from existing workspace work; this audit owns only this note.
  - The `kernel-tuning` skill now explicitly records branch+note requirements, duplicate guards, and the WKV7 segment negative guard.
- Autotune key/source coverage:
  - Current train kernels with `AutotuneKey` structs include LayerNorm, lm-head forward/backward, channel-mixer forward/backward, residual-add, mix6 forward/backward, learning-rate forward/backward, WKV7 pretrain output, weight-decay transform, and value-residual forward/backward.
  - Source inspection confirms LayerNorm key contains `runtime`, `dtype`, `d_model`, `rows`, `num_elements`, hardware fields (`load_width`, `plane_size`, cube/shared/vector limits, SM/tensor-core fields), `is_in_place`, `deterministic`, and `deterministic_min_block_size`.
  - LayerNorm currently gates BF16 deterministic candidates through `deterministic_min_block_size`; GB10 `cuda/BF16/D=768/rows=8192/SM=48` is allowed to use `256`, while other local deterministic BF16 paths fall back to `d_model.next_power_of_two()` capped by hardware.
  - Other elementwise/custom keys include the same hardware/shape pattern plus `max_line_size` or candidate names such as `block_*`, `row_tile_*`, and `line_size_*`, so candidate parameters are represented in tunable names and cache checksums.
- Burn/CubeCL tuner mechanism:
  - `2026-05-16-cubecl-local-tuner-findings.md` records `LocalTuner::init` `TypeId` caching, `LocalTuner::execute` key/cache-hit behavior, cache-miss warmup/profile sampling, persistent checksum behavior, and the lack of trace-baseline accuracy guard.
  - Skill rules now forbid counting first-run retune cost as steady-state kernel timing and require trace-backed guards for BF16-sensitive candidates.
- CUDA/profiler evidence:
  - Remote post-target-logit attribution records clean GB10 activation/timing at `2.43x`.
  - Remote `nsys` gives launch/register/shared-memory metadata for current top kernels, including Cubek/TMA lm-head projection, WKV7, channel-mixer, mix6, lm-head row, GatedReadout, key-prepare, value-residual, and LayerNorm.
  - Remote `ncu` achieved-occupancy / warp-efficiency / memory-throughput counters remain blocked by `ERR_NVGPUCTRPERM`; this is an environment permission gap, not a completed evidence item.
  - Local older `ncu` notes exist for some kernels, but they do not satisfy the current remote GB10 counter requirement.
- Correctness/poor-result diagnosis:
  - LayerNorm ordered-256 root cause is recorded as a device implementation/boundary failure: CPU ordered math matches `1024`, but the device candidate still drifts on `value_from_first_cell` and `lm_head/embedded_context`.
  - WKV7 segment recompute was not blindly reverted: forced path activation passed, then `nsys` attributed the performance failure to transform+scan+recompute overhead and `255` regs/thread in the scan kernel.
  - Key-prepare warps, GatedReadout row-pack/warp32, lm-head online/atomic/prune, channel-mixer forced-Cube/Burn-reference, and residual-add A/B all have negative or kept notes with implementation/timing causes.
- Acceptance:
  - Remote `10.100.1.253` latest clean evidence: `activation_summary compared=54 passed=54 failed=0`; `timing_summary compared=76 passed=76 failed=0 ... speedup=2.43x`.
  - Local latest acceptance evidence: activation passes, but timing is `speedup=0.92x`; local final speedup `>1.0` is not achieved. The user currently asked not to use the local GPU, so this gap cannot be closed in this audit.

## Audit Decision

- Do not call `update_goal`: the objective is not complete.
- Missing or weak requirements:
  - Local speedup `>1.0` is still missing.
  - Remote `ncu` achieved occupancy / warp execution efficiency / memory throughput is still blocked by permissions.
  - Remaining high-impact remote surfaces are broad Cubek/TMA lm-head projection or WKV7 state-scan/time-split, not small duplicate candidates.
- Next concrete non-duplicate action: open a WKV7 low-memory state-scan design branch. This must be materially different from the rejected dense segment `P/Q` transform/scan/recompute path by avoiding global dense f32 transform tensors and the high-register dense segment scan.
