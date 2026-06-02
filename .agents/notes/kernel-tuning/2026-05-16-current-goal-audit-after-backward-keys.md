# Current Goal Audit After Backward Keys

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-current-goal-audit-after-backward-keys-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits and accumulated kernel-tuning branches. This audit owns only this note unless a later entry explicitly opens a separate implementation branch.
- User constraint: continue remote-first on `10.100.1.253`; do not use local GPU timing while the user is using the local GPU.
- Prior search commands:
  - `rtk rg -n "completion|goal|speedup|activation_summary|timing_summary|lm_head/projection|projection-loss|projection loss|fused.*projection|WKV7 state|state-scan|ncu|ERR_NVGPUCTRPERM|LocalTuner|autotune key|hardware|deterministic|10\\.100\\.1\\.253|local" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md /root/.codex/memories/MEMORY.md -S`
  - `rtk git status --short --branch`
  - `rtk sed -n '1,220p' .agents/skills/kernel-tuning/SKILL.md`
- Branch creation command:
  - `rtk git switch -c kernel-tuning-current-goal-audit-after-backward-keys-20260516`
  - Result: switched to the new branch.

## Objective Restatement

- Use branch-per-attempt kernel tuning with notes; failed attempts are reverted but branches/notes remain.
- Move kernel implementation choice toward hardware/shape keyed dispatch/autotune rather than hard-coded constants.
- Cover key dimensions where available: backend/runtime, GPU hardware fingerprint or compute capability, dtype, `d_model`, rows `B*T`, block size, warp/vector parameters through candidate names/groups, alias/in-place behavior, and deterministic numeric boundary.
- Investigate Burn/CubeCL `LocalTuner` key/cache/warmup/measurement behavior and avoid counting retune/host lookup cost as steady-state kernel time.
- Investigate CUDA kernel causes with profiler evidence where possible: warp/block config, occupancy, register/shared-memory pressure, memory behavior, BF16 reduction numerical drift, and implementation correctness before reverting.
- Final performance target: speedup greater than `1.0` on both local and `10.100.1.253`, with activation passing. Current turn must not use local GPU timing.

## Prompt To Artifact Checklist

| Requirement | Current evidence | Status |
| --- | --- | --- |
| Branch per tuning attempt | `$kernel-tuning` skill enforces this. Current recent attempts have dedicated branches/notes, including backward keys and this audit. | Covered for current workflow |
| Notes for each change/run | `.agents/notes/kernel-tuning/` contains per-attempt notes. Backward-key note records code, local/remote compile, remote compare, and decision. | Covered |
| Failed attempts preserved and reverted | Notes preserve rejected residual, channel-mixer, LayerNorm ordered/split, WKV7 row/shared/segment, key-prepare, GatedReadout, lm-head variants. | Covered |
| Hardware/shape autotune keys | LayerNorm, lm-head forward/backward, channel-mixer forward/backward, mix6 forward/backward, learning-rate/value-residual gates, WKV7, weight-decay, and residual paths now carry runtime/shape/hardware/vector/deterministic/alias fields through keys or candidate groups. Literal CUDA CC is not available from public CubeCL `HardwareProperties`; code uses the available hardware fingerprint fields. | Mostly covered |
| LayerNorm remote/local conflict | Notes record GB10 `256` accepted while local smaller blocks drift; ordered-256 CPU math equals `1024`, but rejected device implementation was a boundary/implementation problem. | Covered, but device guard not implemented |
| Burn/CubeCL LocalTuner behavior | Skill records cache-hit vs cache-miss, persistent key plus tunable checksum, warmup/profiling on misses, and `autotune-checks` not being production accuracy guard. | Covered as guidance |
| Accuracy guard for sensitive candidates | Trace-backed compare is used for candidate acceptance; ordered-256 is kept reverted until a device-side/trace-backed guard proves equivalence. | Partially covered |
| Remote `10.100.1.253` acceptance | Latest backward-key remote R9 compare: activation `54/54`, module-sum speedup `1.47x`, only known `lm_head/projection` row at `0.98x`. | Total covered, all-row target not covered |
| Local acceptance | User asked to avoid local GPU timing. Latest memory/notes indicate previous local runs were below `1.0` or stale after many changes. | Missing current verification |
| ncu occupancy/warp/memory counters | Remote normal user is blocked by `ERR_NVGPUCTRPERM`, even for LaunchStats. A privileged ncu command plan exists, but no counters are available yet. | Missing due permission |
| Poor-result root-cause analysis | LayerNorm ordered-256 root cause, residual negative, channel-mixer TMA/forced-Cube, WKV7 segment recompute, lm-head row variants, and LocalTuner bypass have notes. | Covered for attempted directions |

## Current Gaps

- Goal is not complete.
- Local speedup `>1.0` is not currently verified, and the user explicitly said local GPU timing may be inaccurate now.
- Remote total speedup is above `1.0`, but R9 all-row timing still has `lm_head/projection` near-parity/slightly slow (`0.98x` in the latest run).
- `ncu` counter requirements are unmet on `10.100.1.253` because GPU performance counters are restricted for `caizus`.
- Ordered-256 has a mathematical diagnosis, but no live device-side guard. It must stay reverted until such a guard exists.
- Remaining high-impact remote surfaces are mostly broad Cubek/TMA matmul or algorithmic WKV7. Recorded WKV7 f32 `P/Q` segment recompute is negative and must not be repeated unchanged.

## Next Concrete Action

- Do not mark the goal complete.
- Do not rerun local timing.
- Do not rerun remote `ncu` under the same permission boundary.
- The next non-duplicate implementation track should be a design/prototype branch for `lm_head` projection-loss fusion or another broad matmul-aware operator boundary, because small lm-head row-kernel variants and WKV7 row/segment variants are already duplicate-guarded.
- Before implementation, inspect the current `RwkvLM::forward`, trace writer timing boundary, Burn `Linear`/matmul public hooks, and lm-head loss kernel interface to determine whether a tensor-core-preserving projection-loss fusion is feasible without forking a low-quality GEMM.
