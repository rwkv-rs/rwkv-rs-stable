# Current Goal Audit After Remote Clean Gate

- Date: 2026-05-16
- Branch/worktree: `kernel-tuning-current-goal-audit-remote-clean-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout inherits broad unrelated workspace changes and many prior tuning branches/notes. This branch is read-only except for this audit note.
- Prior-note/source search command:
  - `rg -n "autotune key|LocalTuner|LayerNorm|layer_norm|hardware|deterministic|ncu|remote clean|speedup|10\\.100\\.1\\.253|local compare|activation drift|channel_mixer|lm_head/projection" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md /root/.codex/memories/MEMORY.md -S`
- Objective restated as concrete deliverables:
  1. Tuning workflow uses branch/worktree plus prewritten notes for every new attempt; failed attempts are preserved and reverted when needed.
  2. Autotune/runtime dispatch keys cover backend/runtime, GPU hardware/architecture, dtype, `d_model`, rows, tunable parameters such as block size/warps/vector width, in-place/alias state, and deterministic numeric boundary.
  3. LayerNorm is not a universal hard-coded block-size constant; the GB10/local difference must be represented by autotune/runtime dispatch or a documented guarded policy.
  4. Burn/CubeCL `LocalTuner::execute` behavior is understood: cache key, persistent cache, warmup/profiling miss boundary, candidate parameters, and accuracy-guard limitation.
  5. CUDA-level evidence exists for reduction/warp/block design, occupancy/register/shared memory/memory coalescing where profiler permissions allow it; failures are analyzed for root cause, not only reverted.
  6. Remote `10.100.1.253` and local machine both eventually show trace-backed activation passing and timing speedup greater than `1.0`.
- Current known evidence before this audit:
  - Remote `10.100.1.253` post-projection-timing binary has a clean gate: `activation_summary compared=54 passed=54 failed=0`, `timing_summary compared=76 passed=76 failed=0 missing=0 extra=0 ignored=1 actual_total_ms=83.677 baseline_total_ms=175.887 speedup=2.10x`.
  - The local GPU should not be benchmarked right now because the user said it is being used for other work and may be inaccurate.
- Next command: inspect current notes and source keys to map each deliverable to concrete evidence and identify remaining gaps.

## Checklist Evidence

| Requirement | Evidence | Status |
| --- | --- | --- |
| Branch/worktree plus prewritten note for each new tuning attempt | Current attempts have dedicated branches and notes: projection timing contract, channel-mixer edge rerun, and this audit branch. The `kernel-tuning` skill now enforces strict ledger order. | Covered for current workflow |
| Failed attempts preserved and reverted rather than deleted | Negative notes exist for LayerNorm variants, lm-head online softmax, channel-mixer forced Cube/Burn-reference/tuner bypass, GatedReadout warp32, and others. | Covered |
| Runtime/backend, dtype, shape, hardware, alias, deterministic fields in autotune keys | `2026-05-16-autotune-key-audit.md` records current coverage. Source grep confirms key structs include `runtime`, `dtype`, shape fields, CubeCL hardware fingerprint, `is_in_place`/alias or equivalent, and `deterministic`. | Mostly covered |
| Direct CUDA compute capability in keys | CubeCL key sites do not expose portable CUDA CC major/minor. Current keys use hardware fingerprint fields (`plane_size`, `max_units_per_cube`, `max_cube_dim`, shared memory, vector size, SM count, tensor-core fields where available). | Covered by available fingerprint, not literal CC |
| Candidate parameters represented in tunable names/groups | Current tunables name `block_*`, `line_size_*`, row tiles, reduce block, and `bt_tile` combinations, so persistent cache checksum changes when candidate sets change. | Covered for current tuned families |
| LayerNorm avoids universal hard-coded block constant | `layer_norm/forward.rs` has `LayerNormForwardAutotuneKey`, block-size tunables, and a `deterministic_min_block_size` policy keyed by runtime, dtype, `d_model`, rows, and hardware fingerprint. Local-like non-GB10 falls back to `d_model.next_power_of_two()` while the GB10 D768/8192 fingerprint permits `256`. | Covered, with caveat below |
| LayerNorm accuracy root cause investigated | `2026-05-16-layernorm-ordered256-accuracy-rootcause.md` ran remote CPU diagnostics on real trace tensors. Ordered-256 math exactly matched 1024 for checked cases; ordinary 256 can produce BF16 output deltas at cell pre-LayerNorm boundaries. Conclusion: prior ordered-256 failure was implementation/boundary hygiene, not the intended mathematical iteration. | Covered |
| Burn/CubeCL LocalTuner mechanism understood | `2026-05-16-cubecl-local-tuner-findings.md` documents `LocalTuner::init`, `execute`, key cache, persistent cache checksum, miss warmup/profiling (`3` warmup + `10` profile samples), candidate representation, and `autotune-checks` limitation. | Covered |
| Host-side tuner overhead excluded from hot timing or confirmed small | Notes distinguish cache miss tuning cost from steady-state `.time.json`. Host hit overhead is understood qualitatively as key construction plus lookup/dispatch; no focused host-overhead profiler branch has proven it material. | Partially covered |
| CUDA ncu occupancy/warp/memory evidence | Local ncu notes record occupancy/memory metrics for several kernels. Remote `10.100.1.253` ncu counter collection is blocked by `ERR_NVGPUCTRPERM`; remote evidence uses nsys launch attribution and source/static analysis. | Partially covered |
| Poor results analyzed before rejection | Notes explain rejected results: LayerNorm drift root cause, lm-head online-softmax nsys regression, channel-mixer forced Cube regression, value-residual vector-axis neutral/slightly negative profile, projection+loss fusion risk. | Covered |
| Remote speedup > 1.0 with activation passing | Current remote clean gate after projection timing: activation `54/54`, timing `76/76`, speedup `2.10x`. | Covered |
| Local speedup > 1.0 with activation passing | Latest local evidence in notes remains below `1.0` after several branches, and the user explicitly said the local GPU is busy and may be inaccurate. | Not covered |

## Remaining Gaps

- Goal is not complete because local trace-backed timing speedup `>1.0` has not been achieved or freshly verified on an idle local GPU.
- Remote ncu hardware-counter evidence remains incomplete because `10.100.1.253` blocks counter collection with `ERR_NVGPUCTRPERM`; only local ncu or remote admin-enabled ncu can satisfy the literal remote-counter request.
- LayerNorm currently uses a runtime/hardware fingerprint policy plus autotune candidate filtering. That satisfies the current safety boundary, but a stronger design would add a project-level accuracy guard/cache before letting BF16 reduction-order candidates win automatically.
- `lm_head/projection` is now emitted as a noncanonical timing row. It should become canonical only after the regenerated baseline contract emits the same row.

## Decision

- Do not mark the active goal complete.
- The next GPU action should wait for an idle local GPU or explicit user permission to run local timing despite possible interference.
- Non-GPU follow-up can continue by making `lm_head/projection` canonical in both actual and baseline generation paths, or by designing a project-level accuracy guard for LayerNorm autotune candidates.
