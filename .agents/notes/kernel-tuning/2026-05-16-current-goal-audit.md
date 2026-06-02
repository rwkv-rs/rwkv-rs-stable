# Current Kernel-Tuning Goal Audit

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-current-goal-audit-20260516` in the existing dirty checkout.
- Dirty-tree constraint: this checkout carries broad unrelated uncommitted changes and previous tuning branches. This note is a read-only audit artifact and does not change kernel code.
- Prior evidence commands:
  - `rg -n "LayerNorm|layer_norm|runtime dispatch|deterministic_min_block_size|LocalTuner|AutotuneKey|GatedReadout|gated_readout|ncu|ERR_NVGPUCTRPERM|speedup=|timing_summary|activation_summary|10\\.100\\.1\\.253|local" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md -S`
  - `rg -n "deterministic_min_block_size|LayerNormForwardAutotuneKey|struct .*AutotuneKey|runtime_name|num_streaming_multiprocessors|max_units_per_cube|is_in_place|deterministic|block_size|num_warps|line_size" crates/rwkv-nn/src/kernels/train -S`

## Objective Restated

The active goal is not just one speedup run. Completion requires:

1. Kernel tuning attempts use git branches/worktrees plus prewritten notes; failed attempts are reverted but branches/notes are preserved.
2. Train kernel implementation choices are keyed or dispatched by backend/runtime, hardware capability, dtype, shape (`d_model`, rows), candidate parameters (`block_size`, `num_warps`, vector width), alias/in-place state, and deterministic numeric boundary where a candidate set exists.
3. LayerNorm handles the remote GB10/local accuracy split through hardware/shape/deterministic policy, not a universal hard-coded block size.
4. Burn/CubeCL `LocalTuner::execute` behavior is understood: key/cache, warmup/measurement boundary, candidate parameters, host-side cost, and accuracy-guard limitations.
5. CUDA-level evidence exists for reduction/block choices, register/shared-memory behavior, memory coalescing/throughput, and BF16 reduction accuracy differences; poor results are explained, not only reverted.
6. Final validation eventually shows speedup `>1.0` on both local GPU and remote `10.100.1.253` with trace-backed activation passing.

## Prompt-to-Artifact Checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| Branch per new attempt, notes before runs, failed branches preserved | `.agents/skills/kernel-tuning/SKILL.md` Branch And Notes section; many notes under `.agents/notes/kernel-tuning/`; current branches including rejected `kernel-tuning-gated-readout-warp32-gb10-20260516` | Covered operationally |
| LocalTuner mechanics inspected | `2026-05-16-skill-guardrails.md` records `cubecl-runtime-0.10.0-pre.4/src/tune/*` read; `kernel-tuning` skill now has `Burn/CubeCL Autotune Rules` | Covered |
| Autotune key dimensions audited | `2026-05-16-autotune-key-audit.md` matrix plus current source search show key structs carry runtime, dtype, shape, CubeCL hardware fingerprint, alias/in-place, deterministic fields where candidate sets exist | Mostly covered |
| LayerNorm remote/local dispatch | `layer_norm/forward.rs` has `LayerNormForwardAutotuneKey` with `deterministic_min_block_size`; notes record remote GB10 `256` path and local small-block drift | Covered for current BF16 D768 boundary |
| GatedReadout combine performance branch | `2026-05-16-gated-readout-forward-combine-gb10.md` and `2026-05-16-gated-readout-warp32-gb10.md`; 64-thread combine kept, warp32 rejected after profiler showed slower target kernel | Covered as one implementation branch |
| Remote speedup > 1.0 | Latest post-revert remote compare in `2026-05-16-gated-readout-warp32-gb10.md`: activation `54/54`, total `2.07x`, one known `cell_0000/channel_mixer` timing row fail | Partially covered; not a clean `76/76` |
| Local speedup > 1.0 | Latest local evidence in `2026-05-16-wire-time-mixer-gates-gb10.md`: activation `54/54`, total `0.77x`; user later asked to avoid local GPU while it is in use | Not achieved |
| ncu occupancy/warp/memory evidence on current remote | `2026-05-16-remote-channel-mixer-edge-ncu.md` records `ERR_NVGPUCTRPERM`; remote ncu counters unavailable. Remote `nsys` used for timing/launch attribution | Blocked on remote permissions |
| ncu CUDA evidence locally | Earlier local ncu notes exist for several kernels, but current user explicitly asked to avoid local GPU; current post-GatedReadout local state is not reprofiled | Not current |
| Poor-result root cause analysis | LayerNorm drift root-cause notes, warp32 nsys regression, channel-mixer forced-Cube negative ncu, and remote channel-mixer edge analysis all record explanations | Partially covered |

## Current Gaps

- Local final acceptance is still missing: the last recorded local total speedup is below `1.0`, and current user instruction says local GPU may be busy and should not be used for measurement.
- Remote acceptance is strong by total speedup but still not clean because `cell_0000/channel_mixer` can fail by a small margin; this has been characterized as a first-cell variance edge, but `compare-rwkv-nn` still exits nonzero on that run.
- Remote `ncu` counter evidence is blocked by GPU performance-counter permissions. We cannot honestly claim achieved occupancy, warp execution efficiency, or memory throughput for the current remote kernel state.
- `gated_readout_combine` has no autotune key because the only attempted alternate launch geometry, warp32, was slower. It is currently a fixed implementation, justified by trace/profiler evidence rather than an autotuned candidate set.

## Next Concrete Action

Do not mark the goal complete. The next implementation/profiling step should either:

- fix or de-noise the remaining remote `cell_0000/channel_mixer` timing boundary with a fresh branch and note, if clean remote `76/76` is required; or
- wait until local GPU is available and run the local post-GatedReadout compare/profiler path, because the explicit final objective still requires local speedup `>1.0`.

Until remote `ncu` permissions change, use `nsys` plus source/static evidence on `10.100.1.253`, and reserve full occupancy/warp/memory-throughput claims for local ncu or a remote admin-enabled run.
