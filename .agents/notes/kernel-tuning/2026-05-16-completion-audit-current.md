# Completion Audit Current

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-completion-audit-current-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits and accumulated kernel tuning notes/source. This audit owns only this note unless a later entry explicitly records a skill/doc edit.
- User constraint: continue remote-first on `10.100.1.253`; do not use local GPU timing while it may be inaccurate or reserved.
- Prior-note search terms planned:
  - `completion-audit|current-goal|achieved|speedup|local|10.100.1.253|R9|projection|ncu|ERR_NVGPUCTRPERM`
  - `AutotuneKey|CubeHardwareFingerprint|deterministic|LayerNorm|ordered-256|LocalTuner|accuracy guard`
  - `WKV7|mix6|key_prepare|gated_readout|channel_mixer|lm_head|weight_decay|value_residual`
- Objective restated as concrete deliverables:
  1. Tuning attempts use branch/worktree plus prewritten notes; failed code is reverted but branches/notes are retained.
  2. Kernel selection keys/candidates include runtime/backend, hardware fingerprint or compute capability where exposed, dtype, shape (`d_model`, rows), tunable parameters (`block_size`, `num_warps`, vector width), alias/in-place, and deterministic numeric boundary.
  3. LayerNorm is dispatched/autotuned by hardware/shape/policy, with GB10 BF16 D768 able to use `256` and local deterministic path protected from the known drift.
  4. Burn/CubeCL `LocalTuner::execute` mechanics are understood and recorded: keys, cache, warmup/measurement boundary, candidate params, and lack of production accuracy guard.
  5. CUDA-level evidence covers warp/block, register/shared-memory, coalescing/memory, occupancy where permissions allow, and BF16 reduction/numeric drift analysis.
  6. Poor timing or precision results are diagnosed for math/implementation/measurement cause before keep/revert.
  7. Final acceptance requires activation passing and speedup `>1.0` on both local and `10.100.1.253`.
- Expected audit boundary: inspect real files, notes, source, and existing remote logs. Do not mark the goal complete unless every checklist item is covered by evidence. If any item is missing or weakly verified, record the gap and continue from the next non-duplicate action.
- Next command: inspect current branch/status, key source files, kernel-tuning skill, and current audit/projection/local/remote evidence notes.

## Evidence Inspection

- Commands:
  - `rtk git status --short --branch`
  - `rtk rg -n "struct .*AutotuneKey|CubeHardwareFingerprint|runtime:|dtype:|rows:|d_model|embedded_dim|block_size|warps|line_size|row_tile|is_in_place|deterministic|deterministic_min_block_size" crates/rwkv-nn/src/kernels/train -S`
  - `rtk sed -n '1,140p' .agents/skills/kernel-tuning/SKILL.md`
  - `rtk rg -n "activation_summary|timing_summary|speedup|R9|projection|local regenerated|ncu|ERR_NVGPUCTRPERM|RmProfilingAdminOnly|ordered-256|value_from_first_cell|lm_head/embedded_context|LocalTuner|accuracy guard|autotune-checks|not complete|goal is not complete" .agents/notes/kernel-tuning -S`
- Current branch/status:
  - branch is `kernel-tuning-completion-audit-current-20260516`.
  - worktree remains broadly dirty from accumulated workspace work; this audit owns only this note unless updated later.
- Source evidence:
  - `CubeHardwareFingerprint` exists in `crates/rwkv-nn/src/kernels/train/layout.rs`.
  - `layer_norm/forward.rs` has `LayerNormForwardAutotuneKey` with runtime, dtype, `d_model`, rows, hardware fingerprint, alias/in-place, deterministic flag, and `deterministic_min_block_size`; candidates are named `block_*` and launch passes `num_warps = block_size / WARP_SIZE`.
  - `lm_head_l2wrap_ce/forward.rs` has runtime, dtype, token/vocab shape, hardware fingerprint, alias/in-place, deterministic flag, and `block_*` candidates; kernels pass block size and warp count.
  - `mix6`, `weight_decay_transform`, value/learning gates, channel-mixer backward, and residual-add keys carry runtime, dtype, rows/embedded shape, hardware fingerprint where there is a live candidate set, vector width/candidate names, alias/in-place, and deterministic fields.
  - Fixed-shape kernels such as current key-prepare and WKV7 state/backward do not have live alternative candidate sets after rejected attempts; prior notes classify them as not autotune-key gaps until a materially different implementation exists.
- Skill evidence:
  - branch+note discipline and duplicate guards are recorded.
  - LocalTuner mechanics are recorded: cache-hit vs cache-miss, `init` caching, persistent checksum, warmup/profiling miss cost, `autotune-checks` not being a production accuracy guard, and host-side lookup caveat.
  - Remote `ncu` blocker is recorded: `RmProfilingAdminOnly: 1`, no passwordless sudo, use `nsys` metadata until privileged access changes.
  - Projection+loss design guard is recorded: no naive row-dot/post-logits fusion; a real fusion needs TMA epilogue or full fused operator/backward.
- Runtime evidence:
  - Remote clean gate evidence exists: `2026-05-16-channel-mixer-edge-rerun-after-projection-timing.md` records activation `54/54`, timing `76/76`, `actual_total_ms=83.677`, `baseline_total_ms=175.887`, speedup `2.10x`.
  - Remote R9 projection baseline evidence exists: `2026-05-16-current-goal-audit-after-r9.md` records activation `54/54`, total speedup `1.48x`, with the only failed R9 row `lm_head/projection` at `0.99x`.
  - Local regenerated-baseline evidence exists: `2026-05-16-local-regenerated-baseline.md` records activation `54/54` and total speedup `2.44x`, but row-level timing still failed (`22/76` pass) and the note classifies remaining boundary/provenance gaps.
  - Older checked-in local baseline evidence still had sub-1 total speedup; current audit should not mix that baseline with the regenerated-baseline result.
- Precision evidence:
  - LayerNorm ordered-256 precision work is recorded as a device implementation/boundary failure, not a math-iteration failure: CPU ordered diagnostics matched `1024`, but device candidate failed `value_from_first_cell` and `lm_head/embedded_context`.
- CUDA profiler evidence:
  - Local historical `ncu` notes record occupancy/memory/register evidence for several kernels.
  - Remote `10.100.1.253` cannot provide achieved occupancy, warp execution efficiency, or memory-throughput counters in the current user permission boundary; only `nsys` launch/register/shared-memory metadata and standard compare are available.

## Prompt-To-Artifact Checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| New tuning attempts use branch/worktree plus prewritten note; failed code reverted, branches/notes retained. | Current branch and this note exist; skill enforces strict ledger order; prior notes contain keep/revert decisions. | Covered for current workflow |
| Kernel design selects by runtime/backend, hardware/shape, dtype, rows, tunable params, alias/in-place, deterministic boundary. | Source grep confirms these fields in tuned keys; fixed-shape kernels have no live second candidate after rejected attempts. | Mostly covered |
| Include GPU arch / compute capability. | CubeCL public API does not expose literal CUDA CC in the project key path; keys use `CubeHardwareFingerprint` fields, and skill records this limitation. | Covered via available fingerprint, literal CC not exposed |
| LayerNorm no longer hard-coded universal constant. | `LayerNormForwardAutotuneKey`, `BLOCK_SIZE_CANDIDATES`, deterministic min block policy, and GB10 BF16 D768 support guard exist. | Covered |
| GB10 `256` vs local `1024` precision conflict analyzed. | Ordered-256/root-cause notes classify math vs device implementation/boundary and preserve drift rows. | Covered |
| Burn/CubeCL `LocalTuner::execute` key/cache/warmup/accuracy guard understood. | Kernel-tuning skill records the mechanism and limitations. | Covered |
| Candidate parameters are represented in candidate names/checksum. | Notes/source show `block_*`, `line_size_*`, `row_tile_*`, and rejected `warps_per_cube_*` candidates. | Covered where candidates exist |
| Host-side tuner lookup cost is not counted as hot-path regression without proof. | Skill records cache-hit lookup/dispatch distinction and requires profiler proof before bypassing. | Covered |
| CUDA lower-level design uses profiler evidence. | Local `ncu` plus remote `nsys` metadata exist; remote counter-level `ncu` is blocked by permissions. | Partially covered |
| Bad timing/precision results diagnosed before revert. | Notes cover WKV7 segment/shared-lanes, key-prepare warps, lm-head variants, channel-mixer, value-residual, projection+loss, and LayerNorm precision diagnosis. | Covered for recorded attempts |
| Final activation + speedup `>1.0` on `10.100.1.253`. | Remote standard gate activation `54/54`, timing `76/76`, speedup `2.10x`; R9 total speedup `1.48x` with projection row near parity. | Covered for total speedup, not all-row R9 |
| Final activation + speedup `>1.0` on local. | Local regenerated-baseline total speedup `2.44x`, activation `54/54`, but row-level failures and baseline-boundary gaps remain; current user constraint says local GPU timing may be inaccurate. | Partially covered |

## Audit Decision

- Do not mark the goal complete.
- Missing or weakly verified requirements:
  - Remote `ncu` counter evidence for achieved occupancy, warp execution efficiency, and memory throughput is blocked by `RmProfilingAdminOnly: 1`.
  - Local clean row-level acceptance is not achieved; local total speedup evidence exists only under a regenerated baseline with row-level failures.
  - R9 remote timing includes `lm_head/projection` and total speedup is positive, but the projection row itself remains `0.99x`, so an all-row interpretation is still incomplete.
- Next non-duplicate action boundary:
  - Do not rerun WKV7 row-tile/shared-lanes/segment, Mix6 line-size/axis, key-prepare warps, channel-mixer forced-Cube/Burn/fusion/line-size, or lm-head row-kernel variants.
  - Continue only with a materially new algorithm/operator boundary, privileged remote `ncu`, or a local clean timing/baseline-boundary pass when the local GPU is available.
