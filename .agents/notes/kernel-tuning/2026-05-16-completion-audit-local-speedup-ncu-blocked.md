# Completion Audit Local Speedup NCU Blocked

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-completion-audit-local-speedup-ncu-blocked-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout contains broad unrelated workspace edits and previous kernel-tuning notes. This audit owns only this note unless a later entry explicitly expands scope.
- Scope: completion audit after fresh local acceptance showed total speedup above `1.0`, while remote `ncu` counter collection remains permission-blocked.
- Prior evidence inspected:
  - `2026-05-16-local-gpu-availability-preflight.md`
  - `target/rwkv-test/local-gpu-availability-acceptance-compare.log`
  - `2026-05-16-completion-audit-253-current.md`
  - `2026-05-16-ncu-blocker-verification-253.md`
  - `.agents/skills/kernel-tuning/SKILL.md`
- Objective restatement / concrete deliverables:
  1. Use branch/worktree plus prewritten note for each material tuning attempt; failed attempts revert code but keep branch/note evidence.
  2. Implement hardware/shape keyed kernel selection where alternatives exist, covering backend/runtime, hardware capability/fingerprint, dtype, `d_model`, rows, block size, warp count, vector width, alias/in-place, and deterministic boundary.
  3. LayerNorm must not be one hard-coded block size; GB10 `256` and local drift-sensitive behavior must be separated by policy/autotune/runtime dispatch.
  4. Burn/CubeCL LocalTuner mechanics must be understood and recorded.
  5. CUDA-level design must be investigated with launch geometry, register/shared-memory/coalescing reasoning, BF16 numerical diagnosis, and `ncu` achieved counters where possible.
  6. Current local and `10.100.1.253` compare must pass activation and show total speedup `>1.0`.
  7. `ncu` achieved occupancy, warp execution efficiency, and memory throughput must be collected, or the environment blocker must be proven and documented.

## Prompt-To-Artifact Checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| Branch/note workflow. | Current and prior kernel-tuning branches/notes; skill ledger rule. | Covered for recorded attempts. |
| Failed attempts reverted but retained. | Notes and skill guards for LayerNorm ordered-256, WKV7 segment/output-factor/direct-value, key-prepare warps, GatedReadout variants, residual fusion, channel-mixer variants, lm-head variants. | Covered by notes/skill. |
| backend/runtime in key. | Source audits show tuned keys use `runtime: R::name(...)`. | Covered. |
| GPU capability/fingerprint in key. | `CubeHardwareFingerprint` records CubeCL-exposed hardware fields; literal CUDA CC is not public in current CubeCL API. | Covered as fingerprint, not literal CC. |
| dtype. | Live tuned keys include `dtype`. | Covered. |
| `d_model` / rows. | LayerNorm and gate/channel keys include `d_model` or `embedded_dim` and rows; lm-head uses `num_tokens` and vocab boundary. | Covered where semantically relevant. |
| block size / num_warps. | Reduction candidate names encode block size; LayerNorm/lm-head derive `num_warps = block_size / 32`; fixed-policy paths documented. | Covered or fixed-policy documented. |
| vector width. | Elementwise/gate keys use `max_line_size` and `line_size_*` candidates. | Covered where vector candidates exist. |
| in-place / alias. | Tuned keys include `is_in_place`; residual/channel notes record alias constraints. | Covered. |
| deterministic boundary. | Keys include `deterministic`; LayerNorm has `deterministic_min_block_size`; ordered-256 blocked pending device/trace guard. | Covered. |
| LayerNorm GB10/local split. | `supports_gb10_bf16_d768_layer_norm(...)` gates `256` to remote-like BF16 D768 rows=8192 fingerprint; fallback deterministic policy stays larger. | Covered. |
| LocalTuner mechanics. | Skill records cache hit/miss, `LocalTuner::init`, warmup/profiling on miss, persistent checksum, `autotune-checks` limitation, and hot-path lookup expectation. | Covered. |
| CUDA design investigation. | Current 253 `nsys`, prior local `ncu`, WKV7/root-cause notes, LayerNorm precision diagnostics, and negative-attempt notes cover launch geometry/register/shared-memory/root-cause analysis. | Mostly covered; actual remote counters blocked. |
| BF16 precision issue investigated. | Ordered-256 CPU math matched `1024`; reconstructed device implementation still drifted, classifying failure as implementation/boundary hygiene. | Covered for known LayerNorm issue. |
| 253 speedup `>1.0`. | Fresh 253 compare: activation `54/54`, timing `76/76`, `speedup=2.40x`. | Covered. |
| Local speedup `>1.0`. | Fresh local compare: activation `54/54`, timing total `actual_total_ms=33.446`, `baseline_total_ms=35.957`, `speedup=1.08x`; row timing is not clean (`13/76`). | Covered for total speedup; row-level residual risk remains. |
| `ncu` achieved occupancy / warp execution / memory throughput. | 253 `ncu` is installed and section names are valid, but `RmProfilingAdminOnly: 1` plus failed `sudo -n true` prevents normal-user counter collection. | Incomplete / environment-blocked. |

## Completion Decision

- Do not mark the goal complete.
- The previous local speedup gap is now covered by a fresh local run: activation passed and total speedup is `1.08x`.
- The current remaining hard blocker is the explicit `ncu` counter requirement on 253: achieved occupancy, warp execution efficiency, and memory throughput cannot be collected as normal `caizus` under the current driver policy.
- No further small local/remote kernel branch should be opened from the current evidence without either:
  - admin-enabled 253 `ncu` counter data; or
  - a deliberately broader algorithm/operator boundary with a fresh branch and note.
