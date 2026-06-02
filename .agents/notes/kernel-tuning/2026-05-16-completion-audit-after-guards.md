# Completion Audit After Guards

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-completion-audit-after-guards-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits plus accumulated kernel-tuning source, skill, and note changes. This attempt owns only this audit note unless a later entry explicitly expands scope.
- User constraint: continue remote-first on `10.100.1.253`; do not run local GPU timing while the user may be using the local GPU.
- Scope: completion audit against the active kernel-tuning objective after the latest WKV7/current-top-surface skill guard updates. This branch does not edit kernel source, clear caches, run new benchmarks, or run local GPU workloads.
- Prior search commands already run:
  - `rg -n "completion audit|current goal audit|autotune key|LayerNorm|LocalTuner|ncu|speedup|10\\.100\\.1\\.253|local|WKV7 source-level|value_residual_gate.*vector-axis|Mix6 forward autotune|channel-mixer forward line-size" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md -S`
  - `git status --short --branch`
  - `sed -n '1,130p' .agents/skills/kernel-tuning/SKILL.md`
- Matched evidence:
  - Previous completion audits already identify remote GB10 compare as clean and `ncu` achieved-counter collection as permission-blocked.
  - Current skill now records WKV7 source-level root cause, value-residual vector-axis keep/neutral, Mix6 line-size selection, channel-mixer line-size/cache guards, LocalTuner mechanics, LayerNorm deterministic guard, and the remote `ncu` blocker.
  - Current branch status is dirty with many unrelated workspace changes and untracked `.agents/notes` / `.agents/skills/kernel-tuning`; this audit must not claim a clean source tree.
- Machine/GPU: evidence source is remote `10.100.1.253` / NVIDIA GB10 unless otherwise stated. Local GPU timing is intentionally not refreshed.
- Objective restatement / concrete success criteria:
  1. Every materially new tuning attempt uses a fresh branch/worktree and prewritten note; failed attempts are reverted but branches/notes retained.
  2. Kernel implementation selection is keyed by hardware and shape where implementation choices exist, covering backend/runtime, GPU capability/fingerprint, dtype, `d_model`, rows, block size, num_warps, vector width, alias/in-place, and deterministic boundary.
  3. LayerNorm is not a hard-coded constant: remote GB10 may use activation-clean `256`, while local deterministic-sensitive paths avoid known drift.
  4. Burn/CubeCL autotune mechanics are documented: `LocalTuner::execute` key/cache/warmup/profile boundaries, candidate parameters, host lookup overhead expectations, and lack of production accuracy guard.
  5. CUDA-level design has been investigated for reduction block/warp choices, occupancy/register/shared-memory/coalescing, BF16 numeric drift, and poor-result root causes.
  6. `ncu` achieved occupancy, warp execution efficiency, and memory throughput are collected where possible or the blocker is proven.
  7. Final acceptance requires trace-backed activation passing and timing speedup `> 1.0` on both local and `10.100.1.253`.
- Next command: inspect current live source/key files and latest remote artifacts enough to fill a prompt-to-artifact checklist, without running new GPU work.

## Evidence Inspection

- Commands:
  - Source scan for `AutotuneKey`, `runtime`, `dtype`, `d_model`, `rows`, `hardware`, `max_line_size`, `is_in_place`, `deterministic`, `deterministic_min_block_size`, `BLOCK_SIZE_CANDIDATES`, `WARP_SIZE`, `row_tile`, and `CubeHardwareFingerprint` under `crates/rwkv-nn/src/kernels/train`.
  - Read `crates/rwkv-nn/src/kernels/train/layer_norm/forward.rs`.
  - Read `crates/rwkv-nn/src/kernels/train/layout.rs`.
  - Remote read-only parse of current compare logs and ncu CSV artifacts on `10.100.1.253`.
- Source/key evidence:
  - `CubeHardwareFingerprint` records `load_width`, `plane_size`, `max_units_per_cube`, `max_cube_dim`, `max_shared_memory_size`, `max_vector_size`, `num_streaming_multiprocessors`, `num_tensor_cores`, and `min_tensor_cores_dim`.
  - LayerNorm has `BLOCK_SIZE_CANDIDATES = [64, 128, 256, 512, 768, 1024]`, `WARP_SIZE = 32`, and `LayerNormForwardAutotuneKey` with `runtime`, `dtype`, `d_model`, `rows`, `num_elements`, `hardware`, `is_in_place`, `deterministic`, and `deterministic_min_block_size`.
  - LayerNorm launch passes `block_size` and derives `num_warps = block_size / WARP_SIZE`.
  - LayerNorm deterministic policy allows `256` only for CUDA BF16 `d_model=768`, `rows=8192`, and the GB10-like hardware fingerprint with `48` SMs; other BF16 paths default to `next_power_of_two(d_model)` capped by hardware max units.
  - WKV7, lm-head, channel-mixer, Mix6, learning-rate gate, value-residual gate, weight-decay transform, residual-add, and backward keys expose runtime/dtype/shape/hardware/vector or block candidates where live candidate sets exist. Fixed paths are documented in the skill as fixed-policy boundaries.
- Remote compare evidence:
  - `remote-current-after-gb10-results-compare.log`: activation `54/54`; timing `75/76` with one ignored row; `actual_total_ms=72.722`, `baseline_total_ms=175.887`, `speedup=2.42x`.
  - `remote-wkv7-output-factored-reverted-compare.log`: activation `54/54`; timing `76/76`; `actual_total_ms=72.592`, `baseline_total_ms=175.887`, `speedup=2.42x`.
  - `remote-layernorm-hardware-fingerprint-helper-r9-repeat9-compare.log`: activation `54/54`; timing `76/77`; `actual_total_ms=87.373`, `baseline_total_ms=130.484`, `speedup=1.49x`.
  - `remote-residual-matmul-fusion-compare.log`: activation `54/54`; timing `73/76`; `actual_total_ms=79.595`, `baseline_total_ms=175.887`, `speedup=2.21x`; this supports the negative global-fusion guard, not a kept result.
- Remote `ncu` evidence:
  - `/proc/driver/nvidia/params` reports `RmProfilingAdminOnly: 1`.
  - `target/rwkv-test/ncu-gated-readout-groupnorm-combine.csv` and `target/rwkv-test/ncu-weight-decay-transform-gb10-retry.csv` both contain `ERR_NVGPUCTRPERM`.
  - Therefore achieved occupancy / warp execution efficiency / memory throughput remain unavailable on 253 without privileged counter access.

## Prompt-To-Artifact Checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| New tuning attempt uses new branch and prewritten note. | Current branch plus this note; skill ledger rule; many branch notes. | Covered for recorded attempts. |
| Failed attempts revert but branches are retained. | WKV7 direct-value/output-factor/segment notes, key-prepare warps, GatedReadout variants, LayerNorm ordered/split-tail notes and skill guards. | Covered by notes/skill, not by git clean state. |
| Backend/runtime in key. | Live tuned keys use `runtime: R::name(...)`. | Covered. |
| GPU architecture / capability in key. | `CubeHardwareFingerprint` included in live keys where public CubeCL hardware fields are available; skill documents literal CC is not public. | Covered as hardware fingerprint, not literal CC. |
| dtype in key. | Live tuned keys include `dtype: DType`. | Covered. |
| `d_model` / rows. | LayerNorm and WKV7 directly include both; elementwise keys include embedded/innermost dim and rows. | Covered for live tuned kernels. |
| block size / num_warps / vector width. | Candidate names/groups encode `block_*`, `row_tile_*`, `line_size_*`; LayerNorm/lm-head derive warps from block. Fixed-dispatch kernels require explicit future wrapper before constants change. | Covered where candidate sets exist. |
| in-place / alias. | Keys include `is_in_place`; channel-mixer relu-square uses `can_mut && is_nonoverlapping`; residual-add records lhs/rhs alias state. | Covered. |
| deterministic boundary. | Keys include `deterministic`; LayerNorm has `deterministic_min_block_size`; ordered-256 is blocked by skill until device-side/trace guard exists. | Covered for known sensitive paths. |
| LayerNorm remote 256 vs local drift. | Source dispatch gates `256` to the GB10 fingerprint; skill records local ordered/smaller-block drift and remote 256 keep. | Covered for current policy. |
| LocalTuner execute key/cache/warmup/accuracy guard understood. | Skill documents cache hit/miss, `LocalTuner::init`, persistent checksum, retune warmup/profiling, host lookup boundary, and `autotune-checks` limitation. | Covered. |
| Poor results investigated. | Skill guards and notes document WKV7 root cause, segment 13.2x, direct-value regression, output-factor synchronization cost, value-residual neutral result, residual fusion negative, ordered-256 device-boundary failure. | Covered for recorded attempts. |
| Remote speedup > 1.0 on 253. | Current/reverted clean remote logs show `2.42x` with activation `54/54`. | Covered. |
| Local speedup > 1.0. | Prior notes mention local regenerated-baseline evidence, but the user asked not to run local GPU now, and this audit did not refresh local timing after latest skill/notes. | Not freshly verified in this turn. |
| `ncu` achieved occupancy / warp efficiency / memory throughput. | Remote `ncu` is blocked by `RmProfilingAdminOnly: 1` and `ERR_NVGPUCTRPERM`; local GPU is intentionally excluded. | Incomplete / environment-blocked. |

## Completion Decision

- Do not mark the objective complete.
- Reason: remote GB10 acceptance and design/key coverage are strong, but two explicit objective requirements are still not fully satisfied under current constraints:
  - fresh current local speedup `> 1.0` is not verified because the user asked not to use the local GPU now;
  - `ncu` achieved occupancy / warp efficiency / memory throughput on 253 are blocked by NVIDIA counter permissions.
- Current next useful action is not another duplicate kernel branch. The next unblockers are either:
  - local acceptance rerun when the local GPU is free; or
  - privileged `ncu` on 253, or enabling non-admin NVIDIA performance counters, using the already documented kernel filters.
