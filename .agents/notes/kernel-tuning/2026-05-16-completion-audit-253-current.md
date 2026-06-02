# Completion Audit 253 Current

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-completion-audit-253-current-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout contains broad unrelated workspace edits plus accumulated kernel source, skill, and note work. This audit owns only this note unless a later entry explicitly expands scope.
- User constraint: continue remote-first on `10.100.1.253`; do not use local GPU timing because the user is using the local GPU for other work.
- Scope: completion audit against the active kernel-tuning objective after the fresh 253 compare and `nsys` run. This branch does not edit kernel source, clear autotune caches, run local GPU workloads, or repeat blocked `ncu` counter commands.
- Prior-note and memory search commands already run before this audit:
  - `rg -n "kernel|LayerNorm|ordered-256|GB10|10\\.100\\.1\\.253|ncu|残差" /root/.codex/memories/MEMORY.md`
  - `rg -n "ncu|RmProfilingAdminOnly|ERR_NVGPUCTRPERM|10\\.100\\.1\\.253|spark|privileged|admin" .agents/notes/kernel-tuning/2026-05-16-privileged-ncu-command-plan-gb10.md .agents/notes/kernel-tuning`
  - `rg -n "lm_head_l2wrap|target-logit|row-loss|forward_row|atomic|GatedReadout|gated_readout|key_prepare|warps_per_cube|mix6_forward|channel_mixer_relu_square|LayerNorm|ordered-256|WKV7 source-level|kernel_binop" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md`
- Matched evidence:
  - The kernel-tuning skill records branch/note discipline, LocalTuner cache/warmup/accuracy-guard limitations, hardware/shape key requirements, LayerNorm GB10/local drift handling, and duplicate guards for residual-add, WKV7, key-prepare, GatedReadout, channel-mixer, lm-head, Mix6, and ordered-256 LayerNorm.
  - `2026-05-16-privileged-ncu-command-plan-gb10.md` now records fresh 253 compare, fresh 253 `nsys`, and the NVIDIA counter-permission unblock path.
  - Remote `/home/caizus/Projects/Packages/rwkv-rs-stable` is an rsync mirror without `.git`; branch provenance is therefore local, while remote command logs and generated profiler artifacts are evidence.
- Machine/GPU evidence boundary: remote `10.100.1.253` / `spark-35ac` / `NVIDIA GB10`, driver `580.95.05`, baseline `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`. Local GPU is intentionally excluded in this audit.
- Objective restatement / concrete deliverables:
  1. Kernel tuning iterations use branch/worktree plus prewritten notes; failed attempts revert code but keep branch/note evidence.
  2. Kernel implementation selection is keyed by hardware and shape where alternatives exist, including backend/runtime, GPU capability/fingerprint, dtype, `d_model`, rows, block size, warp count, vector width, alias/in-place, and deterministic numeric boundary.
  3. LayerNorm no longer uses a universal hard-coded block size; GB10 `256` and local drift-sensitive behavior are separated by runtime dispatch/autotune policy.
  4. Burn/CubeCL `LocalTuner::execute` behavior is understood: key creation, in-memory and persistent cache, warmup/profiling boundary, candidate naming/checksum, production accuracy-guard limitation, and hot-path lookup cost.
  5. CUDA-level design has been investigated for warp/block configuration, theoretical/actual occupancy where available, register pressure, shared memory, memory/coalescing, BF16 numerical drift, and failed-attempt root causes.
  6. `ncu` achieved occupancy, warp execution efficiency, and memory throughput are collected on the target machines, or an environment blocker is proven and documented.
  7. Final acceptance requires trace-backed activation passing and timing speedup `>1.0` on both the local GPU and `10.100.1.253`.
- Next commands: inspect current key/source evidence, fresh remote compare/profile artifacts, and note/skill coverage enough to build the prompt-to-artifact checklist. Do not run local GPU or blocked remote `ncu`.

## Evidence Inspection

- Source/key scan command:
  - `rg -n "struct .*AutotuneKey|runtime:|dtype:|d_model|embedded_dim|rows|num_tokens|block_size|warps_per_cube|num_warps|line_size|is_in_place|alias|deterministic|deterministic_min_block_size|CubeHardwareFingerprint|hardware:" crates/rwkv-nn/src/kernels/train -S`
- Source files inspected:
  - `crates/rwkv-nn/src/kernels/train/layer_norm/forward.rs`
  - `crates/rwkv-nn/src/kernels/train/layout.rs`
  - `.agents/skills/kernel-tuning/SKILL.md`
- LayerNorm source evidence:
  - `LayerNormForwardAutotuneKey` includes `runtime`, `dtype`, `d_model`, `rows`, `num_elements`, `hardware`, `is_in_place`, `deterministic`, and `deterministic_min_block_size`.
  - `BLOCK_SIZE_CANDIDATES = [64, 128, 256, 512, 768, 1024]`; candidate names are `block_{block_size}`.
  - `layer_norm(...)` derives `num_warps = block_size / WARP_SIZE` and launches with `CubeDim::new_1d(block_size)`.
  - BF16 deterministic candidates must satisfy `block_size >= deterministic_min_block_size`.
  - `supports_gb10_bf16_d768_layer_norm(...)` only admits `256` for runtime `cuda`, dtype `BF16`, `d_model=768`, `rows=8192`, and the GB10-like CubeCL hardware fingerprint.
- Hardware fingerprint evidence:
  - `CubeHardwareFingerprint` records `load_width`, `plane_size`, `max_units_per_cube`, `max_cube_dim`, `max_shared_memory_size`, `max_vector_size`, `num_streaming_multiprocessors`, `num_tensor_cores`, and `min_tensor_cores_dim`.
  - Literal CUDA compute capability is still not exposed through the current public CubeCL `HardwareProperties`; this is covered by project fingerprint fields rather than literal major/minor CC.
- Broader key scan evidence:
  - Live tuned keys for LayerNorm, lm-head/l2wrap CE, backward gates, channel-mixer backward, learning-rate gate backward, weight-decay transform, and related train kernels include runtime/dtype/shape rows/hardware/vector or block candidates/in-place/deterministic fields where alternatives exist.
  - Fixed-dispatch paths such as KeyPrepare `head_size == 64`, GatedReadout combine, and WKV7 row-state kernels are documented as fixed-policy boundaries; changing those constants requires a fresh branch plus explicit keyed dispatch/autotune wrapper or fixed-policy evidence.
- Fresh 253 compare evidence:
  - Command/log: `target/rwkv-test/remote-253-only-current-compare.log`.
  - Activation: `activation_summary compared=54 passed=54 failed=0 missing=0 extra=0`.
  - Timing: `timing_summary compared=76 passed=76 failed=0 missing=0 extra=0 ignored=1 actual_total_ms=73.373 baseline_total_ms=175.887 speedup=2.40x`.
- Fresh 253 `nsys` evidence:
  - Report: `target/rwkv-test/nsys-253-only-current-gb10.nsys-rep`.
  - SQLite: `target/rwkv-test/nsys-253-only-current-gb10.sqlite`.
  - Top current groups:
    - projection-like Cubek/TMA matmul: `47.023ms / 3`, grid `4096x16x1`, block `32x12x1`, regs/thread `73`, dynamic smem `27648`.
    - WKV7 output: `22.294ms / 36`, grid `12x16x1`, block `64x1x1`, regs/thread `108`, dynamic smem `1536`.
    - Mix6: `14.862ms / 36`, grid `24576x1x1`, block `32x8x1`, regs/thread `32`.
    - ChannelMixer ReLU-square: `14.510ms / 36`, grid `49152x1x1`, block `32x8x1`, regs/thread `16`.
    - lm-head row loss: `14.007ms / 3`, grid `8192x1x1`, block `1024x1x1`, regs/thread `40`, dynamic smem `128`.
    - GatedReadout combine: `11.931ms / 36`, grid `8192x12x1`, block `64x1x1`, regs/thread `25`.
    - KeyPrepare: `9.388ms / 36`, grid `24576x1x1`, block `128x1x1`, regs/thread `24`.
    - residual BF16 binop: `8.788ms / 72`, grid `3072x1x1`, block `32x8x1`, regs/thread `16`.
    - LayerNorm: `6.350ms / 78`, grid `8192x1x1`, block `256x1x1`, regs/thread `40`, dynamic smem `32`.
- Precision-root-cause evidence:
  - `2026-05-16-layernorm-ordered256-accuracy-rootcause.md` shows CPU-simulated intended ordered-256 math exactly matches the `1024` reduction for checked LayerNorm inputs, while ordinary `256` can create BF16 output deltas at cell pre-LayerNorm boundaries.
  - `2026-05-16-layernorm-ordered256-reconstruct.md` then reconstructed a device ordered-256 implementation that still failed activation. This classifies the precision issue as device implementation/boundary hygiene, not the intended mathematical iteration.
- `ncu` evidence:
  - Remote `/proc/driver/nvidia/params` reports `RmProfilingAdminOnly: 1`.
  - Prior remote `ncu` CSVs contain `ERR_NVGPUCTRPERM`.
  - Fresh current audit therefore does not rerun `ncu` as normal user; achieved occupancy, warp execution efficiency, memory throughput, and stall reasons remain blocked on admin counter access.

## Prompt-To-Artifact Checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| New tuning attempts use branch/worktree plus prewritten note. | Current branch `kernel-tuning-completion-audit-253-current-20260516`; many prior attempt notes; skill ledger rule. | Covered for recorded attempts. |
| Failed attempts revert code but keep branch/note evidence. | Skill guards and notes for ordered-256 LayerNorm, WKV7 segment/output-factor/direct-value, key-prepare warps, GatedReadout variants, residual fusion, channel-mixer variants, lm-head variants. | Covered by notes/skill; git tree remains dirty. |
| backend/runtime in autotune key. | LayerNorm and other tuned keys use `runtime: R::name(...)`; skill current audit says live keys include runtime/backend. | Covered where alternatives exist. |
| GPU arch/compute capability. | Public CubeCL literal CC is unavailable; `CubeHardwareFingerprint` records available hardware fields and SM/tensor-core fields. | Covered as hardware fingerprint, not literal CC. |
| dtype. | LayerNorm and live tuned keys include `dtype: DType`. | Covered. |
| `d_model`. | LayerNorm key has `d_model`; elementwise/gate/channel keys use `d_model` or `embedded_dim`; lm-head uses `num_tokens` plus reduction axis constraints. | Covered where semantically relevant. |
| rows `B*T`. | LayerNorm key has `rows`; elementwise/gate/channel keys anchor rows. | Covered. |
| block size. | Candidate names/groups encode `block_{block_size}` for reduction kernels; launch passes `CubeDim` from block size. | Covered for candidate paths. |
| num_warps. | LayerNorm and lm-head derive `num_warps = block_size / WARP_SIZE`; KeyPrepare fixed warp policy is documented. | Covered or fixed-policy documented. |
| vector width. | Line-size candidates and `max_line_size` fields cover vector width for elementwise/gate paths; candidate names encode `line_size_*`. | Covered where vectorized candidates exist. |
| in-place / alias. | Keys include `is_in_place`; channel/residual notes cover alias constraints. | Covered for live tuned paths. |
| deterministic numeric boundary. | Keys include `deterministic`; LayerNorm includes `deterministic_min_block_size`; ordered-256 is blocked pending device/trace guard. | Covered. |
| LayerNorm remote GB10 `256` but local drift-sensitive path protected. | `supports_gb10_bf16_d768_layer_norm` gates `256` to the remote-like fingerprint; fallback deterministic minimum is `next_power_of_two(d_model)` capped by hardware. | Covered. |
| Burn/CubeCL LocalTuner mechanics understood. | Skill records cache hit/miss split, `LocalTuner::init`, warmup/profiling on miss, persistent checksum, `autotune-checks` limitation, hot-path lookup expectation. | Covered in skill, not reimplemented here. |
| Candidate parameters identified. | Candidate names/groups cover block size, row tile, line size/vector width; fixed-policy boundaries are documented. | Covered. |
| Host-side tuner lookup cost not counted into hot path or proven negligible. | Skill states cache-hit overhead is key lookup plus dispatch and must be proven before bypassing; prior local tuner bypass was a negative result. | Covered by process evidence; no fresh profiler proof needed because no bypass branch is active. |
| CUDA warp/block/register/shared-memory/coalescing evidence. | Fresh 253 `nsys` records launch geometry, registers, and shared memory; notes record theoretical occupancy and failed-attempt reasons. | Partially covered; actual occupancy counters blocked. |
| BF16 reduction numerical issue investigated for code-vs-math root cause. | Ordered-256 CPU diagnostic and reconstructed device failure notes classify the issue as implementation/boundary failure rather than intended math. | Covered for known LayerNorm precision issue. |
| Poor results investigated before revert. | Skill and notes document root causes or likely limits for WKV7, key-prepare, GatedReadout, lm-head, residual, channel-mixer, value-residual, and LayerNorm. | Covered for recorded attempts. |
| `ncu` achieved occupancy / warp efficiency / memory throughput. | Remote `ncu` remains blocked by `RmProfilingAdminOnly: 1` / `ERR_NVGPUCTRPERM`; runbook documents admin unblock. | Incomplete / environment-blocked. |
| 253 speedup `>1.0`. | Fresh compare: `2.40x`, activation `54/54`, timing `76/76`. | Covered. |
| Local speedup `>1.0`. | The user explicitly asked not to use local GPU now; this audit did not run local acceptance. Existing local notes are stale relative to the final objective. | Incomplete. |

## Completion Decision

- Do not mark the active goal complete.
- Covered now: branch/note discipline for recorded attempts, hardware/shape keyed implementation selection for live tuned paths, LayerNorm GB10/local policy, LocalTuner mechanics, fresh 253 activation/timing, fresh 253 `nsys` launch/register/shared-memory profile, and LayerNorm ordered-256 root-cause classification.
- Missing or blocked:
  - Fresh local speedup `>1.0` is not verified because local GPU is reserved by the user.
  - Remote `ncu` achieved occupancy, warp execution efficiency, memory throughput, and stall counters remain blocked by NVIDIA profiling permissions.
  - The largest remaining 253 surfaces are now either broad Cubek/TMA matmul work or duplicate-closed custom kernels; opening another small kernel branch from this same `nsys` evidence would repeat known negative paths unless `ncu` or a genuinely new algorithm boundary changes the evidence.
- Next useful unblockers:
  - admin-enabled `ncu` on `10.100.1.253` using the existing runbook filters; or
  - a local acceptance rerun when the local GPU is free; or
  - a fresh broad design branch for upstream-style Cubek/TMA epilogue or GEMM-quality fused projection/loss, explicitly outside the small per-kernel tweak boundary.
