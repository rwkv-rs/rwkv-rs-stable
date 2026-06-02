# 2026-05-16 completion audit after sumsq rejection

- Branch/worktree: `kernel-tuning-completion-audit-20260516` in `/mnt/g/Projects/Packages/rwkv-rs-stable`.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits and prior kept/rejected kernel tuning changes. This audit owns only this note and read-only source/result inspection.
- Prior search command: `rg -n "completion|current-goal|autotune key|LocalTuner|ncu|ERR_NVGPUCTRPERM|speedup=|activation_summary|timing_summary|LayerNorm|ordered-256|10\\.100\\.1\\.253|local" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md /root/.codex/memories/MEMORY.md -S`.
- Matched evidence:
  - `kernel-tuning` skill requires branch+note for new tuning attempts, remote `10.100.1.253` as primary timing source, trace-backed correctness, and hardware/shape keyed dispatch.
  - Memory records remote-vs-local LayerNorm conflict and says remote pass must not be generalized to local without proof.
  - Existing notes record `ERR_NVGPUCTRPERM` for remote ncu counter collection.
  - Existing notes record remote speedup above `1.0`, but local speedup remains unverified under the user's current request to avoid local GPU timing.
- Objective restatement:
  1. Use git branches/notes for every materially new kernel tuning attempt; failed attempts are reverted but branches/notes are preserved.
  2. Kernel implementation choice should be selected by hardware and shape, with autotune/runtime keys covering backend/runtime, GPU capability, dtype, `d_model`, rows, block size, warp count, vector width, alias/in-place, and deterministic numeric boundary.
  3. Burn/CubeCL `LocalTuner::execute` mechanics must be understood: key/cache/warmup/measurement boundaries, candidate parameters, accuracy guard limits, and hot-path lookup cost.
  4. CUDA-level reasoning must use reduction design, launch geometry, occupancy/register/shared/shared-memory/coalescing evidence, BF16 numerical analysis, and ncu/alternative profiler evidence where available.
  5. LayerNorm remote/local conflict must be handled by autotune/runtime dispatch or deterministic guard, not hard-coded constants.
  6. Final acceptance target is speedup `>1.0` on both local and `10.100.1.253`, with accuracy drift investigated rather than blindly reverted.
- Next command: inspect current train kernel key structs and recent notes to map checklist evidence to actual artifacts. No GPU timing in this audit.
- Source/key inspection:
  - Current train kernel source contains `AutotuneKey` structs for LayerNorm, lm-head forward/backward, residual add, channel mixer elementwise/backward, WKV7 forward, mix6 forward/backward, learning-rate gate forward/backward, value-residual gate forward/backward, and weight-decay transform.
  - Tuned candidate parameters are represented as named tunables such as `block_*`, `line_size_*`, `row_tile_*`, and reduce-tile combinations, so persistent cache checksum changes when candidate names change.
  - `gated_readout_combine` remains fixed launch geometry. The prior warp32, row-pack, and sumsq-variance attempts are rejected or reverted; the kept change is operation-boundary fusion, not an autotune candidate family.
- LocalTuner mechanism evidence:
  - `2026-05-16-cubecl-local-tuner-findings.md`, `2026-05-16-localtuner-mechanism-skill-update.md`, and the current `kernel-tuning` skill cover cache hit/miss behavior, `TunableSet` init caching, persistent cache checksum, warmup/profiling miss cost, `autotune-checks` limitation, and host-side lookup caveat.
  - Host-side tuner overhead is still only qualitatively bounded as key construction plus lookup/dispatch. No focused current host-overhead profiler branch proves it material or negligible for tiny kernels.
- CUDA profiler evidence:
  - Local historical `ncu` notes cover achieved occupancy, DRAM/SM throughput, register pressure, and spilling for LayerNorm, lm-head loss row, channel mixer matmul, and time-mixer kernels.
  - Remote GB10 `ncu` counter collection is blocked by `ERR_NVGPUCTRPERM`; remote runs can use `nsys` kernel duration, launch geometry, registers/thread, and shared-memory metadata, but cannot honestly claim achieved occupancy, warp execution efficiency, or memory throughput until permissions change.
- Speedup evidence:
  - Remote GB10 has several valid standard compares above `1.0`; latest kept GroupNorm-combine evidence was activation `54/54`, timing `75/76`, `actual_total_ms=72.812`, `baseline_total_ms=175.887`, `speedup=2.42x`.
  - Local regenerated-baseline note records total speedup `2.44x` against a local regenerated baseline, but row-level failures remain and the user currently asked to avoid local GPU timing. The final local acceptance gate is therefore not current.
- LayerNorm conflict evidence:
  - Current LayerNorm code has `LayerNormForwardAutotuneKey` with runtime/hardware/shape/deterministic fields and `deterministic_min_block_size`.
  - Notes record remote GB10 ordinary `256` path passing while local smaller-block paths drifted. Ordered-256 CPU diagnostics showed the intended math can match `1024`, but the device candidate path was rejected/reverted due implementation/boundary risk.
- Missing or weak items:
  - No current remote `ncu` occupancy/warp/memory metrics because of permission.
  - No current local final compare because local GPU should not be used right now.
  - `lm_head/projection` is exposed as an ignored timing row, but not yet canonical in both actual and regenerated baseline contracts. This hides one of the largest `nsys` matmul surfaces from `.time.json` acceptance.
  - There is no project-level production accuracy guard that lets BF16-sensitive candidates prove trace equivalence before winning; current guard is policy-based deterministic filtering plus trace validation after the run.
- Completion decision: not achieved. Continue work without calling `update_goal`.
- Next concrete non-GPU action: open a separate branch to make `lm_head/projection` a comparable canonical timing row only if the baseline generator emits the same row; otherwise keep it ignored. This improves measurement coverage before further kernel implementation attempts.
