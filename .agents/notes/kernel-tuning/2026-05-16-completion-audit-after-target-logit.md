# Completion Audit After Target Logit

- Date: 2026-05-16 17:12 +0800.
- Branch/worktree: `kernel-tuning-completion-audit-after-target-logit-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits plus retained/reverted kernel tuning changes. This attempt owns only this audit note unless later entries explicitly open implementation or profiling branches.
- Prior-note and memory search commands:
  - `rg -n "completion|audit|current-goal|LocalTuner|autotune|ordered-256|LayerNorm|10\\.100\\.1\\.253|GB10|ncu|ERR_NVGPUCTRPERM|speedup=|activation_summary|timing_summary" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md /root/.codex/memories/MEMORY.md -S`
  - `rg -n "runtime|dtype|d_model|rows|block|warps|line_size|row_tile|deterministic|is_in_place|hardware|num_streaming|load_width|plane_size" crates/rwkv-nn/src/kernels/train -S`
- Matched prior evidence:
  - Kernel tuning skill now requires branch+note attempts, duplicate guards, remote GB10 primary timing, LocalTuner cache-hit/miss semantics, and hardware/shape/deterministic key dimensions.
  - LayerNorm ordered-256 and split-tail smaller reductions are recorded local accuracy failures; remote GB10 verified `block_256`, while local deterministic policy remains `1024`.
  - Remote `10.100.1.253` ncu counter collection is blocked by `ERR_NVGPUCTRPERM`; nsys can provide launch/register/shared-memory metadata but not achieved occupancy or warp/memory-throughput counters.
  - Latest remote kept implementation branch `2026-05-16-lm-head-target-logit-gb10.md` records row-kernel improvement from `13.879ms / 3` to `12.851ms / 3`, activation `54/54`, and total speedup `2.34x` with one known channel-mixer edge row.
- Machine/GPU context: audit is local read-only unless a later entry records a command. Remote acceptance source remains `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`. The local GPU remains unavailable per user instruction and must not be used for acceptance in this audit.
- Objective restated as concrete deliverables:
  1. Kernel tuning attempts use fresh git branches/worktrees plus prewritten notes; failed attempts are reverted but branches/notes remain.
  2. Kernel implementation selection is hardware/shape aware, with autotune/runtime-dispatch keys covering backend/runtime, GPU arch/capability, dtype, `d_model`, rows, block size, warps, vector width, in-place/alias, and deterministic numeric boundary.
  3. LayerNorm no longer hard-codes one block size; GB10 can select/pass `256`, local deterministic path avoids the known drift.
  4. Burn/CubeCL `LocalTuner::execute` mechanism is understood and reflected in rules: key/cache/warmup/measurement, candidate parameters, accuracy-guard limits, and hot-path lookup cost.
  5. CUDA lower-level work investigates warp/block config, occupancy/register/shared/shared-memory/coalescing, BF16 reduction error, and device differences; poor results are diagnosed, not blindly reverted.
  6. Acceptance target is speedup `>1.0` on both local and `10.100.1.253`, with precision issues investigated for implementation bugs before reverting.
- Expected audit boundary: if every deliverable has current evidence, call `update_goal`. If any item is missing or weakly verified, keep the goal open and identify the next concrete non-duplicate action.
- Next command: inspect current notes/source evidence for each checklist item; no GPU commands in this audit.
