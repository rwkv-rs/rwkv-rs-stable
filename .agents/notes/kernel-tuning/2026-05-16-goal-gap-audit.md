# Goal Gap Audit

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-goal-gap-audit-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this branch inherits broad unrelated workspace changes plus retained kernel-tuning source from prior branches. This attempt is audit/evidence only; it must not edit kernels or run local GPU.
- Prior-note search command:
  - `rg -n "completion-audit|current-goal|achieved|local.*speedup|remote.*clean|speedup>1|speedup > 1|objective|missing|not achieved|WKV7|lm_head projection|timing contract" .agents/notes/kernel-tuning /root/.codex/memories/MEMORY.md -S`
  - `rg -n "mix6|WKV7|row_tile|key_prepare|gated_readout|lm_head projection|ncu|occupancy|LocalTuner|value_residual_gate" .agents/notes/kernel-tuning crates/rwkv-nn/src/kernels/train -S`
- Matched prior evidence:
  - Remote `10.100.1.253` is the current primary timing source; local GPU is intentionally reserved by the user and local timing must not be used while it may be inaccurate.
  - LayerNorm runtime dispatch already records the GB10-vs-local conflict: GB10 admits `block_256`; local deterministic policy keeps `1024` because smaller blocks drift.
  - Local overall speedup remains unachieved or not freshly verified under an idle local GPU.
  - Remote standard compare currently stays above `1.0`, but can still show the known marginal `cell_0000/channel_mixer` row.
  - Duplicate-closed directions include residual-add/Burn-add A/B, channel-mixer forced Cube/Burn-reference/fusion, LocalTuner bypass, LayerNorm ordered-256, WKV7 forced `row_tile=16`, lm-head row variants, GatedReadout warp32/row-pack, and key-prepare warps-per-cube.
- Machine/GPU: audit only. If new runtime evidence is needed, use remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`; do not use local GPU.
- Shape/dtype scope: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, heads `12`, head size `64`, regenerated remote baseline under `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Objective restatement as concrete deliverables:
  1. Use git branch/worktree discipline for each material tuning attempt; failed code is reverted but branch/note is retained.
  2. Kernel implementation dispatch/autotune keys must include backend/runtime, hardware fingerprint, dtype, `d_model`, rows, candidate parameters such as block size / warps / vector width, alias/in-place, and deterministic numeric boundary where applicable.
  3. LayerNorm must not be a hard-coded constant; GB10 and local should be distinguished by hardware/shape/determinism policy with trace-backed correctness.
  4. Burn/CubeCL `LocalTuner::execute` behavior must be understood: key/cache/warmup/measurement boundary, candidate parameters, and accuracy-guard limitations.
  5. CUDA tuning must use profiler evidence where possible: launch geometry, occupancy/register/shared-memory/memory-coalescing reasoning, and BF16 numerical drift analysis.
  6. End state requires `speedup > 1.0` on both local and `10.100.1.253`, with activation passing.
- Audit commands planned:
  - Inspect existing audit/contract notes and current source for LayerNorm, autotune-key fields, timing contract rows, and duplicate-guarded kernel attempts.
  - Do not run local GPU. Do not run remote benchmark until the checklist identifies a new non-duplicate evidence gap.
- Expected decision boundary: if any deliverable is missing or weakly verified, keep the goal open and open a fresh implementation/profiling branch for the next concrete gap. Do not mark the goal complete from remote-only success.

## Audit Continuation

- Current user constraint: debug first on remote `10.100.1.253`; do not use the local GPU for timing because the user is running other work there and local timing may be inaccurate.
- Confirmed duplicate/closed implementation boundaries:
  - `key_prepare` warps-per-cube tuner was already tested on GB10, selected `warps_per_cube=2`, doubled block count, and slightly worsened targeted `nsys` time; code was reverted to fixed `HEAD64_WARPS_PER_CUBE=4`.
  - WKV7 row-tile-only tuning is closed: remote selected `row_tile=64`, forced `row_tile=16` was a negative result, and a real time/chunk split would require state handoff or scan-style composition rather than a small launch-geometry change.
  - GatedReadout row-pack is closed: targeted combine improved slightly, but end-to-end speedup did not beat the retained 64-thread implementation, so it was reverted.
- Decision: keep the goal open. The next valid GPU action is a fresh remote-current profiling branch on `10.100.1.253`, using the current post-revert live source/binary boundary to rank remaining CUDA surfaces before choosing another implementation branch.
