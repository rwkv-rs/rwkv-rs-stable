# Goal Gap Audit GB10

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-goal-gap-audit-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits plus retained/reverted kernel tuning notes. This audit owns only this note unless a later entry explicitly opens an implementation branch.
- User constraint: tune/debug first on remote `10.100.1.253`; do not use local GPU timing.
- Prior-note and memory search command already run before opening this note:
  - `rtk rg -n "next.*surface|duplicate|closed|current profile|nsys-current|not repeat|Known GB10|negative result|keep result|matmul|wkv7|mix6|channel_mixer|gated_readout|key_prepare|layer_norm|lm_head|weight_decay" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md /root/.codex/memories/MEMORY.md`
- Matched prior evidence:
  - `kernel-tuning` skill now records duplicate guards for residual add, broad Burn fusion, channel mixer Burn/reference/fusion/LocalTuner bypass, lm-head atomic/row variants, LayerNorm ordered-256/split512, WKV7 segment/factored/direct-value, key-prepare warps, GatedReadout row-pack, weight-decay transform keep, projection/loss broad-design blocker, and remote ncu permission blocker.
  - `2026-05-16-remote-post-target-logit-attribution-gb10.md` identifies the current top remaining surface as broad Cubek/TMA matmul, with custom kernels WKV7 output, channel-mixer relu-square, Mix6, lm-head row, gated-readout combine, key-prepare, residual add, and LayerNorm all already attributed or partially closed.
  - `2026-05-16-residual-matmul-epilogue-gb10.md` rejects using broad `rwkv-nn/fusion` as a residual-matmul epilogue.
  - Memory says remote `10.100.1.253` is the primary tuning target for this project and records the LayerNorm `block_256` GB10 keep result plus local drift caveat.
- Objective checklist for this audit:
  1. Confirm the current source/remote state has hardware/shape keyed autotune where already implemented, especially LayerNorm.
  2. Confirm Burn/CubeCL `LocalTuner` mechanics are documented and no active next step is just re-reading the same cache/warmup behavior.
  3. Confirm current GB10 profile evidence and duplicate-closed surfaces.
  4. Identify one remaining concrete non-duplicate implementation or measurement boundary, or record that the next step is a broader matmul/operator design rather than another small kernel edit.
- Machine/GPU: remote `10.100.1.253`, GB10, CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`.
- Expected keep/revert boundary: keep this audit note as a routing artifact. Do not edit kernels here. If it finds a non-duplicate boundary, open a fresh implementation branch and note before code changes or benchmark/profile runs.
- Next command: inspect the live source and notes for exact current state: LayerNorm key fields, LocalTuner notes, and latest remote nsys/autotune evidence.

## Audit Findings

- LayerNorm source state:
  - `LayerNormForwardAutotuneKey` includes runtime, dtype, `d_model`, rows, total elements, hardware fingerprint, in-place flag, deterministic flag, and deterministic minimum block size.
  - Candidate `block_size` is represented in tunable names `block_{block_size}` over `[64,128,256,512,768,1024]`.
  - GB10 BF16 `D=768, rows=8192` is explicitly allowed to use deterministic minimum block size `256`; other BF16 paths fall back to the next-power-of-two deterministic boundary.
- LocalTuner mechanism state:
  - `2026-05-16-cubecl-local-tuner-findings.md` already documents `LocalTuner::execute` key construction, cache hit/miss behavior, persistent checksum, warmup/profile sampling, and the lack of project-level accuracy guard.
  - Re-reading LocalTuner would be duplicate work unless a branch changes the measured host boundary or the candidate set.
- Current GB10 profile state:
  - `2026-05-16-remote-post-target-logit-attribution-gb10.md` records the current top kernels and autotune selections. The selected candidates are not stale-cache artifacts.
  - The largest remaining surface is broad Cubek/TMA matmul, especially `lm_head/projection`, but existing notes classify projection+loss fusion as a broader operator/API problem requiring GEMM-quality tiling/backward design.
  - The largest project-owned custom kernel is still WKV7 output (`~22ms / 36` launches in the current GB10 nsys profile), but row-tile-only, shared-value direct load, output factorization, and f32 segment-recompute attempts are all rejected.
  - Mix6, channel-mixer relu-square, key-prepare, gated-readout combine, lm-head row, residual add, LayerNorm, and value-residual gate all have current kept/rejected evidence. None of them currently exposes a non-duplicate small candidate.
- Completion status against the explicit objective:
  - Branch/worktree + notes discipline: active and now encoded in `kernel-tuning` skill.
  - Hardware/shape keyed LayerNorm: implemented in source for the current CubeCL path and validated by prior remote/local notes.
  - Burn/CubeCL autotune mechanism: documented; no missing basic mechanism read remains.
  - CUDA bottom-up profiling: `ncu` is blocked on remote by `RmProfilingAdminOnly=1`; current usable evidence is `nsys` launch/register/shared-memory metadata plus compare. Local `ncu` evidence exists for earlier local attempts, but the user asked to avoid local GPU for now.
  - Both local and remote `speedup > 1.0`: remote is currently satisfied in prior standard compare notes; local remains unresolved and should not be tested while the user reserves the local GPU.
- Next concrete non-duplicate track: WKV7 algorithm-boundary design. It must avoid the rejected f32 global segment tensors and high-register scan, and it must not repeat row-tile/shared-value/output-factor tweaks. A separate branch should inspect whether a materially different state-scan/time-split design can reduce WKV7 output time without huge global intermediate tensors.
- Decision: close this audit as routing evidence. Open a fresh WKV7 algorithm-design branch before inspecting or editing that boundary.
- Follow-up correction: the fresh WKV7 state/time split branch immediately found that `2026-05-16-wkv7-remote-launch-design.md` had already done the same source/recurrence inspection, and `2026-05-16-wkv7-segment-recompute-gb10.md` had already implemented the mathematically safe transform/scan/recompute candidate as a negative. Treat WKV7 state/time split as duplicate unless a future branch names a design that avoids both global f32 segment tensors and row-tile/shared-value/output-factor families.
