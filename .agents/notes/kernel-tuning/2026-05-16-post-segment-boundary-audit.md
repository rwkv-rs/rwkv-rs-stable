# Post Segment Boundary Audit

- Date: 2026-05-16 17:56 +0800.
- Branch/worktree: `kernel-tuning-post-segment-boundary-audit-20260516` in the existing dirty checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits plus retained/reverted kernel tuning work. This attempt is read-only boundary selection unless a later entry explicitly opens a separate implementation branch.
- Prior-note/source search command:
  - `rg -n "completion-audit|current-goal|post-segment|WKV7|segment_recompute|next boundary|duplicate-closed|speedup>1|local.*speedup|remote.*speedup|ncu|achieved occupancy|warp execution|memory throughput" .agents/notes/kernel-tuning /root/.codex/memories/MEMORY.md -S`
- Matched prior evidence:
  - `2026-05-16-wkv7-segment-recompute-gb10.md` closed segment transform/scan/recompute: activation passed, but transform + scan + recompute CUDA time was about `13.2x` the current row-tile WKV7 output.
  - `2026-05-16-current-goal-audit.md` says the active goal is not complete: local final acceptance is missing, and remote `ncu` counters are blocked by `ERR_NVGPUCTRPERM`.
  - `2026-05-16-remote-current-profile-gb10-post-contract.md` classifies remaining surfaces: Cubek matmul dominates, WKV7 is largest project-owned custom kernel, and many smaller direct attempts are duplicate-closed.
  - `2026-05-16-lm-head-projection-loss-fusion-analysis.md` rejects projection+loss fusion as too broad because it would need to compete with optimized Cubek/TMA matmul and cross the autograd/operator boundary.
  - `2026-05-16-forward-slowgroup-current-ncu.md` and `2026-05-16-local-ncu-forward-slow-kernels.md` contain local ncu metrics for LayerNorm, channel-mixer, and lm-head loss, but those are not current remote counters.
  - `2026-05-16-remote-channel-mixer-edge-ncu.md` records that even remote `LaunchStats` ncu is blocked by counter permissions.
- Machine/GPU: no local GPU commands in this branch. Remote source of truth remains `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, heads `12`, head size `64`.
- Changed boundary: this is not another kernel code attempt. It is an evidence audit after rejecting WKV7 segment recompute, to choose the next non-duplicate implementation boundary or prove that only broad operator-level work remains.
- Candidate parameters: none yet. Audit dimensions are explicit objective requirements: branch/notes discipline, hardware/shape autotune keys, Burn/CubeCL tuner behavior, CUDA-level evidence, accuracy-root-cause handling, remote and local speedup gates.
- Expected keep/revert boundary: keep this note as decision evidence. If a non-duplicate small implementation boundary exists, create a fresh implementation branch before editing. If all remaining surfaces are duplicate-closed or require broad Cubek/operator work, record that and do not edit kernels in this branch.
- Next command: inspect current source/key coverage and the latest notes for LayerNorm, WKV7, lm-head, channel-mixer, mix6, gated-readout, key-prepare, value-residual, residual-add, and remote/local acceptance status.

## Continuation After Remote-Only User Constraint

- User constraint: continue debugging first on `10.100.1.253`; do not run local GPU timing because the local GPU is reserved for other work and may give inaccurate timing.
- Rechecked duplicate boundaries:
  - `key_prepare` warps-per-cube is closed: it selected `warps_per_cube=2` on GB10, changed launch to `block=(64,1,1)`, and slightly worsened targeted `nsys` time versus fixed `block=(128,1,1)`.
  - `lm_head_l2wrap_ce` row-loss postprocessing has one kept GB10 change: direct target-logit loading reduced row-kernel registers and improved the row kernel in `nsys`. Other row-kernel variants (`atomic`, `online-softmax`, `prune`, local-only target-logit boundary) are closed or superseded.
  - `GatedReadout` GroupNorm+combine is the kept positive boundary; warp32, row-pack, and sumsq-variance variants are closed.
  - WKV7 row-tile, shared-lanes, and segment transform/scan/recompute are closed. The segment path was activation-clean but architecturally too slow because transform+scan+recompute took about `13.2x` the current row-tile kernel and the scan kernel used `255` registers/thread.
  - Channel-mixer matmul/activation fusion and lm-head projection+loss fusion remain broad Cubek/TMA or new-operator work, not a small kernel tweak.
- Current honest next step: remote provenance check only. Confirm that the remote mirror on `10.100.1.253` still has the kept target-logit row kernel, no WKV7 segment code, WKV7 `row_tile_{16,32,64}` only, and the latest clean compare/profile logs. Do not run a new benchmark unless the provenance check shows the remote state drifted.
- Next command: remote read-only check on `10.100.1.253` for GPU state, source snippets, and latest compare/profile log timestamps.

## Remote Provenance Check

- Host/path: `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Command: read-only `ssh` check of `hostname`, `nvidia-smi`, source snippets for lm-head target-logit and WKV7 candidates, and latest target-logit / post-target / WKV7 log timestamps.
- Result:
  - Host is `spark-35ac`.
  - GPU is `NVIDIA GB10`, compute capability `12.1`, utilization `0%`.
  - Remote source has the kept direct target-logit load in `lm_head_l2wrap_ce/kernel.rs`: `target_logit = logits[row_start + target]` under `UNIT_POS == 0`.
  - Remote WKV7 source has only `PRETRAIN_OUTPUT_ROW_TILE_CANDIDATES = [16, 32, 64]`; no `segment_recompute`, `segment_transform`, `segment_scan`, or `shared_lanes` identifiers are present.
  - Latest relevant logs include clean post-target-logit attribution (`remote-post-target-logit-attribution-compare.log`) and the post-segment WKV7 revert compare (`remote-wkv7-segment-recompute-reverted-compare.log`).
- Decision: remote mirror is in the expected kept/reverted state. Do not rerun the same compare/profile just to refresh numbers. The next valid implementation work must be a materially new boundary, most likely broad WKV7 state-scan redesign or Cubek/TMA/lm-head projection operator design, not another small candidate retry.
- Next edit: update `.agents/skills/kernel-tuning/SKILL.md` with a durable duplicate guard for the rejected WKV7 segment transform/scan/recompute path, so future runs stop before repeating the same GB10 experiment.

## Skill Guard Update

- Edited `.agents/skills/kernel-tuning/SKILL.md`.
- Added a known GB10 negative result for WKV7 segment transform/scan/recompute:
  - activation passed on `10.100.1.253`,
  - transform+scan+recompute CUDA time was about `13.2x` the current `row_tile_64` path,
  - scan kernel used `255` registers/thread,
  - do not retry this exact design for `B=16,T=512,D=768,head_size=64` unless the algorithm removes global f32 transform tensors and the high-register segment scan.
- Keep/revert state: keep this skill update as duplicate-experiment protection.
- Next command: sync only `.agents/skills/kernel-tuning/SKILL.md` and this note to `10.100.1.253` so the remote mirror carries the same duplicate guard and provenance note.

## Remote Skill Sync

- Command: scoped `rsync -azR` of `.agents/skills/kernel-tuning/SKILL.md` and this note to `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable/`.
- Result: sync exited `0`.
- Note: this was a docs/provenance sync only; no GPU benchmark or profiler command was run.
