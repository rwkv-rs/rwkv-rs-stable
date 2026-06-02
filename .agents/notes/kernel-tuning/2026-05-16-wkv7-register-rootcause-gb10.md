# WKV7 Register Root Cause GB10

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-wkv7-register-rootcause-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout already carries broad unrelated workspace edits, generated traces, untracked notes, and untracked skill files. This attempt owns only this WKV7 root-cause note unless a later entry explicitly names a source or skill edit.
- User constraint: continue remote-first on `10.100.1.253`; do not use local GPU timing while the local GPU may be used elsewhere.
- Scope: source-level root-cause audit for current WKV7 register pressure and launch underfill. This branch does not run GPU workloads, clear caches, or edit kernel source.
- Prior-note search commands already run:
  - `rg -n "WKV7|wkv7|row_tile|register|occupancy|shared_lanes|segment|state-scan|state scan|time split|direct-value|output_factor|output-factor|lowrank|low-rank" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md -S`
  - `rg -n "wkv7_pretrain_forward_output_kernel|wkv7_pretrain_forward_kernel|wkv7_state_forward_kernel|wkv7_backward_kernel|for .*time|while .*time|state|SharedMemory|RuntimeCell|row_tile|value_shared|replacement|weight_decay|receptance" crates/rwkv-nn/src/kernels/train/time_mixer/wkv7 -g '*.rs'`
- Matched prior evidence:
  - Current GB10 R9 `nsys` evidence records WKV7 output at roughly `23ms / 36`, launch `grid=(12,16,1)`, `block=(64,1,1)`, `regs/thread=127`, dynamic shared memory `1536`.
  - GB10 device limits plus nsys metadata estimate theoretical occupancy around `33.3%` and effective wave occupancy around `16.7%`, with only `192` blocks per launch (`4 blocks/SM`) under `B=16,H=12,row_tile=64`.
  - Closed WKV7 families: forced smaller row tiles, shared-lanes/shared-state reductions, direct value load, output factorization, dense f32 segment transform/scan/recompute, and low-rank segment designs.
  - Skill guard requires any future WKV7 branch to name a materially different algorithm that increases parallel work or reduces per-row register state without repeating global f32 transform tensors, repeated input loads, or high-synchronization shared-state reductions.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, heads `12`, head size `64`, rows `8192`.
- Hypothesis: the current WKV7 output hotspot is not primarily a missing row-tile candidate. It is rooted in the per-row recurrent state representation: each active output row keeps a `head_size=64` f32 state live across the serial time loop, so the launch is both grid-underfilled and register-constrained. A useful next algorithm must change the state representation, time decomposition, or output contract.
- Expected keep/revert boundary: keep this note as design evidence. If the audit only restates closed row-tile/segment ideas, do not open a source branch. If it identifies a genuinely new source boundary, open a separate implementation branch and prewrite its note before editing.
- Next command: inspect the current WKV7 kernel body around the output recurrence, the saved-state recurrence, and the backward recurrence to map register state and memory traffic to source constructs.

## Source Inspection Result

- Command: read `crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/{kernel.rs,forward.rs,backward.rs}` and re-read the prior WKV7 notes/skill duplicate guards.
- Output fast path: `wkv7_pretrain_forward_output_kernel` maps `grid=(num_heads,batch,row_tiles)` and `block=(row_tile)`. For the active GB10 candidate `row_tile=64`, the fixture launches `12 * 16 * 1 = 192` cubes. Each active unit owns one state row and allocates `Array::<f32>::new(head_size)`, so for `head_size=64` the compiler must keep a full f32 row state live across the entire `while time < context_len` recurrence. This is the direct source construct behind high registers/thread and underfilled waves.
- Shared-input behavior: each timestep stages six `head_size` vectors in shared memory: receptance, decay, replacement key, value, normalized removal key, and replacement. This makes `row_tile=64` attractive because the six vectors are loaded once per `(batch, head, time)` cube; smaller row tiles create more cubes and repeat these loads. This matches the recorded row-tile/shared-lanes/direct-value/output-factor negatives.
- Saved-state forward path: `wkv7_pretrain_forward_kernel` uses one cube per `(batch, head)` with `block=head_size`, writes `snapshots[B,H,T/chunk,HS,HS]` and `state_replacement[B,T,H,HS]`, and repeatedly reloads previous snapshots inside the time loop. This path is required for backward state but is not a faster forward-output candidate because it trades register state for global snapshot traffic.
- Statepass path: `wkv7_state_forward_kernel` has the same snapshot/state-replacement traffic as saved pretrain and additionally handles `initial_state`/`next_state`. It is a fixed semantic boundary rather than an output-only replacement for the pretrain fast path.
- Backward path: `launch_wkv7_backward` allocates four `[B,H,HS,HS]` scratch tensors plus six sequence-gradient tensors and launches `wkv7_backward_kernel` with `grid=(H,B,1)`, `block=HS`. The kernel keeps row/column state-gradient views live across the reverse time scan and repeatedly touches snapshots/state replacement. Changing this path safely would be a separate backward algorithm project, not a small forward-output row-tile follow-up.
- Root-cause conclusion: the current output bottleneck is a real algorithm/contract tradeoff. The row-tile fast path minimizes global state traffic by holding per-row state in registers, but that causes high register pressure and only 192 blocks per launch for the current shape. The saved/state/backward variants expose more global-memory traffic and do not provide a simple low-register substitute.
- Decision: do not open another WKV7 implementation branch from this audit. The materially new WKV7 directions would need to change the recurrence contract itself, such as an exact state-composition algorithm that avoids dense f32 transform tensors and avoids the already measured slow recompute lower bound. The current audit found no such small candidate.
- Keep/revert state: no source edits in this branch. Keep this note as WKV7 root-cause evidence.
- Next command: sync this note to `10.100.1.253`, then run a remote read-only provenance check for the current mirror, GPU id/utilization, latest compare/profile logs, and current top kernel surfaces. Do not run a new GPU benchmark from this WKV7 branch unless the provenance check shows missing or stale current evidence.

## Remote Provenance Check

- Command: scoped `rsync -azR` of this note to `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable/`, then remote read-only shell check for host, GPU, note presence, and latest compare/profile artifacts.
- Host/GPU: `spark-35ac`, `NVIDIA GB10`, compute capability `12.1`, utilization `0%`.
- Remote note check: `.agents/notes/kernel-tuning/2026-05-16-wkv7-register-rootcause-gb10.md` exists with `32` lines.
- Latest current profile artifact: `target/rwkv-test/nsys-current-r9-gb10.sqlite`, mtime `2026-05-16 08:36`.
- Latest compare artifacts include `remote-residual-matmul-fusion-compare.log`, `remote-wkv7-direct-value-post-revert-compare.log`, and `remote-current-after-gb10-results-compare.log`.
- Decision: provenance is sufficient for a read-only current-surface query. No new benchmark is needed before querying the existing sqlite.
- Next command: query `target/rwkv-test/nsys-current-r9-gb10.sqlite` on `10.100.1.253` for aggregate kernel time/count/launch metadata.

## Current R9 Kernel Surface Query

- Command: remote Python/sqlite query against `target/rwkv-test/nsys-current-r9-gb10.sqlite`, grouping `CUPTI_ACTIVITY_KIND_KERNEL` by demangled name plus launch geometry.
- Top surfaces:
  - `matmul_entry_lhs_bf16...`: largest groups are `46.174ms / 3`, `26.447ms / 144`, `21.327ms / 36`, and `18.860ms / 36`.
  - `wkv7_pretrain_forward_output_kernel_f_bf16`: `23.171ms / 36`, `grid=(12,16,1)`, `block=(64,1,1)`, `regs=127`, dynamic shared memory `1536`.
  - `mix6_forward_kernel_f__n_1`: `14.484ms / 36`, `regs=32`, `grid=(24576,1,1)`, `block=(32,8,1)`.
  - `channel_mixer_relu_square_forward_kernel_f__n_2`: `14.220ms / 36`, `regs=16`, `grid=(49152,1,1)`, `block=(32,8,1)`.
  - `lm_head_l2wrap_ce_forward_row_kernel_f__i_i32`: `13.470ms / 3`, `regs=40`, `block=(1024,1,1)`.
  - `gated_readout_combine_forward_kernel_f_`: `12.164ms / 36`, `regs=25`, `grid=(8192,12,1)`, `block=(64,1,1)`.
  - `key_prepare_forward_64_kernel_f_`: `9.401ms / 36`, `regs=24`, `block=(128,1,1)`.
  - `kernel_binop_c_bf16_n_8`: `8.477ms / 72`.
  - `value_residual_gate_forward_kernel_f__n_2`: `6.506ms / 33`.
  - `layer_norm_forward_kernel_f_`: `6.205ms / 78`, `block=(256,1,1)`.
- Interpretation:
  - WKV7 is still the largest project-owned custom kernel, but this branch found no non-duplicate small implementation boundary.
  - The largest overall time is still Burn/Cubek matmul. Prior notes already reject naive project-local projection/loss fusion and global Burn fusion as substitutes for a real Cubek/TMA epilogue.
  - Several remaining project-owned kernels are already keyed `LocalTuner` outputs; the next decision should check their current autotune selections and compare status before opening a new branch.
- Next command: read the latest remote compare summaries and current autotune logs for the top custom train kernels without changing source, cache, or benchmark state.

## Remote Compare Summary Check

- Command: remote read of `target/rwkv-test/remote-residual-matmul-fusion-compare.log`, `remote-wkv7-direct-value-post-revert-compare.log`, `remote-current-after-gb10-results-compare.log`, and `remote-projection-r9-compare.log`.
- `remote-current-after-gb10-results-compare.log`: activation passed `54/54`; timing compared `76`, passed `75`, failed `1`, ignored `1`; `actual_total_ms=72.722`, `baseline_total_ms=175.887`, `speedup=2.42x`.
- `remote-wkv7-direct-value-post-revert-compare.log`: activation passed `54/54`; timing `actual_total_ms=73.153`, `baseline_total_ms=175.887`, `speedup=2.40x`.
- `remote-residual-matmul-fusion-compare.log`: activation passed `54/54`; timing `actual_total_ms=79.595`, `baseline_total_ms=175.887`, `speedup=2.21x`. This is slower than the current-after results and matches the known negative global-fusion evidence.
- `remote-projection-r9-compare.log`: activation passed `54/54`; timing `actual_total_ms=87.891`, `baseline_total_ms=130.484`, `speedup=1.48x` under the projection timing contract.
- Autotune log parsing attempt: the initial `target/autotune/**/*.json.log` scan did not print train-kernel entries, so the next read-only command should first list the remote autotune cache/log layout before drawing conclusions.
- Decision: the remote acceptance requirement is currently satisfied on GB10. This branch should not open a new WKV7 source attempt. The next useful tuning work must target a top surface that is not already duplicate-guarded negative.
- Next command: list remote `target/autotune` files and parse any current tunable cache/log files for the top custom kernels.

## Remote Autotune Cache Check

- Command: remote `find target/autotune -maxdepth 4 -type f`, then JSON-log parsing for `rwkv_nn-*` train-kernel autotune entries.
- Current train-kernel autotune files exist under `target/autotune/0.10.0/device-0-0-cuda/`.
- Selected/fastest entries:
  - WKV7 output: `row_tile_64`, median about `606.673us`; `row_tile_32` about `945.378us`, `row_tile_16` about `1762.514us`.
  - LayerNorm forward: `block_256`, median about `92.915us` to `102.881us`; `block_1024` about `281us` to `289us` on GB10.
  - LM head l2wrap CE forward: `block_1024`, median about `4.24ms` to `4.60ms`; smaller blocks are slower.
  - Mix6 forward: `line_size_1`, median about `408us` to `412us`; other valid line sizes are close but slower in the cache.
  - Channel mixer relu-square: `line_size_2`, median about `428us` to `434us`.
  - Channel mixer mix: current entries select `line_size_4` or `line_size_8` depending on key/checksum; both are close.
  - Value residual gate: `line_size_2`, median about `220.7us`.
  - Weight decay transform: current entries select around `line_size_4`/`line_size_8`; both are close.
- Interpretation: the current top custom kernels are not obviously stale-candidate failures. WKV7 has a real root-cause bottleneck but no non-duplicate small candidate; LayerNorm/lm_head have clear selected block sizes; mix6/channel/value/weight-decay are near-tie line-size choices rather than easy forced-constant wins.
- Decision: expand this branch scope only to update the project kernel-tuning skill with the WKV7 source-level duplicate guard, then close this WKV7 root-cause attempt as evidence. No source kernel edit is justified by the read-only checks.
- Next edit: update `.agents/skills/kernel-tuning/SKILL.md` to record the source-level WKV7 root cause: per-row `Array<f32>(head_size)` state across the full time loop, `row_tile=64` minimizing shared input reloads, saved/state/backward variants trading registers for global snapshot/scratch traffic, and future WKV7 candidates needing a genuinely new recurrence-contract boundary.

## Skill Update

- Edited `.agents/skills/kernel-tuning/SKILL.md`.
- Added a durable WKV7 source-level root-cause guard:
  - fast output kernel keeps one `Array<f32>(head_size)` state per active row across the full serial time loop;
  - current GB10 shape with `row_tile=64` launches only `12 * 16 = 192` cubes;
  - smaller row tiles repeat the six shared input-vector loads;
  - saved/state/backward variants trade register state for global snapshot/state-replacement/scratch traffic;
  - future WKV7 branches must change the recurrence contract itself and avoid the closed row-tile, shared-lanes, direct-value, output-factor, dense segment, and low-rank segment failure modes.
- Next command: sync this note and `.agents/skills/kernel-tuning/SKILL.md` to `10.100.1.253`, then verify the remote skill contains the new guard.

## Remote Skill Sync

- Command: scoped `rsync -azR` of `.agents/skills/kernel-tuning/SKILL.md` and this note to `10.100.1.253`, then remote `grep` for the new guard.
- Remote verification: `.agents/skills/kernel-tuning/SKILL.md:48` contains `Known GB10 WKV7 source-level root cause`.
- Decision: close this WKV7 root-cause attempt. No WKV7 source edits were made. The current workspace keeps only the note and skill update from this branch.
