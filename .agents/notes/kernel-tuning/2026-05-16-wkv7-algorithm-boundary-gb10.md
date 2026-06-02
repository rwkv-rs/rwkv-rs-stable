# WKV7 Algorithm Boundary GB10

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-wkv7-algorithm-boundary-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits and accumulated kernel tuning changes. This attempt owns only this note unless a later entry explicitly opens a code prototype.
- User constraint: debug/tune first on remote `10.100.1.253`; do not use local GPU timing while the local GPU may be busy.
- Prior-note search commands already run in the current session:
  - `rg -n "WKV7|wkv7|state-scan|state scan|segment|row_tile|shared-lanes|lowrank|recompute|register|regs|time-split|time split|10\\.100\\.1\\.253" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md crates/rwkv-nn/src/kernels/train/time_mixer/wkv7 -S`
  - `rg -n "Next valid|Next action|next implementation|next .*branch|unresolved|gap|remaining|do not open|only design-level|larger remaining|dominating|current honest" .agents/notes/kernel-tuning/2026-05-16-*.md`
- Matched prior evidence:
  - WKV7 row-tile-only tuning is closed; `row_tile=16` was negative and current GB10 policy uses the existing tuned row-tile family.
  - Shared-state-lanes and low-rank/state-scan design attempts are recorded and must not be repeated unchanged.
  - Segment recompute with global f32 `P/Q` segment tensors and `segment_len=16` passed activation on `10.100.1.253`, but was about `13.2x` slower than the current `row_tile_64` WKV7 output path and used `255` registers/thread in the scan kernel.
  - Latest current-profile notes say the largest project-owned non-matmul surface is WKV7, but a viable improvement must be a materially different algorithm, not another launch-geometry retry.
  - `lm_head/projection` remains the largest overall surface, but current Burn/CubeCL/Cubek public APIs do not expose a project-owned TMA epilogue hook; naive projection-loss fusion is blocked by design.
- Machine/GPU: remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`; source/design inspection can happen locally, but no local GPU run is valid for this attempt.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, `num_heads=12`, `head_size=64`.
- Kernel/stage: `time_mixer/wkv7` forward output.
- Changed boundary:
  - This is not row-tile, shared-lanes, or the rejected segment-recompute implementation.
  - The read-only question is whether WKV7 has a different state-handoff/scan formulation that avoids global f32 transform tensors, avoids the high-register scan kernel, and preserves the exact trace output boundary.
- Expected keep/revert boundary:
  - Keep this note as design evidence.
  - Do not edit WKV7 code unless the source/math inspection identifies a concrete implementation whose state representation and validation boundary differ from the failed segment-recompute design.
  - If the only plausible path repeats global segment transforms or needs broad algorithmic/backward redesign, close as design-blocked and do not run a benchmark.
- Next command: inspect current WKV7 source and the prior WKV7 design/negative notes to identify the first state recurrence and the exact reason the previous scan formulation became too slow.

## Source And Prior Evidence

- Source inspection result:
  - Current WKV7 output path is `wkv7_pretrain_output` with `row_tile_{16,32,64}` candidates.
  - GB10 selects `row_tile_64`, which maps one cube to one `(batch, head)` pair and one unit to one state/output row.
  - Each active unit stores its `64`-column row state in registers and serially scans all `T=512` steps.
  - At each step the kernel first computes `state_replacement = dot(state_row, removal_key_normalized)`, then updates every state column and accumulates output as `sum(updated_col * receptance_col)`.
- Prior negative evidence:
  - `shared_lanes_{4,8}` was activation-safe but much slower than `row_tile_64`, so row/column lane packing is closed.
  - Dense `segment_recompute_16` was activation-safe but about `13.2x` slower than `row_tile_64`, dominated by global f32 transform tensors and a high-register scan kernel.
  - Low-rank segment summaries are design-negative because exact output generation still needs recompute or dense correction work; even the measured recompute kernel alone was slower than current row-tile.
- New non-duplicate candidate:
  - Keep the current one-cube-per `(batch, head)` and `row_tile_64` state update.
  - Algebraically rewrite output:
    - current: `output = sum_c updated_c * receptance_c`, where `updated_c = state_c * decay_c + state_replacement * replacement_c + value_row * replacement_key_c`.
    - factored: `output = sum_c(state_c * decay_c * receptance_c) + state_replacement * sum_c(replacement_c * receptance_c) + value_row * sum_c(replacement_key_c * receptance_c)`.
  - The two latter dot products are shared by all 64 rows for the same `(batch, time, head)`, so compute them once per timestep per cube with block reductions instead of repeating their contribution inside every row.
  - This is not a row-tile/shared-lanes/segment-scan retry; it changes only output arithmetic inside the current row-tile implementation family.
- Accuracy risk:
  - The state update order is unchanged, but output accumulation order changes. For BF16 trace outputs this may be acceptable or may drift; remote activation decides admission.
- Expected implementation boundary:
  - Add one extra WKV7 output candidate named `row_tile_64_output_factored`.
  - Valid only for `head_size=64` and hardware allowing at least `64` units per cube.
  - Keep existing `row_tile_{16,32,64}` candidates and let `LocalTuner` choose.
  - Revert if remote activation drifts or if autotune/`nsys` shows the factored candidate slower than `row_tile_64`.
- Next edit: add the factored output kernel and candidate in `wkv7/{kernel.rs,forward.rs}`, then run local non-GPU format/compile before syncing to `10.100.1.253`.

## Implementation Edit 1

- Edited `crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/kernel.rs`.
  - Added block/warp f32 reduction helpers.
  - Added `wkv7_pretrain_forward_output_factored_kernel`.
  - The new kernel keeps one cube per `(batch, head)`, computes shared `replacement·receptance` and `replacement_key·receptance` per timestep, preserves the original state update loop, and writes the factored output expression.
- Edited `crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/forward.rs`.
  - Added candidate `row_tile_64_output_factored` to the existing WKV7 output `LocalTuner`.
  - Candidate validity requires `head_size == 64` and at least `64` units per cube.
  - Existing `row_tile_{16,32,64}` candidates remain unchanged.
- Local non-GPU check:
  - Command: `rustup run nightly rustfmt crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/forward.rs crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/kernel.rs && cargo check -p rwkv-nn --features cuda,fusion`.
  - Result: passed, but emitted one `unused_mut` warning for `shared_reduce`.
- Next edit: remove the unnecessary `mut` from `shared_reduce`, rerun the same non-GPU check, then sync to `10.100.1.253` if clean.

## Local Non-GPU Check 2

- Code fix: removed the unnecessary `mut` from `shared_reduce` in the factored WKV7 output kernel.
- Command: `rustup run nightly rustfmt crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/forward.rs crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/kernel.rs && cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed with no warnings from the changed files.
- Next command: sync `wkv7/forward.rs`, `wkv7/kernel.rs`, and this note to `10.100.1.253`; remove only the WKV7 pretrain-output autotune cache so the new candidate is measured; run remote `cargo check -p rwkv-nn --features cuda,fusion`.

## Remote Sync And Compile

- Host/path: `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Command: scoped `rsync -azR` of the two WKV7 source files plus this note, then removed only `target/autotune/0.10.0/device-0-0-cuda/rwkv_nn-kernels-train-time_mixer-wkv7-forward-wkv7-pretrain-output-forward.json.log`.
- Remote compile command: `cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed in dev check profile.
- Next command: remote standard compare with regenerated GB10 baseline, `repeat=3,warmup=1`, capturing output in `target/rwkv-test/remote-wkv7-output-factored-compare.log`.

## Remote Standard Compare 1

- Host/path: `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Command: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1 | tee target/rwkv-test/remote-wkv7-output-factored-compare.log`.
- Binary/build provenance: remote release binary rebuilt after adding `row_tile_64_output_factored`.
- Result: activation passed.
  - `activation_summary compared=54 passed=54 failed=0`.
- Timing:
  - `timing_summary compared=76 passed=75 failed=1 missing=0 extra=0 ignored=1 actual_total_ms=74.658 baseline_total_ms=175.887 speedup=2.36x`.
  - The only failed timing row is the known short `cells/cell_0000/channel_mixer` row, outside WKV7.
  - `cells/*/time_mixer` remained strongly positive: `43.359ms` actual vs `120.381ms` baseline, `2.78x`.
- Decision: correctness is acceptable enough to inspect the WKV7 autotune log. Do not keep or revert yet; first determine whether the new factored candidate was selected and how it timed against `row_tile_64`.
- Next command: inspect remote WKV7 pretrain-output autotune log.

## Remote Autotune Log 1

- Host/path: `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Command: `tail -n 180 target/autotune/0.10.0/device-0-0-cuda/rwkv_nn-kernels-train-time_mixer-wkv7-forward-wkv7-pretrain-output-forward.json.log`.
- Result:
  - Selected candidate: `row_tile_64`, `fastest_index=2`.
  - `row_tile_64`: median `601.431us`, mean `600.717us`.
  - `row_tile_64_output_factored`: median `763.475us`, mean `763.819us`.
  - `row_tile_32`: median `927.713us`.
  - `row_tile_16`: median `1702.004us`.
- Interpretation:
  - The output-factorized algebra is numerically admissible on GB10 but slower than the current serial output accumulation.
  - The two extra block reductions and synchronization per timestep cost more than the saved repeated output arithmetic for this shape.
- Decision: reject and remove `row_tile_64_output_factored` from the active candidate set. Keep this note as negative evidence.
- Next edit: remove the factored kernel, reduction helpers, and candidate wiring from `wkv7/{kernel.rs,forward.rs}`, then run local non-GPU format/compile.

## Factored Candidate Revert

- Edited `crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/forward.rs`.
  - Removed `row_tile_64_output_factored` from the WKV7 output `LocalTuner`.
  - Removed the factored host launcher and validity helper.
- Edited `crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/kernel.rs`.
  - Removed the factored output kernel and its local reduction helpers.
- Source cleanup check: `rg -n "factored|output_factored|block_reduce_sum_f32|warp_reduce_sum_f32|WARP_SIZE" crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/{forward.rs,kernel.rs}` found no matches.
- Local non-GPU check:
  - Command: `rustup run nightly rustfmt crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/forward.rs crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/kernel.rs && cargo check -p rwkv-nn --features cuda,fusion`.
  - Result: passed.
- Keep/revert state: reject `row_tile_64_output_factored`; active source should return to the existing row-tile-only WKV7 output tuner.
- Next command: sync restored WKV7 files and this note to `10.100.1.253`, remove the WKV7 output autotune cache so the row-tile-only checksum is restored, then run remote compile and a standard compare.

## Remote Revert Sync And Compile

- Host/path: `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Command: scoped `rsync -azR` of restored WKV7 source files plus this note; removed the WKV7 output autotune cache.
- Remote compile command: `cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed in dev check profile.
- Next command: remote standard compare with regenerated GB10 baseline, `repeat=3,warmup=1`, capturing output in `target/rwkv-test/remote-wkv7-output-factored-reverted-compare.log`.

## Remote Revert Compare

- Host/path: `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Command: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1 | tee target/rwkv-test/remote-wkv7-output-factored-reverted-compare.log`.
- Binary/build provenance: remote release binary rebuilt after removing the factored candidate and clearing the WKV7 output autotune cache.
- Result: passed.
  - `activation_summary compared=54 passed=54 failed=0`.
  - `timing_summary compared=76 passed=76 failed=0 missing=0 extra=0 ignored=1 actual_total_ms=72.592 baseline_total_ms=175.887 speedup=2.42x`.
  - `cells/*/time_mixer`: `42.623ms` actual vs `120.381ms` baseline, `2.82x`.
- Interpretation: remote active source recovered to a clean row-tile-only state after rejecting the factored candidate.
- Next command: inspect the restored WKV7 autotune log to confirm only `row_tile_{16,32,64}` candidates remain and `row_tile_64` is selected.

## Restored Autotune Log

- Host/path: `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Command: `tail -n 140 target/autotune/0.10.0/device-0-0-cuda/rwkv_nn-kernels-train-time_mixer-wkv7-forward-wkv7-pretrain-output-forward.json.log`.
- Result:
  - Candidate set contains only `row_tile_64`, `row_tile_32`, and `row_tile_16`.
  - Selected candidate: `row_tile_64`, `fastest_index=2`.
  - `row_tile_64`: median `606.673us`.
  - `row_tile_32`: median `945.378us`.
  - `row_tile_16`: median `1762.514us`.
- Decision:
  - `row_tile_64_output_factored` is rejected and fully removed from active source and remote cache.
  - The factored output algebra was numerically safe on GB10, but slower because extra reductions/synchronization outweighed saved arithmetic.
  - Keep this branch/note as negative evidence. Do not retry this output-factorization candidate for `B=16,T=512,D=768,BF16,GB10` unless the implementation removes the per-step block reductions or changes the output computation boundary materially.
- Next edit: add a concise known-negative guard to `.agents/skills/kernel-tuning/SKILL.md`, then sync the skill and this final note to `10.100.1.253`.

## Skill Update

- Edited `.agents/skills/kernel-tuning/SKILL.md`.
- Added a known GB10 negative result:
  - `row_tile_64_output_factored` passed CUDA BF16 activation on `10.100.1.253`;
  - it was slower than current `row_tile_64` (`~763us` median vs `~601us`);
  - future attempts should not retry this output-factorization candidate unless the per-step reductions/synchronization or output boundary materially changes.
- Next command: sync this note and the updated skill to the remote mirror.

## Remote Note And Skill Sync

- Command: scoped `rsync -azR` of this note and `.agents/skills/kernel-tuning/SKILL.md` to `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable/`.
- Result: sync completed.
- Final keep/revert state:
  - Code: active WKV7 source is restored to row-tile-only output candidates; remote standard compare is clean (`54/54` activation, `76/76` timing, `2.42x` total speedup).
  - Note/skill: retained as negative evidence for WKV7 output factorization.
