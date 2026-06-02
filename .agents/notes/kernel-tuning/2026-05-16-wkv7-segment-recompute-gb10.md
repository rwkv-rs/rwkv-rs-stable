# WKV7 Segment Recompute GB10

- Date: 2026-05-16 17:22 +0800.
- Branch/worktree: `kernel-tuning-wkv7-segment-recompute-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits plus retained/reverted kernel tuning changes. This attempt owns only `crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/{forward.rs,kernel.rs}` and this note unless later entries explicitly expand scope.
- Prior-note/source search commands:
  - `rg -n "WKV7|wkv7|state-scan|state scan|time-split|time split|chunk|handoff|shared-lanes|row_tile|prefix|scan" .agents/notes/kernel-tuning crates/rwkv-nn/src/kernels/train/time_mixer/wkv7 /root/.codex/memories/MEMORY.md -S`
  - `sed -n '270,690p' crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/forward.rs`
  - `sed -n '1,260p' crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/kernel.rs`
- Matched prior evidence:
  - `2026-05-16-wkv7-state-scan-design-gb10b.md` derived the state transform `S_{t+1}=S_t M_t + outer(value_t, replacement_key_t)` and classified segment recompute as the first mathematically safe implementation candidate.
  - Current remote post-target-logit attribution on `10.100.1.253` records `wkv7_pretrain_forward_output_kernel_f_bf16` at `22.412ms / 36`, `grid=(12,16,1)`, `block=(64,1,1)`, regs `108`, dynamic smem `1536`.
  - Row-tile-only and shared-lanes attempts are duplicate-closed and must not be repeated.
- Machine/GPU: primary validation on remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`; local GPU must not be used.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, heads `12`, head size `64`, chunk len `16`.
- Kernel/stage: WKV7 pretrain forward output path only.
- Hypothesis: a segment recompute candidate with `segment_len=16` exposes `B * H * (T/16) = 6144` output blocks instead of the current `192` row-tile blocks. It uses f32 segment transforms `(P,Q)` and segment-start states to preserve the recurrence, then recomputes the original per-segment recurrence from the correct start state.
- Candidate parameters:
  - Add one new tuner candidate `segment_recompute_16` alongside existing `row_tile_{16,32,64}`.
  - Segment length is fixed to the existing `chunk_len=16`.
  - New intermediates are f32 tensors:
    - `transform_p`: `[B,H,num_segments,D,D]`
    - `transform_q`: `[B,H,num_segments,D,D]`
    - `segment_start`: `[B,H,num_segments,D,D]`
  - Kernel launches:
    1. segment transform, one block per `(head,batch,segment)`, one unit per state row;
    2. segment-start scan, one block per `(head,batch)`, one unit per state row;
    3. segment recompute output, one block per `(head,batch,segment)`, one unit per state row.
- Accuracy risk: f32 `P/Q` composition changes accumulation order versus the single serial kernel. If activation drifts, inspect transform composition, indexing, and f32/BF16 conversion before reverting.
- Expected keep/revert boundary:
  - Keep only if local non-GPU compile/format checks pass, remote compile passes, remote activation remains `54/54`, and targeted remote nsys shows WKV7 time improves versus `22.412ms / 36` without total speedup dropping below `1.0`.
  - If activation fails, do not assume the idea is invalid; first inspect whether segment transform composition or index layout is wrong.
  - If activation passes but timing is worse, record whether the cause is extra f32 transform work, global-memory intermediate traffic, scan overhead, or launch overhead before reverting.
- Next edit: add the three segment kernels and wire `segment_recompute_16` into the existing WKV7 output `LocalTuner`.

## Implementation Edit

- Edited `crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/kernel.rs`.
  - Added `segment_state_index`.
  - Added `wkv7_pretrain_segment_transform_kernel`.
  - Added `wkv7_pretrain_segment_scan_kernel`.
  - Added `wkv7_pretrain_segment_recompute_output_kernel`.
- Edited `crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/forward.rs`.
  - Added candidate `segment_recompute_16` to `wkv7-pretrain-output-forward` `LocalTuner`.
  - Added `wkv7_pretrain_output_segment_recompute` host launcher with f32 `transform_p`, `transform_q`, and `segment_start` intermediates.
  - Kept existing `row_tile_{16,32,64}` candidates.
- Next command: run rustfmt on the two edited WKV7 files, then local non-GPU compile check `cargo check -p rwkv-nn --features cuda,fusion`.

## Local Non-GPU Check 1

- Command: `rustup run nightly rustfmt crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/kernel.rs crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/forward.rs && cargo check -p rwkv-nn --features cuda,fusion`.
- Result: failed at Rust compile. CubeCL generated `launch_unchecked` for the new `#[define(F)]` kernels takes only one generic runtime argument, but the host code called `::<F, R>`.
- Interpretation: implementation wiring error, not a numerical or design result.
- Next edit: change the two new launch calls to `::<R>`.

## Launch Generic Fix

- Edited `crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/forward.rs`.
  - Changed the two new segment kernel launch calls from `::<F, R>` to `::<R>`.
- Command: `rustup run nightly rustfmt crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/kernel.rs crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/forward.rs && cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed in dev check profile.
- Next command: sync the two edited WKV7 files and this note to `10.100.1.253`, remove only the WKV7 pretrain-output autotune cache entry so the new candidate is measured, then run remote compile check.

## Remote Sync And Compile

- Command: scoped `rsync -azR` of:
  - `crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/forward.rs`
  - `crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/kernel.rs`
  - `.agents/notes/kernel-tuning/2026-05-16-wkv7-segment-recompute-gb10.md`
- Remote cache action: removed only `target/autotune/0.10.0/device-0-0-cuda/rwkv_nn-kernels-train-time_mixer-wkv7-forward-wkv7-pretrain-output-forward.json.log`.
- Remote compile command: `cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed in dev check profile on `10.100.1.253`.
- Next command: remote standard compare with regenerated GB10 baseline, `repeat=3,warmup=1`, capturing output in `target/rwkv-test/remote-wkv7-segment-recompute-compare.log`.

## Remote Standard Compare 1

- Command: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline ~/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1`, captured in `target/rwkv-test/remote-wkv7-segment-recompute-compare.log`.
- Binary/build provenance: remote release binary rebuilt after WKV7 segment-recompute code changes and WKV7 output autotune cache removal.
- Result: passed.
- Activation: `activation_summary compared=54 passed=54 failed=0`.
- Timing: `timing_summary compared=76 passed=76 failed=0 missing=0 extra=0 ignored=1 actual_total_ms=72.854 baseline_total_ms=175.887 speedup=2.41x`.
- Next command: inspect the WKV7 output autotune log to see whether `segment_recompute_16` was selected.

## Remote Autotune Log 1

- Command: `tail -n 160 target/autotune/0.10.0/device-0-0-cuda/rwkv_nn-kernels-train-time_mixer-wkv7-forward-wkv7-pretrain-output-forward.json.log`.
- Selected candidate: `row_tile_64`, `fastest_index=2`.
- Candidate timings:
  - `row_tile_64`: median `607.728us`, mean `608.129us`.
  - `row_tile_32`: median `929.344us`, mean `1038.370us`.
  - `row_tile_16`: median `1690.639us`, mean `1692.613us`.
  - `segment_recompute_16`: median `7693.660us`, mean `7829.533us`.
- Interpretation:
  - The segment-recompute candidate is about `12.7x` slower than `row_tile_64` in tuner timing and was not selected.
  - Standard activation did not validate the segment implementation because the selected winner was still `row_tile_64`.
  - The candidate's slowness is plausible from design: it adds two large f32 `[B,H,segments,64,64]` transform tensors plus a segment-start tensor, two extra global-memory-heavy launches, and an O(`segments * 64^3`) scan kernel before recomputing the original recurrence.
- Next edit: temporarily force the segment-recompute candidate so activation/profile can validate implementation correctness. This forced state is for diagnosis only and must be reverted or removed after evidence is recorded.

## Forced Segment Validation Edit

- Edited `crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/forward.rs`.
  - Added temporary `FORCE_SEGMENT_RECOMPUTE_FOR_VALIDATION = true`.
  - Row-tile candidates now return `-1` from the tuner group while this flag is true, forcing `segment_recompute_16` to be selected.
- This is a diagnostic edit, not a keepable state.
- Next command: run rustfmt and local non-GPU compile check before syncing the forced diagnostic to remote.

## Forced Segment Local Check

- Command: `rustup run nightly rustfmt crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/forward.rs crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/kernel.rs && cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed in dev check profile.
- Next command: sync forced WKV7 files and this note to `10.100.1.253`, delete the WKV7 output autotune cache again, and run remote compile plus standard compare.

## Forced Segment Remote Compare

- Remote sync/compile: synced the forced diagnostic WKV7 files to `10.100.1.253`, removed the WKV7 output autotune cache, and `cargo check -p rwkv-nn --features cuda,fusion` passed on the remote host.
- Command: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline ~/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1`, captured in `target/rwkv-test/remote-wkv7-segment-recompute-forced-compare.log`.
- Binary/build provenance: remote release binary rebuilt after the forced segment diagnostic flag.
- Result: activation passed, timing failed.
- Activation: `activation_summary compared=54 passed=54 failed=0`.
- Timing evidence from the log tail:
  - `cells/*/time_mixer actual_total_ms=129.605 baseline_total_ms=120.381 speedup=0.93x`.
  - `timing_summary compared=76 passed=65 failed=11 missing=0 extra=0 ignored=1 actual_total_ms=159.664 ...`.
  - The failed rows are `timing/cells/cell_0001..cell_0011/time_mixer.time.json`; `cell_0000/time_mixer` still passed.
- Follow-up exact extraction:
  - `grep` confirmed `activation_summary compared=54 passed=54 failed=0`.
  - `grep` confirmed `timing_summary compared=76 passed=65 failed=11 missing=0 extra=0 ignored=1 actual_total_ms=159.664 ...`.
  - Remote `rg` is unavailable, so further remote log extraction uses `grep`/`find`.
  - Forced autotune log reports `fastest_index=3`, which is the `segment_recompute_16` candidate after row-tile groups are invalidated by the diagnostic flag.
- Interpretation:
  - The segment recompute implementation is numerically valid for the trace-backed CUDA BF16 fixture, so the recurrence composition/indexing is not the immediate defect.
  - The candidate is too slow in the actual path, consistent with the earlier tuner evidence (`segment_recompute_16` about `12.7x` slower than `row_tile_64`).
- Next command: run a short remote nsys profile of the forced path, export sqlite, and group CUDA kernels to split transform/scan/recompute cost before reverting the diagnostic force flag.

## Forced Segment Nsys Attempt 1

- Command: remote `nsys profile --force-overwrite=true --stats=false -o target/rwkv-test/nsys-wkv7-segment-forced-gb10 target/release/rwkv-test compare-rwkv-nn --color never --baseline ~/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 1 --warmup 1`, then `nsys export --type sqlite`.
- Result: combined command exited non-zero because the export step did not produce sqlite, but the profile step itself completed and generated `target/rwkv-test/nsys-wkv7-segment-forced-gb10.nsys-rep`.
- Profile log timing from `repeat=1,warmup=1` under nsys:
  - activation still passed `54/54`.
  - `cells/*/time_mixer actual_total_ms=133.746 baseline_total_ms=120.381 speedup=0.90x`.
  - `timing_summary ... actual_total_ms=167.031 ...`; all timing rows are marked fail because `repeat=1` is outside the standard timing acceptance boundary.
- Status: `.nsys-rep` is usable for kernel attribution; timing pass/fail from this profiled `repeat=1` run is diagnostic only.
- Separate sqlite export succeeded: `target/rwkv-test/nsys-wkv7-segment-forced-gb10.sqlite`.
- Kernel attribution from forced segment sqlite:
  - `wkv7_pretrain_segment_transform_kernel_f_bf16`: 36 launches, `131.157ms` total, `3.643ms` avg, grid `12x16x32`, block `64x1x1`, regs/thread `163`, dynamic smem `1280`.
  - `wkv7_pretrain_segment_scan_kernel`: 36 launches, `112.660ms` total, `3.129ms` avg, grid `12x16x1`, block `64x1x1`, regs/thread `255`, dynamic smem `0`.
  - `wkv7_pretrain_segment_recompute_output_kernel_f_bf16`: 36 launches, `46.976ms` total, `1.305ms` avg, grid `12x16x32`, block `64x1x1`, regs/thread `115`, dynamic smem `1536`.
- Baseline row-tile attribution from existing `nsys-wkv7-rowtile-current.sqlite`:
  - `wkv7_pretrain_forward_output_kernel_f_bf16`: 36 launches, `22.083ms` total, `0.613ms` avg, grid `12x16x1`, block `64x1x1`, regs/thread `108`, dynamic smem `1536`.
- Diagnosis:
  - The segment recompute path is mathematically correct for the trace fixture but architecturally wrong for this shape/GPU. It replaces one compact serial-per-head-row kernel with transform + scan + recompute kernels whose CUDA time is about `13.2x` the current row-tile kernel in the profile.
  - The scan kernel is especially poor: `255` registers/thread indicates severe register pressure, and the transform/recompute grids introduce `32x` more z-dimension blocks plus global f32 state traffic.
- Decision: reject `segment_recompute_16`. Revert the diagnostic force flag and remove the segment candidate/kernels from the active source so the branch does not leave a slow candidate in the autotune set.
- Next edit: remove the segment recompute code from `wkv7/forward.rs` and `wkv7/kernel.rs`, then run local compile and sync the restored WKV7 files back to `10.100.1.253`.

## Segment Candidate Revert Edit

- Edited `crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/forward.rs`.
  - Removed `FORCE_SEGMENT_RECOMPUTE_FOR_VALIDATION`.
  - Removed `segment_recompute_16` from the WKV7 output `LocalTuner`.
  - Removed the segment recompute host launcher and validity predicate.
  - Restored row-tile grouping to use only `is_valid_pretrain_output_row_tile`.
- Edited `crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/kernel.rs`.
  - Removed `segment_state_index`.
  - Removed the transform, scan, and recompute segment kernels.
- Source check: `rg` now finds segment identifiers only in this note, not in active WKV7 source.
- Next command: rustfmt the two WKV7 files and run local non-GPU `cargo check -p rwkv-nn --features cuda,fusion`.

## Local Revert Check

- Command: `rustup run nightly rustfmt crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/forward.rs crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/kernel.rs && cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed in dev check profile.
- Next command: sync restored WKV7 files and this note to `10.100.1.253`, remove the WKV7 output autotune cache so the row-tile-only checksum is restored, then run remote compile and standard compare.

## Remote Revert Sync And Compile

- Command: scoped `rsync -azR` of restored `wkv7/forward.rs`, `wkv7/kernel.rs`, and this note to `/home/caizus/Projects/Packages/rwkv-rs-stable` on `10.100.1.253`, then removed the WKV7 output autotune cache and ran `cargo check -p rwkv-nn --features cuda,fusion`.
- Result: remote compile passed in dev check profile.
- Next command: remote standard compare with `repeat=3,warmup=1`, capturing output in `target/rwkv-test/remote-wkv7-segment-recompute-reverted-compare.log`.

## Remote Revert Compare

- Command: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline ~/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1`, captured in `target/rwkv-test/remote-wkv7-segment-recompute-reverted-compare.log`.
- Result: passed.
- Activation: `activation_summary compared=54 passed=54 failed=0`.
- Timing:
  - `cells/*/time_mixer actual_total_ms=44.501 baseline_total_ms=120.381 speedup=2.71x`.
  - `timing_summary compared=76 passed=76 failed=0 missing=0 extra=0 ignored=1 actual_total_ms=75.804 baseline_total_ms=175.887 speedup=2.32x`.
- Autotune restore check:
  - WKV7 output cache now has only `row_tile_{64,32,16}` names.
  - `fastest_index=2`, selecting `row_tile_64`.
- Source restore check: `rg` finds no segment recompute identifiers in active WKV7 source.
- Keep/revert state: active source has no segment recompute candidate and no diagnostic force flag. The negative evidence remains in this note and the remote profiler artifacts.
- Decision: closed as rejected design. Do not retry segment transform/scan/recompute for this `B=16,T=512,D=768,BF16,GB10` boundary unless the algorithm avoids global f32 transform tensors and the high-register segment scan.
