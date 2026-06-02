# WKV7 Direct Value Load GB10

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-wkv7-direct-value-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits and accumulated kernel tuning changes. This attempt owns only `crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/kernel.rs` and this note unless later entries explicitly expand scope.
- User constraint: debug/tune first on remote `10.100.1.253`; do not use local GPU timing while the local GPU may be busy.
- Prior-note search commands:
  - `rg -n "WKV7|wkv7|row_tile|shared-lanes|segment|factored|shared_value|direct value|10\\.100\\.1\\.253|GB10" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md crates/rwkv-nn/src/kernels/train/time_mixer/wkv7 -S`
  - `rg -n "shared_value|value direct|direct value|WKV7.*value|wkv7.*shared.*value" .agents/notes/kernel-tuning crates/rwkv-nn/src/kernels/train/time_mixer/wkv7`
- Matched prior evidence:
  - WKV7 row-tile tuning is closed; GB10 selects `row_tile_64`.
  - WKV7 shared-lanes, segment recompute, low-rank/state-scan design, and output factorization are recorded negatives or larger design blockers.
  - `2026-05-16-remote-current-after-gb10-results.md` current profile: `wkv7_pretrain_forward_output_kernel_f_bf16` is `22.551ms / 36`, `grid=(12,16,1)`, `block=(64,1,1)`, regs/thread `127`, dynamic shared memory `1536`.
  - No prior note mentions removing `shared_value` or direct-loading the per-row value scalar inside the WKV7 output kernel.
- Machine/GPU: remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`; local GPU must not be used.
- Remote run directory: `/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, heads `12`, head size `64`.
- Baseline path: `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Kernel/stage: `wkv7_pretrain_forward_output_kernel`, forward output path only.
- Hypothesis:
  - The kernel currently loads all six timestep vectors into shared memory. `value` differs from the other five: each output row uses only `value[row_index]`, so it is not shared across rows.
  - Removing `shared_value` and direct-loading `inputs.value[value_index]` once per active row/timestep should preserve math while reducing shared-memory footprint and one shared write/read pair per timestep.
  - This is not another row-tile/shared-lanes/segment/factored-output retry; it is a narrower implementation-traffic change inside the current `row_tile_64` winner.
- Accuracy risk: state update and output accumulation order are unchanged. Activation should remain `54/54`; any drift means the implementation was wrong.
- Expected keep/revert boundary:
  - Keep only if remote compile passes, activation passes, total speedup remains above `1.0`, and `nsys` shows WKV7 output time improves or at least does not regress versus current `22.551ms / 36`.
  - Revert if activation drifts, standard compare regresses meaningfully, or WKV7 targeted profiler time is flat/worse.
- Next edit: remove `shared_value` from `wkv7_pretrain_forward_output_kernel` and load `value` directly from `inputs.value[value_index]` inside the active row block.

## Implementation Edit

- Edited `crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/kernel.rs`.
- Removed `shared_value = SharedMemory::<f32>::new(head_size)`.
- Removed the per-timestep `shared_value[lane_index]` write.
- Replaced `let value = shared_value[row_index]` with a direct `inputs.value[value_index]` load after `value_index` is computed for the active row.
- Next command: rustfmt the edited WKV7 kernel file and run local compile check only, `cargo check -p rwkv-nn --features cuda,fusion`; no local GPU execution.

## Local Non-GPU Check

- Command: `rustup run nightly rustfmt crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/kernel.rs && cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed.
- No local GPU execution was performed.
- Next command: scoped rsync of `wkv7/kernel.rs` and this note to `10.100.1.253`, then remote compile check. Do not clear WKV7 autotune cache because the candidate set and tunable names are unchanged; the cached `row_tile_64` selection should dispatch the rebuilt kernel.

## Remote Sync And Compile

- Command: scoped `rsync -azR` of `crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/kernel.rs` and this note to `10.100.1.253`, then `cargo check -p rwkv-nn --features cuda,fusion`.
- Result: sync completed and remote compile passed.
- Next command: standard remote compare with regenerated GB10 baseline, `repeat=3,warmup=1`, captured in `target/rwkv-test/remote-wkv7-direct-value-compare.log`.

## Remote Standard Compare

- Command: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1`, captured in `target/rwkv-test/remote-wkv7-direct-value-compare.log`.
- Binary/build provenance: remote release binary rebuilt after the direct-value WKV7 kernel edit.
- Activation result: passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: positive with one known unrelated edge failure, `timing_summary compared=76 passed=75 failed=1 missing=0 extra=0 ignored=1 actual_total_ms=73.884 baseline_total_ms=175.887 speedup=2.38x`.
- Failed row: known short `timing/cells/cell_0000/channel_mixer.time.json`, `1.817ms` vs `1.668ms`, `0.92x`.
- Target module signal:
  - `cells/*/time_mixer`: `43.721ms` vs baseline `120.381ms`, `2.75x`.
  - Current-profile pre-edit reference was `42.360ms`, so standard compare does not show an obvious win.
- Decision: do not keep/revert from standard compare alone. Run targeted `nsys` to see whether WKV7 output time improves versus current reference `22.551ms / 36`; if flat/worse, revert.
- Next command: short remote `nsys` profile with `repeat=1,warmup=1`, export sqlite, and query WKV7 output kernel launch metadata/time.

## Remote Nsys Profile

- Command: `nsys profile --trace=cuda,nvtx --sample=none --force-overwrite=true --output=target/rwkv-test/nsys-wkv7-direct-value-gb10 target/release/rwkv-test compare-rwkv-nn --color never --baseline ... --repeat 1 --warmup 1`.
- Result: exited nonzero because profile mode uses `repeat=1,warmup=1` against a `repeat=3,warmup=1` baseline. This is expected for attribution mode.
- Activation sanity inside profile run: passed, `activation_summary compared=54 passed=54 failed=0`.
- Profile-mode timing is invalid for acceptance due timing-profile mismatch, but rough total stayed positive: `actual_total_ms=76.467 baseline_total_ms=175.887 speedup=2.30x`.
- Generated artifact: `target/rwkv-test/nsys-wkv7-direct-value-gb10.nsys-rep`.
- Next command: export the nsys report to sqlite and query WKV7 output kernel plus adjacent top groups.

## Nsys Export And Kernel Query

- Command: exported `target/rwkv-test/nsys-wkv7-direct-value-gb10.nsys-rep` to sqlite, then queried WKV7 and adjacent custom kernels by demangled name.
- Result:
  - `wkv7_pretrain_forward_output_kernel_f_bf16`: `28.364ms / 36`, avg `787.880us`, `grid=(12,16,1)`, `block=(64,1,1)`, regs/thread `104`, dynamic smem `1280`.
  - Current-profile reference before this edit: `22.551ms / 36`, avg `626.420us`, regs/thread `127`, dynamic smem `1536`.
  - Adjacent rows stayed in expected ranges: `mix6_forward` `14.173ms / 36`, `channel_mixer_relu_square` `14.344ms / 36`, `lm_head_l2wrap_ce_forward_row` `12.812ms / 3`.
- Interpretation:
  - The edit reduced register count and dynamic shared-memory footprint, but it made the WKV7 output kernel about `25.8%` slower in this profile.
  - For this access pattern, direct global `value[row]` loads are worse than staging `value` through the existing shared vector, likely because the shared staging keeps the per-timestep input vector load pattern coalesced with the other five vectors.
- Decision: reject and revert the direct-value WKV7 change. Keep this note as negative evidence.
- Next edit: restore `shared_value` allocation, per-lane load, and `let value = shared_value[row_index]` in `wkv7/kernel.rs`, then run local compile and sync the revert to remote.

## Revert Edit

- Reverted `crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/kernel.rs`.
- Restored `shared_value = SharedMemory::<f32>::new(head_size)`.
- Restored the per-timestep `shared_value[lane_index]` load.
- Restored `let value = shared_value[row_index]`.
- Next command: rustfmt the WKV7 kernel file and run local compile check only, `cargo check -p rwkv-nn --features cuda,fusion`; no local GPU execution.

## Revert Local Check

- Command: `rustup run nightly rustfmt crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/kernel.rs && cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed.
- No local GPU execution was performed.
- Next command: sync the reverted WKV7 kernel and this note to `10.100.1.253`, run remote compile, then run one post-revert standard compare to leave the remote run directory in a known good state.

## Post-Revert Remote Check

- Command: scoped `rsync -azR` of reverted `wkv7/kernel.rs` plus this note, then remote `cargo check -p rwkv-nn --features cuda,fusion` and standard compare captured in `target/rwkv-test/remote-wkv7-direct-value-post-revert-compare.log`.
- Remote compile: passed.
- Release rebuild: completed after the revert, proving the remote binary no longer contains the direct-value WKV7 edit.
- Activation result: passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: positive with the known channel-mixer edge, `timing_summary compared=76 passed=75 failed=1 missing=0 extra=0 ignored=1 actual_total_ms=73.153 baseline_total_ms=175.887 speedup=2.40x`.
- Failed row: known short `timing/cells/cell_0000/channel_mixer.time.json`, `1.743ms` vs `1.668ms`, `0.96x`.
- Time-mixer group recovered versus direct-value standard compare: `42.907ms` post-revert vs `43.721ms` direct-value.
- Final decision: reject and revert WKV7 direct-value loading. Keep the branch/note as negative evidence; active local and remote source use the original shared `value` staging.
- Next edit: add a concise known-negative guard to `.agents/skills/kernel-tuning/SKILL.md`, then sync this final note and the updated skill to remote.

## Skill Update

- Edited `.agents/skills/kernel-tuning/SKILL.md`.
- Added a known GB10 negative result for WKV7 direct-value loading:
  - reducing registers/shared memory did not help;
  - WKV7 output time regressed from `22.551ms / 36` to `28.364ms / 36`;
  - keep shared `value` staging unless a branch changes the broader input-load/coalescing strategy.
- Next command: run `git diff --check` on the updated skill and this note, then sync both to `10.100.1.253`.

## Final Sync

- Command: `git diff --check -- .agents/skills/kernel-tuning/SKILL.md .agents/notes/kernel-tuning/2026-05-16-wkv7-direct-value-gb10.md`, then scoped `rsync -azR` of the updated skill, this note, and `2026-05-16-remote-current-after-gb10-results.md` to `10.100.1.253`.
- Result: diff check passed and sync completed.
- Keep/revert state:
  - Code: reverted; active local and remote WKV7 output kernel still stages `value` through shared memory.
  - Skill/note: updated with the direct-value negative result.
