# Remote Current After Low-Rank WKV7 GB10

- Date: 2026-05-16 18:11 +0800.
- Branch/worktree: `kernel-tuning-remote-current-after-lowrank-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits plus retained/reverted kernel tuning changes. This attempt owns only this attribution note unless a later entry explicitly opens a source implementation boundary.
- User constraint: debug first on remote `10.100.1.253`; do not use the local GPU.
- Prior-note and memory search commands:
  - `rg -n "WKV7|wkv7|state-scan|segment|low-rank|lowrank|projection|lm_head|GroupNorm|sumsq|target-logit|ncu|ERR_NVGPUCTRPERM|speedup" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md /root/.codex/memories/MEMORY.md -S`
  - `rg -n "Decision:|Keep/revert state|Final decision|Next command|Next action|current post-fusion top groups" .agents/notes/kernel-tuning/2026-05-16-*.md`
- Matched prior evidence:
  - WKV7 row-tile-only, shared-lanes, dense segment transform/scan/recompute, and low-rank segment-scan are closed. The low-rank design is mathematically valid for summaries but still cannot beat the measured recompute/output lower bound on GB10.
  - GatedReadout GroupNorm+combine is kept; warp32, row-pack, and sumsq-variance variants are closed.
  - LM-head direct target-logit row kernel is kept on GB10; online softmax is rejected.
  - LM-head projection timing is emitted as an optional comparable row; projection+loss fusion analysis says a real fused implementation is a broad operator/API project, not another row-loss tweak.
  - LayerNorm GB10 uses the fast `256` policy; local deterministic boundary remains separate and must not be generalized.
  - Remote `ncu` is blocked by `ERR_NVGPUCTRPERM`, so this branch can use standard compare, `nsys` kernel time, launch geometry, registers/thread, and shared-memory metadata, but not achieved occupancy/warp/memory counters.
- Machine/GPU: remote `caizus@10.100.1.253`, host `spark-35ac`, `NVIDIA GB10`, compute capability `12.1`, utilization `0%` at preflight.
- Remote run directory: `/home/caizus/Projects/Packages/rwkv-rs-stable`, synchronized non-git mirror.
- Baseline path: `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, heads `12`, head size `64`, vocab `65536`.
- Hypothesis: after closing WKV7 low-rank, the correct next step is to refresh remote current-source attribution before opening any new implementation branch. The previous top groups may be stale relative to the kept target-logit/projection-timing/source state.
- Candidate parameters: none. This is a read-only attribution attempt.
- Expected keep/revert boundary: keep this note as evidence. If the top remaining surfaces are duplicate-closed or broad Cubek/TMA matmul work, do not edit kernels. If a new small non-duplicate project-owned boundary appears, open a separate branch and prewrite its note before code changes.
- Next command: sync the current local source mirror to `10.100.1.253` excluding `.git`, `target`, `weights`, and `results`; run remote `cargo check -p rwkv-test --features cuda`, standard compare, then short `nsys` and sqlite ranking.

## Remote Sync

- Command: `rsync -az --delete` of the local source mirror to `/home/caizus/Projects/Packages/rwkv-rs-stable` on `10.100.1.253`, excluding `.git/`, `target/`, `weights/`, and `results/`.
- Result: completed successfully.
- Remote source provenance: synchronized mirror of local branch `kernel-tuning-remote-current-after-lowrank-gb10-20260516` with the dirty-tree constraint recorded above.
- Next command: remote compile check `cargo check -p rwkv-test --features cuda`.

## Remote Compile Check

- Command: `cargo check -p rwkv-test --features cuda` on `10.100.1.253`.
- Result: passed in dev check profile.
- Next command: remote standard compare with regenerated GB10 baseline, `repeat=3,warmup=1`, capturing output in `target/rwkv-test/remote-current-after-lowrank-compare.log`.

## Remote Standard Compare

- Command: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1`, captured in `target/rwkv-test/remote-current-after-lowrank-compare.log`.
- Binary/build provenance: remote release binary was already up to date after sync.
- Activation result: passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: clean, `timing_summary compared=76 passed=76 failed=0 missing=0 extra=0 ignored=1 actual_total_ms=74.015 baseline_total_ms=175.887 speedup=2.38x`.
- Module timing signal:
  - `cells/*/time_mixer`: `43.453ms` vs `120.381ms`, `2.77x`.
  - `cells/*/channel_mixer`: `19.544ms` vs `25.759ms`, `1.32x`.
  - `loss/l2wrap_cross_entropy`: `4.423ms` vs `5.957ms`, `1.35x`.
- Decision: current remote source is activation-clean and timing-clean. Use this release binary for profiler attribution.
- Next command: short remote `nsys` profile with `repeat=1,warmup=1`, export sqlite, and rank kernel groups. The profiled compare timing is instrumentation-only and not an acceptance gate.

## Remote Nsys Profile

- Command: `nsys profile --trace=cuda,nvtx --sample=none --force-overwrite=true --output=target/rwkv-test/nsys-current-after-lowrank-gb10 target/release/rwkv-test compare-rwkv-nn --color never --baseline ... --repeat 1 --warmup 1`.
- Result: generated `target/rwkv-test/nsys-current-after-lowrank-gb10.nsys-rep`. Command exit was `1` because profile mode uses `repeat=1,warmup=1` against a `repeat=3,warmup=1` timing baseline; this is expected and is not an acceptance result.
- Export command: `nsys export --type sqlite --force-overwrite=true --output target/rwkv-test/nsys-current-after-lowrank-gb10.sqlite ...`.
- Export result: succeeded.
- Top current CUDA groups from the sqlite summary:
  - lm-head projection Cubek/TMA matmul: `46.376ms / 3`, `grid=(4096,16,1)`, `block=(32,12,1)`, regs/thread `73`, dynamic smem `27648`.
  - recurring Cubek/TMA matmul groups: `25.405ms / 144`, `21.061ms / 36`, `18.911ms / 36`.
  - `wkv7_pretrain_forward_output_kernel_f_bf16`: `23.166ms / 36`, `grid=(12,16,1)`, `block=(64,1,1)`, regs/thread `108`, dynamic smem `1536`.
  - `mix6_forward_kernel_f__n_1`: `14.579ms / 36`, `grid=(24576,1,1)`, `block=(32,8,1)`, regs/thread `32`.
  - `channel_mixer_relu_square_forward_kernel_f__n_2`: `14.216ms / 36`, `grid=(49152,1,1)`, `block=(32,8,1)`, regs/thread `16`.
  - `lm_head_l2wrap_ce_forward_row_kernel_f__i_i32`: `12.800ms / 3`, `grid=(8192,1,1)`, `block=(1024,1,1)`, regs/thread `40`, dynamic smem `128`.
  - `gated_readout_combine_forward_kernel_f_`: `11.721ms / 36`, `grid=(8192,12,1)`, `block=(64,1,1)`, regs/thread `25`.
  - `key_prepare_forward_64_kernel_f_`: `9.467ms / 36`, `grid=(24576,1,1)`, `block=(128,1,1)`, regs/thread `24`.
  - `kernel_binop_c_bf16_n_8`: `8.550ms / 72`.
  - `layer_norm_forward_kernel_f_`: `6.541ms / 78`, `block=(256,1,1)`.
  - `value_residual_gate_forward_kernel_f__n_2`: `6.524ms / 33`.
- Interpretation:
  - The largest remaining surface is the unembed/lm-head projection matmul, which is broad Cubek/TMA work and already classified as too large for a quick row-loss tweak.
  - WKV7 is still the largest project-owned custom kernel, but row-tile, shared-lanes, dense segment recompute, and low-rank segment-scan are now closed.
  - The other custom kernels in the top list match existing kept/rejected notes; no obvious stale candidate selection is visible from launch geometry alone.
- Next command: read current remote autotune logs for the top custom kernels without changing caches, then decide whether there is any non-duplicate small branch left.

## Remote Autotune Audit

- Command: read `target/autotune/**/*.json.log` entries matching current train-kernel names on `10.100.1.253`; no cache files were deleted or changed.
- Current selections:
  - `channel_mixer_mix_forward`: `line_size_8`.
  - `channel_mixer_relu_square_forward`: `line_size_2`; `line_size_4` remains near-tied but slower.
  - `layer_norm_forward`: `block_256`, with `deterministic_min_block_size=256` for the GB10 key.
  - `lm_head_l2wrap_ce_forward`: `block_1024`.
  - `learning_rate_gate_forward`: `line_size_8`.
  - `mix6_forward`: `line_size_1`.
  - `value_residual_gate_forward`: `line_size_2`.
  - `weight_decay_transform_forward`: `line_size_4`, with `line_size_8` near-tied.
  - `wkv7_pretrain_output_forward`: `row_tile_64`.
- Caveat: a `key_prepare-forward-64` autotune log from the rejected warps branch still exists and reports `warps_per_cube_2`, but current source and current nsys launch use fixed `block=(128,1,1)` / four warps. Treat that cache file as stale residue, not live dispatch.
- Decision:
  - Do not open another line-size/block/row-tile branch from this evidence.
  - The next correct small step is measurement coverage, not another kernel retry: update the Python baseline generator in companion `rwkv-rs-test` so it emits `timing/lm_head/projection.time.json` without exporting logits. The Rust actual path already emits this optional row, but current baseline lacks it, so compare counts it as `ignored=1` and the largest projection matmul stays outside canonical timing.
- Next command: inspect companion repo `/mnt/g/Projects/Packages/rwkv-rs-test` branch/dirty state and open a separate branch there for the baseline timing contract if safe.
