# Gated Readout Row-Pack on GB10

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-gated-readout-rowpack-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this branch inherits broad unrelated workspace changes and the retained 64-thread/two-warp GatedReadout combine. This attempt owns only `gated_readout_combine` forward launch geometry/kernel code plus this note unless later entries explicitly expand scope.
- Prior-note search command:
  - `rg -n "gated_readout|GatedReadout|gate|combine|warp32|row-pack|rowpack|block=\\(64|block=\\(32|key_prepare|remote|GB10|10\\.100\\.1\\.253" .agents/notes/kernel-tuning /root/.codex/memories/MEMORY.md -S`
- Matched prior evidence:
  - `2026-05-16-gated-readout-forward-combine-gb10.md`: retained 64-thread/two-warp combine passes activation and reduces generic binop/reduce surface. Clean-cache standard compares reached `2.05x-2.14x`; clean `nsys` reported `gated_readout_combine_forward_kernel_f_` at `11.793ms / 36`, `grid=(8192,12,1)`, `block=(64,1,1)`, `regs=21`, `dynamic_smem=8`.
  - `2026-05-16-gated-readout-warp32-gb10.md`: one-warp-per-head with `grid=(8192,12,1)`, `block=(32,1,1)` passed activation but made the targeted kernel slower (`12.167ms / 36`, `regs=26`) and was reverted. Do not retry that same boundary.
  - `2026-05-16-key-prepare-warps-gb10.md`: key-prepare `warps_per_cube` tuner selected `block=(64,1,1)` and slightly regressed targeted time, so it was rejected and reverted.
  - Duplicate guards remain closed: residual-add/Burn-add A/B, channel-mixer forced Cube/Burn-reference/fusion, LocalTuner bypass, WKV7 forced `row_tile=16`, lm-head target-logit/atomic/online-softmax/prune-256, ordered-256 LayerNorm, GatedReadout warp32, and key-prepare warps-per-cube.
- Machine/GPU: remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`. Do not run local GPU.
- Remote run directory: `/home/caizus/Projects/Packages/rwkv-rs-stable`, synchronized non-git copy. Branch provenance remains local.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, heads `12`, head size `64`, regenerated remote baseline under `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Kernel/stage: TimeMixer GatedReadout combine forward.
- Changed boundary: this is not the rejected warp32 per-head-block retry. It packs all 12 heads for the same row into one cube: `grid=(rows,1,1)`, `block=(12*32,1,1)`, one warp per head, no inter-warp shared reduction.
- Hypothesis: the rejected warp32 version lost per-head due to more registers and one-warp math, but it still launched one block per `(row, head)`. Row packing reduces block count by `12x` and removes the 64-thread shared-memory reduction, so it may improve scheduler/block overhead enough to beat the retained 64-thread per-head kernel on GB10.
- Candidate parameters:
  - `HEAD_SIZE=64`
  - `NUM_HEADS=12` for the current `D=768` fixture
  - `BLOCK_SIZE=384`
  - vector width by logical lane pair: one warp computes one head and each lane writes up to two head lanes
  - deterministic boundary changes from two-warp tree to one-warp pair-sum per head; activation must pass before timing is trusted
- Expected keep/revert boundary:
  - Keep only if remote activation passes and targeted `nsys` attribution improves over the retained 64-thread clean profile (`gated_readout_combine_forward_kernel_f_` below `11.793ms / 36`) without total compare regression.
  - Revert if activation drifts, compile fails, standard compare falls below the retained speedup range, or `nsys` shows the combine kernel is slower.
- Next command: edit only `crates/rwkv-nn/src/kernels/train/time_mixer/gated_readout_combine/{forward.rs,kernel.rs}` for the row-pack candidate, then run local compile/format checks without local GPU.

## Implementation edit

- Edited `crates/rwkv-nn/src/kernels/train/time_mixer/gated_readout_combine/forward.rs`:
  - Added a shape-specific row-pack dispatch for `num_heads == 12`.
  - Row-pack launch uses `CubeCount::Static(rows, 1, 1)` and `CubeDim::new_1d(384)`.
  - Existing 64-thread/two-warp per-head implementation remains as fallback for other shapes.
- Edited `crates/rwkv-nn/src/kernels/train/time_mixer/gated_readout_combine/kernel.rs`:
  - Added `gated_readout_combine_row_pack_forward_kernel`.
  - One warp handles one head; each lane accumulates up to two head lanes and writes up to two output lanes.
  - No shared memory or inter-warp synchronization is used.
- Next command: local compile check only, `cargo check -p rwkv-nn --features cuda,fusion`.

## Local compile check 1

- Command: `cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed in `11.80s`.
- No local GPU execution was performed.
- Next command: `cargo +nightly fmt --all`.

## Format

- Command: `cargo +nightly fmt --all`.
- Result: passed.
- Next command: rerun `cargo check -p rwkv-nn --features cuda,fusion` after formatting.

## Local compile check 2

- Command: `cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed in `11.02s`.
- No local GPU execution was performed.
- Next command: sync only this attempt's owned files to remote `10.100.1.253`, then run remote compile check.

## Remote sync attempt 1

- Command: `rsync -avR ... caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable/`.
- Result: failed with `Permission denied (publickey,password)` because the explicit SSH key was omitted.
- Validity: invalid sync attempt; no remote build or timing conclusion.
- Next command: rerun the same scoped sync with `-e "ssh -i ~/.ssh/id_ed25519_dgx_spark_windows"`.

## Remote sync attempt 2

- Command: scoped `rsync -avR -e 'ssh -i ~/.ssh/id_ed25519_dgx_spark_windows -o BatchMode=yes'` for this note plus `gated_readout_combine/forward.rs` and `kernel.rs`.
- Result: exited `0`; only the owned files were listed in rsync output.
- Destination: `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable/`.
- Next command: remote `cargo check -p rwkv-nn --features cuda,fusion`.

## Remote compile check

- Host: `caizus@10.100.1.253`.
- Command: `cd /home/caizus/Projects/Packages/rwkv-rs-stable && cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed in `2.49s`.
- Next command: remote standard compare with regenerated remote baseline:
  - `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1`.

## Remote compare 1

- Host: `caizus@10.100.1.253`.
- Command: standard compare with regenerated remote baseline, `repeat=3,warmup=1`.
- Build: release build completed in `2m 22s`.
- Activation result: passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: clean pass, `timing_summary compared=76 passed=76 failed=0 actual_total_ms=86.756 baseline_total_ms=175.887 speedup=2.03x`.
- Module timing signal:
  - `cells/*/time_mixer`: `54.923ms` actual vs `120.381ms` baseline, `2.19x`.
  - `cells/*/channel_mixer`: `20.584ms` actual vs `25.759ms` baseline, `1.25x`.
  - `loss/l2wrap_cross_entropy`: `4.438ms` actual vs `5.957ms`, `1.34x`.
- Interpretation: row-pack preserves activation and gives a clean standard timing pass, but total speedup is slightly below the retained 64-thread post-revert compare (`2.07x`) and near the retained clean-cache range. Need `nsys` to decide whether the targeted combine kernel improved.
- Next command: remote `nsys` with `repeat=1,warmup=1`, output `target/rwkv-test/nsys-gated-readout-rowpack`.

## Remote nsys

- Host: `caizus@10.100.1.253`.
- Command: `nsys profile --trace=cuda,nvtx --stats=true --force-overwrite=true -o target/rwkv-test/nsys-gated-readout-rowpack target/release/rwkv-test compare-rwkv-nn --color never --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 1 --warmup 1`.
- Result: exited nonzero because profiler mode used `repeat=1` against a `repeat=3` baseline, as expected. Activation passed; use only profiler attribution.
- Generated:
  - `/home/caizus/Projects/Packages/rwkv-rs-stable/target/rwkv-test/nsys-gated-readout-rowpack.nsys-rep`
  - `/home/caizus/Projects/Packages/rwkv-rs-stable/target/rwkv-test/nsys-gated-readout-rowpack.sqlite`
- `cuda_gpu_kern_sum` key rows:
  - `gated_readout_combine_row_pack_forward_kernel_f_`: `36` launches, `11.159ms` total, `0.310ms` avg.
  - retained 64-thread clean profile reference: `gated_readout_combine_forward_kernel_f_`, `36` launches, `11.793ms` total, `0.328ms` avg.
  - rejected warp32 profile reference: `gated_readout_combine_forward_kernel_f_`, `36` launches, `12.167ms` total, `0.338ms` avg.
  - adjacent surfaces: WKV7 `23.190ms`, mix6 `14.803ms`, channel-mixer relu-square `14.070ms`, key-prepare `9.342ms`.
- Interpretation: targeted attribution is positive: row-pack improves the combine kernel by about `0.634ms` total, or roughly `5.4%`, versus the retained 64-thread profile. Standard compare remained clean, but total compare did not clearly exceed the retained post-revert run; run a launch-config query before the keep decision.
- Next command: query the new SQLite for `gated_readout_combine_row_pack_forward_kernel_f_` launch geometry and registers.

## SQLite launch query attempt 1

- Command: remote Python `sqlite3` query against `target/rwkv-test/nsys-gated-readout-rowpack.sqlite`.
- Result: invalid query. Shell quoting stripped the SQL string quotes around the `LIKE` pattern and produced `sqlite3.OperationalError: near "%": syntax error`.
- Next command: rerun the same query with a parameterized `LIKE ?` clause.

## SQLite launch query attempt 2

- Query result: `gated_readout_combine_row_pack_forward_kernel_f_`, `36` launches, `11.15936ms` total, `309.982us` avg, `grid=(8192,1,1)`, `block=(384,1,1)`, `registers/thread=26`, no static or dynamic shared memory.
- Interpretation: the row-pack kernel is active with the intended launch shape. It uses the same register count as the rejected warp32 per-head version, but block count is `12x` lower and targeted total time improves over the retained 64-thread profile.
- Next command: run one more standard remote compare with `repeat=3,warmup=1` to check steady-state total timing before deciding keep/revert.

## Remote compare 2

- Host: `caizus@10.100.1.253`.
- Command: standard compare with regenerated remote baseline, `repeat=3,warmup=1`.
- Activation result: passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: clean pass, `timing_summary compared=76 passed=76 failed=0 actual_total_ms=87.846 baseline_total_ms=175.887 speedup=2.00x`.
- Module timing signal:
  - `cells/*/time_mixer`: `56.422ms` actual vs `120.381ms` baseline, `2.13x`.
  - `cells/*/channel_mixer`: `20.302ms` actual vs `25.759ms` baseline, `1.27x`.
  - `loss/l2wrap_cross_entropy`: `4.327ms` actual vs `5.957ms`, `1.38x`.
- Decision: reject row-pack as an end-to-end performance change. It improves the targeted combine kernel in `nsys`, but both standard compares (`2.03x`, then `2.00x`) are below the retained 64-thread post-revert state (`2.07x`) and do not show a usable total improvement. Keep this note as partial positive kernel-level evidence and revert the live code to the 64-thread/two-warp combine.
- Next edit: revert only `crates/rwkv-nn/src/kernels/train/time_mixer/gated_readout_combine/{forward.rs,kernel.rs}` to remove row-pack dispatch and kernel code.

## Revert edit

- Reverted `crates/rwkv-nn/src/kernels/train/time_mixer/gated_readout_combine/forward.rs`:
  - Removed `ROW_PACK_NUM_HEADS`, `ROW_PACK_BLOCK_SIZE`, and the shape-specific row-pack dispatch.
  - Restored the always-on 64-thread/two-warp per-head launch.
- Reverted `crates/rwkv-nn/src/kernels/train/time_mixer/gated_readout_combine/kernel.rs`:
  - Removed `gated_readout_combine_row_pack_forward_kernel`.
  - Retained the original two-warp shared-memory reduction kernel.
- Next command: local `cargo check -p rwkv-nn --features cuda,fusion` without local GPU.

## Revert compile check

- Command: `cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed in `11.81s`.
- No local GPU execution was performed.
- Next command: `cargo +nightly fmt --all`.

## Revert format

- Command: `cargo +nightly fmt --all`.
- Result: passed.
- Next command: rerun local `cargo check -p rwkv-nn --features cuda,fusion`, then sync the reverted files and note to remote.

## Revert compile check 2

- Command: `cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed in `4.47s`.
- No local GPU execution was performed.
- Next command: scoped sync of reverted `gated_readout_combine` files and this note to remote, then remote compile check.

## Revert remote sync

- Command: scoped `rsync -avR -e 'ssh -i ~/.ssh/id_ed25519_dgx_spark_windows -o BatchMode=yes'` for this note plus reverted `gated_readout_combine/forward.rs` and `kernel.rs`.
- Result: exited `0`.
- Next command: remote `cargo check -p rwkv-nn --features cuda,fusion`.

## Revert remote compile check

- Host: `caizus@10.100.1.253`.
- Command: `cd /home/caizus/Projects/Packages/rwkv-rs-stable && cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed in `2.41s`.
- Next command: one standard remote compare to prove the remote run copy is restored to the retained 64-thread GatedReadout combine state.

## Post-revert remote compare

- Host: `caizus@10.100.1.253`.
- Command: standard compare with regenerated remote baseline, `repeat=3,warmup=1`.
- Build: release build completed in `2m 25s`.
- Activation result: passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: positive but not clean, `timing_summary compared=76 passed=75 failed=1 actual_total_ms=86.004 baseline_total_ms=175.887 speedup=2.05x`.
- Failed row: known unrelated marginal row `timing/cells/cell_0000/channel_mixer.time.json`, `1.847ms` actual vs `1.668ms` baseline, `0.90x`.
- Module timing signal:
  - `cells/*/time_mixer`: `55.028ms` actual vs `120.381ms`, `2.19x`.
  - `cells/*/channel_mixer`: `20.115ms` actual vs `25.759ms`, `1.28x`.
  - `loss/l2wrap_cross_entropy`: `4.478ms` actual vs `5.957ms`, `1.33x`.
- Final decision: row-pack remains rejected and reverted. The remote live copy is restored to the retained 64-thread/two-warp GatedReadout combine. The only post-revert failure is the previously characterized channel-mixer edge outside this attempt boundary; total speedup remains above `1.0`.
