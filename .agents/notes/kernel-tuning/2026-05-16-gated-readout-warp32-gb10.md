# Gated Readout Combine Warp32 on GB10

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-gated-readout-warp32-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this branch inherits the positive but not clean `gated_readout_combine` forward kernel, broad unrelated workspace changes, and prior notes. This attempt only changes the GatedReadout combine forward kernel launch geometry/algorithm plus this note.
- Prior-note search command:
  - `rg -n "gated_readout_combine|warp32|one-warp|num_warps|block=\\(64|block\\(64|block 64|HEAD_SIZE=64|forward combine" .agents/notes/kernel-tuning crates/rwkv-nn/src/kernels/train/time_mixer/gated_readout_combine -S`
- Matched prior evidence:
  - `2026-05-16-gated-readout-forward-combine-gb10.md`: the current 64-thread/two-warp combine candidate passes activation and reaches clean-cache standard compare `2.05x-2.14x`, but remains `75/76` because of the known unrelated `cell_0000/channel_mixer` edge.
  - Clean `nsys` for that candidate shows `gated_readout_combine_forward_kernel_f_` launched 36 times with `grid=(8192,12,1)`, `block=(64,1,1)`, `registers/thread=21`, `dynamic_smem=8`, total `11.793ms`, avg `327.595us`.
  - The prior profile confirms the kernel replaces a generic binop/reduce surface after WKV7 and before output projection.
- Machine/GPU: remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`. Do not run local GPU.
- Remote run directory: `/home/caizus/Projects/Packages/rwkv-rs-stable`, synchronized non-git copy. Branch provenance remains local.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, heads `12`, head size `64`, regenerated remote baseline under `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Kernel/stage: TimeMixer GatedReadout combine forward.
- Hypothesis: use one warp per `(row, head)` instead of two warps. Each thread computes two reduction lanes and writes two output lanes, so the head reduction stays within a single warp and avoids shared-memory inter-warp reduction and cube syncs.
- Candidate parameters:
  - `HEAD_SIZE=64`
  - `BLOCK_SIZE=32`
  - no shared memory
  - vector width by logical lane pair: two head lanes per thread
  - deterministic order changes from 64-lane two-warp tree to 32-lane pair-sum plus warp tree; correctness must be trace-guarded before using timing.
- Expected keep/revert boundary:
  - Keep only if remote activation passes and standard remote compare is at least as good as the 64-thread candidate (`>2.05x` clean-cache warm run or better targeted GatedReadout kernel attribution) without introducing new failing rows beyond the known channel-mixer edge.
  - Revert or close as negative if activation drifts, compile fails, or `nsys` shows the combine kernel gets slower than the 64-thread total/avg.
- Next command: edit only `crates/rwkv-nn/src/kernels/train/time_mixer/gated_readout_combine/{kernel.rs,forward.rs}` for the warp32 candidate, then run local compile/format checks without local GPU.

## Implementation edit

- Changed `gated_readout_combine_forward_kernel` from two-warps/block to one-warp/block:
  - removed the shared-memory two-warp reduction helper.
  - each thread accumulates `lane` and `lane + 32` when in range.
  - each thread writes the same two lanes after the warp reduction.
- Changed launch `BLOCK_SIZE` from `64` to `32`.
- No Autodiff fallback or module wiring changes.
- Next command: `cargo check -p rwkv-nn --features cuda,fusion`.

## Local compile check 1

- Command: `cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed in `11.34s`.
- No local GPU execution was performed.
- Next command: `cargo +nightly fmt --all`.

## Format

- Command: `cargo +nightly fmt --all`.
- Result: passed.
- Next command: rerun `cargo check -p rwkv-nn --features cuda,fusion` after formatting.

## Local compile check 2

- Command: `cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed in `4.71s`.
- No local GPU execution was performed.
- Next command: sync only this attempt's owned files to remote `10.100.1.253`.

## Remote sync

- Command: explicit-file `rsync -avR` to `/home/caizus/Projects/Packages/rwkv-rs-stable/` for:
  - `.agents/notes/kernel-tuning/2026-05-16-gated-readout-warp32-gb10.md`
  - `crates/rwkv-nn/src/kernels/train/time_mixer/gated_readout_combine/kernel.rs`
  - `crates/rwkv-nn/src/kernels/train/time_mixer/gated_readout_combine/forward.rs`
- Result: rsync exited `0`.
- Next command: remote `cargo check -p rwkv-nn --features cuda,fusion`.

## Remote compile check

- Host: `caizus@10.100.1.253`.
- Command: `cd /home/caizus/Projects/Packages/rwkv-rs-stable && cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed in `1.25s`.
- Next command: remote standard compare with regenerated remote baseline:
  - `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1`.

## Remote compare 1

- Host: `caizus@10.100.1.253`.
- Command: standard compare with regenerated remote baseline, `repeat=3,warmup=1`.
- Build: release build completed in `2m 19s`.
- Activation result: passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: clean pass, `timing_summary compared=76 passed=76 failed=0 actual_total_ms=85.812 baseline_total_ms=175.887 speedup=2.05x`.
- Module timing signal:
  - `cells/*/time_mixer`: `55.097ms` actual vs `120.381ms` baseline, `2.18x`.
  - `cells/*/channel_mixer`: `19.747ms` actual vs `25.759ms` baseline, `1.30x`.
  - `loss/l2wrap_cross_entropy`: `4.531ms` actual vs `5.957ms`, `1.31x`.
- Interpretation: the warp32 candidate preserves activation and produces a clean `76/76` timing pass. It also removes the previous marginal `cell_0000/channel_mixer` failure in this run. Because total speedup is similar to the 64-thread clean-cache warm run, profile the combine kernel before making a keep decision.
- Next command: remote `nsys` with `repeat=1,warmup=1` and output `target/rwkv-test/nsys-gated-readout-warp32`.

## Remote nsys

- Host: `caizus@10.100.1.253`.
- Command: `nsys profile --trace=cuda,nvtx --stats=true --force-overwrite=true -o target/rwkv-test/nsys-gated-readout-warp32 target/release/rwkv-test compare-rwkv-nn --color never --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 1 --warmup 1`.
- Result: exited nonzero because profiler mode used `repeat=1` against a `repeat=3` baseline, as expected. Activation passed; use only profiler attribution.
- Generated:
  - `/home/caizus/Projects/Packages/rwkv-rs-stable/target/rwkv-test/nsys-gated-readout-warp32.nsys-rep`
  - `/home/caizus/Projects/Packages/rwkv-rs-stable/target/rwkv-test/nsys-gated-readout-warp32.sqlite`
- Kernel summary:
  - `gated_readout_combine_forward_kernel_f_`: 36 launches, `12.167ms` total, `0.338ms` avg.
  - 64-thread reference from the previous clean profile was `11.793ms` total, `0.328ms` avg.
  - Other main surfaces are similar: WKV7 `22.421ms`, mix6 `14.724ms`, channel-mixer relu-square `14.060ms`.
- Interpretation: the standard compare is clean, but kernel-level attribution does not show the intended improvement. Check the sqlite launch config to confirm this profile used `block=(32,1,1)` before deciding keep/revert.
- Next command: query `target/rwkv-test/nsys-gated-readout-warp32.sqlite` for the combine launch config.

## SQLite Launch Query

- Launch config: `gated_readout_combine_forward_kernel_f_` launched 36 times with `grid=(8192,12,1)`, `block=(32,1,1)`, `registers/thread=26`, `static_smem=0`, `dynamic_smem=0`, total `12.167ms`, avg `337.973us`.
- Comparison against the 64-thread clean profile:
  - 64-thread: `block=(64,1,1)`, `registers/thread=21`, `dynamic_smem=8`, total `11.793ms`, avg `327.595us`.
  - warp32: `block=(32,1,1)`, `registers/thread=26`, `dynamic_smem=0`, total `12.167ms`, avg `337.973us`.
- Decision: reject warp32 as a targeted performance change. It passed standard compare, but the intended kernel got slower and used more registers. The `76/76` standard result likely reflects the known channel-mixer timing variance, not a causal improvement from this kernel.
- Keep/revert state: revert the warp32 code back to the 64-thread/two-warp combine implementation while preserving this note as negative evidence.
- Next command: revert only `crates/rwkv-nn/src/kernels/train/time_mixer/gated_readout_combine/{kernel.rs,forward.rs}` to the 64-thread combine implementation.

## Revert edit

- Restored the 64-thread/two-warp implementation:
  - `BLOCK_SIZE=64`.
  - `NUM_WARPS=2`.
  - restored shared-memory inter-warp reduction.
  - removed the two-lanes-per-thread warp32 path.
- Next command: local `cargo check -p rwkv-nn --features cuda,fusion` without local GPU.

## Revert compile check

- Command: `cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed in `10.12s`.
- No local GPU execution was performed.
- Next command: `cargo +nightly fmt --all`, then sync the reverted files and note to remote.

## Revert format

- Command: `cargo +nightly fmt --all`.
- Result: passed.
- Next command: sync the reverted 64-thread files and this note back to remote so `/home/caizus/Projects/Packages/rwkv-rs-stable` no longer carries the rejected warp32 code.

## Revert remote sync

- Command: explicit-file `rsync -avR` to `/home/caizus/Projects/Packages/rwkv-rs-stable/` for this note and the two `gated_readout_combine` source files.
- Result: rsync exited `0`.
- Next command: remote `cargo check -p rwkv-nn --features cuda,fusion` to confirm the remote copy after reverting warp32.

## Revert remote compile check

- Host: `caizus@10.100.1.253`.
- Command: `cd /home/caizus/Projects/Packages/rwkv-rs-stable && cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed in `1.34s`.
- Final decision: warp32 is rejected and reverted. The retained live implementation is the previous 64-thread/two-warp GatedReadout combine, which has trace-correct remote evidence and better direct `nsys` kernel attribution than warp32.

## Post-revert remote compare

- Branch/worktree note: this validation is a continuation of the rejected warp32 attempt after a branch change; the code boundary is the reverted 64-thread/two-warp GatedReadout combine already synced to `/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Reason: the remote run directory briefly carried the rejected warp32 code. After syncing the 64-thread revert and passing remote compile, run one standard compare to prove the live remote copy is back on an accuracy-valid, speedup-positive state.
- Host: `caizus@10.100.1.253`.
- Command to run: `cd /home/caizus/Projects/Packages/rwkv-rs-stable && cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1`.
- Expected boundary: activation must pass; timing must stay above `1.0x`. A marginal `cell_0000/channel_mixer` row failure is already known unrelated variance, but a broad time-mixer or GatedReadout regression would reopen the revert boundary.

### Result

- Build: release build completed in `2m 26s`.
- Activation result: passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: positive but not clean, `timing_summary compared=76 passed=75 failed=1 actual_total_ms=84.876 baseline_total_ms=175.887 speedup=2.07x`.
- Failed row: `timing/cells/cell_0000/channel_mixer.time.json`, `1.756ms` actual vs `1.668ms` baseline, `0.95x`.
- Module timing signal:
  - `cells/*/time_mixer`: `54.144ms` actual vs `120.381ms`, `2.22x`.
  - `cells/*/channel_mixer`: `19.845ms` actual vs `25.759ms`, `1.30x`.
  - `loss/l2wrap_cross_entropy`: `4.464ms` actual vs `5.957ms`, `1.33x`.
- Decision: the remote copy is back on the 64-thread/two-warp combine and remains accuracy-valid with a strong total speedup. The only failure is the previously characterized marginal first-cell channel-mixer row, outside the reverted warp32/GatedReadout boundary.
