# WKV7 Shared-State Lanes GB10

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-wkv7-shared-state-lanes-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits plus retained/reverted kernel tuning changes. This attempt owns only WKV7 shared-state prototype files and this note unless later entries explicitly expand scope.
- Prior-note search commands:
  - `rg -n "WKV7|wkv7|row_tile|time split|chunk|state handoff|scan|row-tile|shared-state|lanes_per_row" .agents/notes/kernel-tuning crates/rwkv-nn/src/kernels/train/time_mixer/wkv7 -S`
  - `rg -n "key_prepare|warps_per_cube|row_tile|WKV7|gated_readout|LayerNorm|ordered-256|channel_mixer|lm_head|LocalTuner|remote|10\\.100\\.1\\.253" .agents/notes/kernel-tuning crates/rwkv-nn/src/kernels/train -S`
- Matched prior evidence:
  - Current GB10 profile has `wkv7_pretrain_forward_output_kernel_f_bf16` at `22.034 ms / 36`, `grid=(12,16,1)`, `block=(64,1,1)`, regs `108`, dynamic smem `1536`.
  - Forced `row_tile=16` is a recorded negative and must not be repeated.
  - `2026-05-16-wkv7-state-scan-design-gb10.md` rejects full chunk scan/state-handoff as too large for the current pass and identifies a smaller intra-block shared-state candidate.
  - Other current large project-owned surfaces are duplicate-closed: channel-mixer forced Cube/Burn/reference/fusion, mix6 line-size/axis retry, lm-head target-logit/atomic/online-softmax/prune, GatedReadout warp32/row-pack, key-prepare warps.
- Machine/GPU: primary validation on remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`; do not use local GPU.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, heads `12`, head size `64`, `chunk_len=16`.
- Kernel/stage: `wkv7_pretrain_forward_output_kernel` replacement candidate for pretrain output only.
- Hypothesis: current WKV7 uses one lane per row and stores `Array<f32>(64)` in registers, leaving each lane to serially scan/update all 64 columns for each timestep. A shared-state kernel with `lanes_per_row=4` or `8` can split the per-row column work, reduce register pressure, and possibly improve GB10 WKV7 time, at the cost of shared memory and synchronization.
- Candidate parameters:
  - Keep existing register-row candidates `row_tile_{16,32,64}`.
  - Add shared-state candidates `shared_lanes_4` and `shared_lanes_8`, with block sizes `256` and `512`; `num_warps = block_size / 32`.
  - Candidate names encode the implementation so persistent autotune checksum changes.
- Accuracy risk: shared-state reductions change the order of the 64-column dot products versus the current serial row loop. Activation comparison decides whether the candidate is admissible. If it drifts, localize whether the drift comes from reduction order or implementation bug before reverting.
- Expected keep/revert boundary:
  - Keep only if local compile/format checks pass, remote compile passes, remote activation remains `54/54`, and targeted nsys shows WKV7 time improves without total speedup dropping below `1.0`.
  - If activation fails, inspect whether the shared candidate was selected and whether the code implements the same recurrence before reverting.
  - If shared candidates are slower, keep the branch/note as negative evidence and revert code to the existing register-row tuner.
- Next edit: add a `wkv7_pretrain_forward_output_shared_lanes_kernel` and wire it as extra `LocalTuner` candidates in `wkv7/forward.rs` without removing the current row-tile candidates.

## Implementation Edit

- Edited `crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/kernel.rs`.
  - Added `wkv7_pretrain_forward_output_shared_lanes_kernel`.
  - It stores the full `64 x 64` state matrix in shared memory, maps `lanes_per_row` lanes to each row, splits column dot/update work across those lanes, and reduces partial `state_replacement` / `output` through shared memory.
- Edited `crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/forward.rs`.
  - Added `PRETRAIN_OUTPUT_SHARED_LANES_CANDIDATES = [4, 8]`.
  - Added tuner candidates `shared_lanes_4` and `shared_lanes_8`.
  - Kept existing `row_tile_{16,32,64}` candidates.
  - Candidate filtering requires `head_size == 64`, `head_size % lanes_per_row == 0`, and `head_size * lanes_per_row <= max_units_per_cube`.
- Next command: run rustfmt on the two edited WKV7 files, then local non-GPU compile check `cargo check -p rwkv-nn --features cuda,fusion`.

## Local Non-GPU Checks

- Command: `rustup run nightly rustfmt crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/kernel.rs crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/forward.rs && cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed.
- Next command: sync the two edited WKV7 files and this note to remote `10.100.1.253`, remove only the WKV7 pretrain-output autotune cache entry so candidate timing is retuned, then run remote `cargo check -p rwkv-nn --features cuda,fusion`.

## Resume Preflight

- Resume date: 2026-05-16 after context compaction.
- Prior-note search rerun:
  - `rg -n "shared-lanes|shared_lanes|wkv7|state-scan|residual|ordered-256|LayerNorm|10\\.100\\.1\\.253|GB10" .agents/notes/kernel-tuning /root/.codex/memories/MEMORY.md 2>/dev/null`
- Matched evidence: no prior completed `shared_lanes_{4,8}` WKV7 result was found. The search did find the LayerNorm ordered-256 negative/keep notes, remote GB10 acceptance notes, residual duplicate guards, and the WKV7 state-scan design note that motivates this branch.
- Current branch confirmed: `kernel-tuning-wkv7-shared-state-lanes-gb10-20260516`.
- Dirty-tree constraint unchanged: the checkout still has broad unrelated edits; this continuation owns only `crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/{forward.rs,kernel.rs}` and this note.
- Boundary correction: current `kernel.rs` diff is broader than the initial output-only hypothesis. It also changes existing pretrain/state saved kernels to keep the state matrix in shared memory and write snapshots only at chunk boundaries. Treat activation failures as potentially caused by this implementation change, not only by `shared_lanes_{4,8}` reduction order.
- Next command: on remote `10.100.1.253`, run the standard `compare-rwkv-nn` gate with the regenerated BF16 baseline, then inspect the WKV7 autotune log and decide whether to keep, narrow, or revert this branch.

## Remote Sync Attempt

- Command: `rsync -av --relative ... caizus@10.100.1.253:~/Projects/Packages/rwkv-rs-stable/ && ssh ... compare-rwkv-nn ...`.
- Result: invalid, authentication failed with the default SSH identity before any sync/build/compare work ran.
- Next command: rerun the same sync and compare using the project remote identity `~/.ssh/id_ed25519_dgx_spark_windows`.

## Remote Standard Compare

- Command: `rsync -av -e 'ssh -i ~/.ssh/id_ed25519_dgx_spark_windows ...' --relative crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/{forward.rs,kernel.rs} .agents/notes/kernel-tuning/2026-05-16-wkv7-shared-state-lanes-gb10.md caizus@10.100.1.253:~/Projects/Packages/rwkv-rs-stable/ && ssh -i ~/.ssh/id_ed25519_dgx_spark_windows ... 'cd ~/Projects/Packages/rwkv-rs-stable && cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline ~/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1 | tee target/rwkv-test/remote-wkv7-shared-lanes-compare.log'`.
- Binary/build provenance: remote release binary rebuilt after the WKV7 code changes.
- Result: activation passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: `timing_summary compared=76 passed=74 failed=2 missing=0 extra=0 ignored=1 actual_total_ms=75.728 baseline_total_ms=175.887 speedup=2.32x`.
- Failing timing rows: `cells/cell_0000/channel_mixer` at `0.97x` and `cells/cell_0009/channel_mixer` at `0.99x`. WKV7/time_mixer rows all passed, and `cells/*/time_mixer` total was `44.080ms` vs `120.381ms` baseline, `2.73x`.
- Interpretation: this branch is accuracy-safe on remote GB10. The remaining standard-gate failures are channel-mixer noise/regression, not WKV7 drift.
- Next command: inspect the remote WKV7 autotune log to see whether `shared_lanes_{4,8}` won or whether an old row-tile candidate won after retune.

## Remote Autotune Log

- Command: `ssh -i ~/.ssh/id_ed25519_dgx_spark_windows ... 'cd ~/Projects/Packages/rwkv-rs-stable && find target/autotune -name "*wkv7*pretrain*output*forward*.log" -print -exec tail -n 120 {} \\;'`.
- Result path: `target/autotune/0.10.0/device-0-0-cuda/rwkv_nn-kernels-train-time_mixer-wkv7-forward-wkv7-pretrain-output-forward.json.log`.
- Selected candidate: `row_tile_64` (`fastest_index=2`, name `row_tile_64`).
- Candidate timings:
  - `row_tile_64`: median `598085ns`, mean `597355ns`.
  - `row_tile_32`: median `925257ns`, mean `926190ns`.
  - `row_tile_16`: median `1691279ns`, mean `1691700ns`.
  - `shared_lanes_8`: median `2894042ns`, mean `2894489ns`.
  - `shared_lanes_4`: median `3632705ns`, mean `3694290ns`.
- Interpretation: shared-lanes is a clear performance negative on GB10 for this shape. It passed activation because it was either mathematically acceptable or not selected for the final path, but the tuner data rejects it as an implementation candidate.
- Decision before edit: remove `shared_lanes_{4,8}` and the broad shared-state saved/state kernel edits from this branch. Keep the note as negative evidence. The next code state should return to the existing row-tile/output path only, then rerun compile and remote compare if the diff changes.

## Shared-Lanes Revert

- Edited `crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/forward.rs`.
  - Removed `PRETRAIN_OUTPUT_SHARED_LANES_CANDIDATES`.
  - Removed `shared_lanes_{4,8}` tunables, their validity helper, and the launch helper.
  - Kept the row-tile output tuner and `row_tile_{16,32,64}` candidates.
- Edited `crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/kernel.rs`.
  - Removed `wkv7_pretrain_forward_output_shared_lanes_kernel`.
  - Reverted the broad saved pretrain/state kernels back to snapshot-backed state reads and per-step snapshot writes, leaving this attempt scoped to output row-tile tuning only.
- Post-edit duplicate check: `rg -n "shared_lanes|shared_state|wkv7_pretrain_forward_output_shared|lanes_per_row|is_valid_pretrain_output_shared" crates/rwkv-nn/src/kernels/train/time_mixer/wkv7` found no matches.
- Local non-GPU check command: `rustup run nightly rustfmt crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/kernel.rs crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/forward.rs && cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed.
- Next command: sync the narrowed WKV7 files and this note to remote, remove only the WKV7 pretrain-output autotune cache entry because the candidate checksum changed, then run remote standard compare again.

## Remote Compare After Narrowing

- Command: `rsync -av -e 'ssh -i ~/.ssh/id_ed25519_dgx_spark_windows ...' --relative crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/{forward.rs,kernel.rs} .agents/notes/kernel-tuning/2026-05-16-wkv7-shared-state-lanes-gb10.md caizus@10.100.1.253:~/Projects/Packages/rwkv-rs-stable/ && ssh -i ~/.ssh/id_ed25519_dgx_spark_windows ... 'cd ~/Projects/Packages/rwkv-rs-stable && rm -f target/autotune/0.10.0/device-0-0-cuda/rwkv_nn-kernels-train-time_mixer-wkv7-forward-wkv7-pretrain-output-forward.json.log && cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline ~/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1 | tee target/rwkv-test/remote-wkv7-rowtile-narrow-compare.log'`.
- Binary/build provenance: remote release binary rebuilt after removing shared-lanes code and deleting only the WKV7 pretrain-output autotune cache.
- Result: activation passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: `timing_summary compared=76 passed=75 failed=1 missing=0 extra=0 ignored=1 actual_total_ms=72.922 baseline_total_ms=175.887 speedup=2.41x`.
- Remaining failing row: `cells/cell_0000/channel_mixer.time.json`, `1.727ms` vs `1.668ms`, `0.97x`.
- WKV7/time_mixer result: all time-mixer rows passed; `cells/*/time_mixer` was `42.629ms` vs `120.381ms`, `2.82x`.
- Interpretation: removing the slow shared-lanes candidates improved the total from `75.728ms` to `72.922ms` and the time-mixer group from `44.080ms` to `42.629ms`. Keep shared-lanes reverted.
- Next command: inspect the narrowed WKV7 autotune log and then run a short nsys profile to attribute selected WKV7 kernels if the log confirms `row_tile_64`.

## Narrowed Autotune Log

- Command: `ssh -i ~/.ssh/id_ed25519_dgx_spark_windows ... 'cd ~/Projects/Packages/rwkv-rs-stable && tail -n 120 target/autotune/0.10.0/device-0-0-cuda/rwkv_nn-kernels-train-time_mixer-wkv7-forward-wkv7-pretrain-output-forward.json.log'`.
- Selected candidate: `row_tile_64`, `fastest_index=2`.
- Candidate timings:
  - `row_tile_64`: median `597990ns`, mean `597807ns`.
  - `row_tile_32`: median `925608ns`, mean `925203ns`.
  - `row_tile_16`: median `1695168ns`, mean `1699798ns`.
- Interpretation: current best WKV7 pretrain-output implementation for remote GB10 and this shape is still one unit per row with `row_tile=64`. Shared-lanes should stay out of the candidate set.
- Next command: run short remote `nsys` with `repeat=1,warmup=1` and export/query SQLite for `wkv7_pretrain_forward_output_kernel` launch count and total duration. The nsys compare timing itself is instrumentation-only and not a speedup gate.

## Remote Nsys After Narrowing

- Command: `nsys profile --trace=cuda,nvtx --sample=none --force-overwrite=true --output=target/rwkv-test/nsys-wkv7-rowtile-current target/release/rwkv-test compare-rwkv-nn --color never --baseline ~/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 1 --warmup 1`.
- First query attempt result: invalid command chain for attribution because `compare-rwkv-nn` returns nonzero under `repeat=1,warmup=1` timing-profile mismatch against a `repeat=3,warmup=1` baseline. The profile file was still generated.
- Export/query recovery command: `nsys export --type sqlite --force-overwrite=true --output target/rwkv-test/nsys-wkv7-rowtile-current.sqlite target/rwkv-test/nsys-wkv7-rowtile-current.nsys-rep`, then SQLite grouped kernel durations by `StringIds.value`.
- Result: `wkv7_pretrain_forward_output_kernel_f_bf16` launched `36` times, total `22.083ms`, avg `613.428us`.
- Top project-owned kernels in the same profile:
  - `wkv7_pretrain_forward_output_kernel_f_bf16`: `22.083ms / 36`.
  - `channel_mixer_relu_square_forward_kernel_f__n_2`: `14.699ms / 36`.
  - `mix6_forward_kernel_f__n_1`: `14.517ms / 36`.
  - `lm_head_l2wrap_ce_forward_row_kernel_f__i_i32`: `12.758ms / 3`.
  - `gated_readout_combine_forward_kernel_f_`: `11.964ms / 36`.
  - `key_prepare_forward_64_kernel_f_`: `9.567ms / 36`.
- Interpretation: the narrowed row-tile code does not materially reduce WKV7 profiler time versus the prior current profile (`22.034ms / 36`). The shared-lanes branch is a negative WKV7 attempt, not a keepable optimization.
- Next command: collect a focused `ncu` report for `wkv7_pretrain_forward_output_kernel_f_bf16` on remote GB10 to record achieved occupancy, warp behavior, register/shared-memory pressure, and memory throughput before closing this branch.

## Remote NCU Attempt

- Command: `command -v ncu && ncu --target-processes all --kernel-name-base function --kernel-name wkv7_pretrain_forward_output_kernel_f_bf16 --launch-count 3 --set speed-of-light --csv --log-file target/rwkv-test/ncu-wkv7-rowtile-current.csv ...`.
- Result: invalid, no CSV generated because `ncu` is not on the remote login PATH.
- Tool lookup: `which ncu || true; ls -l /usr/local/cuda*/bin/ncu /opt/nvidia/nsight-compute/*/ncu` found `/usr/local/cuda/bin/ncu`, `/usr/local/cuda-13/bin/ncu`, `/usr/local/cuda-13.0/bin/ncu`, and `/opt/nvidia/nsight-compute/2025.3.1/ncu`.
- Next command: rerun the same focused capture with `/usr/local/cuda/bin/ncu`.

## Remote NCU Set Lookup

- Command: `/usr/local/cuda/bin/ncu --target-processes all --kernel-name-base function --kernel-name wkv7_pretrain_forward_output_kernel_f_bf16 --launch-count 3 --set speed-of-light --csv --log-file target/rwkv-test/ncu-wkv7-rowtile-current.csv ...`.
- Result: invalid, `ncu` started but reported `No metrics to collect found in sections`; no useful WKV7 metrics were captured.
- Set lookup command: `/usr/local/cuda/bin/ncu --list-sets`.
- Available sets include `basic`, `detailed`, `full`, `nvlink`, `pmsampling`, and `roofline`; the correct lightweight section set for this remote version is `basic`, which includes `LaunchStats`, `Occupancy`, `SpeedOfLight`, and `WorkloadDistribution`.
- Next command: rerun focused WKV7 capture with `--set basic`.

## Remote NCU Permission Boundary

- Command: `/usr/local/cuda/bin/ncu --target-processes all --kernel-name-base function --kernel-name wkv7_pretrain_forward_output_kernel_f_bf16 --launch-count 3 --set basic --csv --log-file target/rwkv-test/ncu-wkv7-rowtile-current.csv ...`.
- Result: blocked by remote GPU counter permission: `ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0`.
- Consequence: achieved occupancy, warp execution efficiency, and hardware memory-throughput counters are not available to this user on `10.100.1.253` until the host enables NVIDIA performance counter access or the command is run with the required privileges.
- Fallback static launch data from existing nsys SQLite:
  - Kernel: `wkv7_pretrain_forward_output_kernel_f_bf16`.
  - Grid: `(12,16,1)`.
  - Block: `(64,1,1)`.
  - Registers/thread: `108`.
  - Static shared memory: `0`.
  - Dynamic shared memory: `1536` bytes.
  - Average duration: `613.428us` across `36` launches.
- Final decision for this branch: shared-lanes is rejected. Keep the code narrowed to row-tile output tuner only, with `row_tile_64` selected on GB10. Do not claim a WKV7 improvement from this attempt; the WKV7 output kernel remains a top hotspot at about `22ms / 36` launches and needs a different algorithmic direction.
