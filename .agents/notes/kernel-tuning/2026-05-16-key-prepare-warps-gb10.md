# Key Prepare Warps Per Cube GB10

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-key-prepare-warps-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this branch inherits broad unrelated workspace changes and the retained TimeMixer gate wiring / 64-thread GatedReadout combine. This attempt owns only `key_prepare` forward launch-packing code plus this note unless later entries explicitly expand scope.
- Prior-note/source search command:
  - `rg -n "key_prepare|KeyPrepare|key prepare|replacement|removal_key" .agents/notes/kernel-tuning crates/rwkv-nn/src/kernels/train/time_mixer/key_prepare crates/rwkv-nn/src/modules/time_mixer -S`
  - `rg -n "10\\.100\\.1\\.253|remote|GB10|channel_mixer|LayerNorm|ordered-256|residual|gated_readout|WKV7|lm_head|LocalTuner|autotune" .agents/notes/kernel-tuning /root/.codex/memories/MEMORY.md`
- Matched prior evidence:
  - `2026-05-16-remote-next-surface-gb10.md`: current newest live-state remote profile has `key_prepare_forward_64_kernel_f_` at `9.444ms / 36`, launch `grid=(24576,1,1)`, `block=(128,1,1)`, `regs=24`, no shared memory.
  - `2026-05-16-autotune-key-audit.md`: key_prepare was previously classified as having no candidate set, so it was not an autotune-key gap until a second launch configuration is introduced.
  - `2026-05-16-time-mixer-ncu.md`: local ncu showed key_prepare is bandwidth-heavy and no-spill; do not do line-size microtuning. This branch changes launch packing/warps-per-cube on GB10, not line width.
  - Duplicate guards: residual-add/Burn-add A/B, channel-mixer forced Cube/Burn-reference/fusion, LocalTuner bypass, WKV7 forced `row_tile=16`, lm-head target-logit/atomic/online-softmax/prune-256, and GatedReadout warp32 are closed.
- Machine/GPU: remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`. Do not run local GPU.
- Remote run directory: `/home/caizus/Projects/Packages/rwkv-rs-stable`, synchronized non-git copy.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, heads `12`, head size `64`, regenerated remote baseline under `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Kernel/stage: `key_prepare_forward_64_kernel`, forward path only.
- Hypothesis: the current fixed `HEAD64_WARPS_PER_CUBE=4` (`block=128`) may not be best on GB10. Since each warp owns one independent head vector and uses no inter-warp communication, `warps_per_cube` is a pure launch-packing/block-size parameter that should be selected by hardware/shape dispatch rather than a hard-coded constant.
- Candidate parameters for this first implementation:
  - `warps_per_cube=1` -> `block=32`
  - `warps_per_cube=2` -> `block=64`
  - `warps_per_cube=4` -> `block=128` current baseline
  - `warps_per_cube=8` -> `block=256`
- Expected keep/revert boundary:
  - Keep only if remote activation passes and standard remote compare remains speedup `>1.0`, with profiler evidence showing the selected key-prepare launch is not slower than current `9.444ms / 36`.
  - If all alternatives are slower, revert to the fixed `4`-warp path or leave only a hardware-key scaffold if it has no hot-path cost and preserves the current winner.
- Next command: inspect existing `LocalTuner` patterns for multi-output forward kernels; if the generic tuner path is too invasive for this attempt, implement a narrow runtime dispatch candidate with explicit remote A/B runs recorded before deciding whether to convert it into full autotune.

## LocalTuner Pattern Inspection

- Inspected `mix6/forward.rs`, `value_residual_gate/forward.rs`, `layer_norm/forward.rs`, and `rg LocalTuner` across train kernels.
- Result: multi-output autotune is already supported by `Mix6ForwardPrimitiveOutput` implementing `AutotuneOutput`; `KeyPrepareForwardPrimitiveOutput` can use the same pattern by checking its three output tensors.
- Implementation choice: add a real `LocalTuner` for the `head_size=64` path, keyed by runtime, dtype, shape, rows, head count, hardware fingerprint, and deterministic flag. Candidate parameter is `warps_per_cube`, represented in tunable names (`warps_per_cube_1/2/4/8`) so persistent cache checksum changes if the candidate set changes.
- Non-64 `head_size` remains on the existing generic fallback path without autotune.
- Next edit: update only `crates/rwkv-nn/src/kernels/train/time_mixer/key_prepare/forward.rs` to replace the fixed `HEAD64_WARPS_PER_CUBE=4` path with the tuner-backed candidate set.

## Implementation Edit

- Edited `crates/rwkv-nn/src/kernels/train/time_mixer/key_prepare/forward.rs`.
- Added `KeyPrepareForwardAutotuneKey` with runtime, dtype, shape, rows, head count/size, hardware fingerprint, and deterministic fields.
- Added `AutotuneOutput` for the three-output `KeyPrepareForwardPrimitiveOutput`.
- Replaced fixed `HEAD64_WARPS_PER_CUBE=4` with candidates `[1, 2, 4, 8]` under `local_tuner!("key-prepare-forward-64")`.
- Candidate names encode `warps_per_cube_*`; invalid candidates are filtered by max units/cube and shape constraints.
- Non-64 fallback is unchanged.
- Next command: run local compile check only, `cargo check -p rwkv-nn --features cuda,fusion`; no local GPU execution.

## Local Compile Check 1

- Command: `cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed in `13.47s`.
- No local GPU execution was performed.
- Next command: run `cargo +nightly fmt --all`, then rerun the same compile check after formatting.

## Format

- Command: `cargo +nightly fmt --all`.
- Result: passed.
- Next command: rerun `cargo check -p rwkv-nn --features cuda,fusion`.

## Local Compile Check 2

- Command: `cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed in `10.37s`.
- No local GPU execution was performed.
- Next command: sync only `crates/rwkv-nn/src/kernels/train/time_mixer/key_prepare/forward.rs` and this note to remote `10.100.1.253`, then run remote compile check.

## Remote Sync

- Command: scoped `rsync -avR` of:
  - `crates/rwkv-nn/src/kernels/train/time_mixer/key_prepare/forward.rs`
  - `.agents/notes/kernel-tuning/2026-05-16-key-prepare-warps-gb10.md`
- Destination: `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable/`.
- Result: rsync exited `0`.
- Next command: remote `cargo check -p rwkv-nn --features cuda,fusion`.

## Remote Compile Check

- Host: `caizus@10.100.1.253`.
- Command: `cd /home/caizus/Projects/Packages/rwkv-rs-stable && cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed in `2.74s`.
- Next command: run one standard remote compare as retune/warm-cache setup for the new `key-prepare-forward-64` tuner, then run a second standard compare for the decision.

## Remote Compare 1 Retune

- Host: `caizus@10.100.1.253`.
- Command: standard compare with regenerated remote baseline, output captured in `target/rwkv-test/remote-key-prepare-warps-compare-retune.log`.
- Role: first run after adding `local_tuner!("key-prepare-forward-64")`; treat as retune/warm-cache setup, not final keep decision.
- Build: release build completed in `2m 29s`.
- Activation result: passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: clean, `timing_summary compared=76 passed=76 failed=0 actual_total_ms=83.650 baseline_total_ms=175.887 speedup=2.10x`.
- Module timing signal: `cells/*/time_mixer` `53.523ms` vs baseline `120.381ms`, `2.25x`; `cells/*/channel_mixer` `19.532ms`, `1.32x`.
- Next command: inspect the new remote key-prepare autotune cache/log to see which `warps_per_cube` candidate won, then run the second standard compare without clearing cache.

## Remote Autotune Log Attempt 1

- Command: remote `find target/autotune ... | rg ...` and `tail` loop.
- Result validity: invalid shell command. Remote zsh environment does not have `rg`, and the echo marker was parsed badly by zsh. No tuning conclusion from this attempt.
- Next command: rerun with POSIX `grep`/`printf` and explicit quoted file paths.

## Remote Autotune Log

- Log path: `target/autotune/0.10.0/device-0-0-cuda/rwkv_nn-kernels-train-time_mixer-key_prepare-forward-key-prepare-forward-64.json.log`.
- Key recorded: CUDA BF16, `num_elements=8388608`, `B=16`, `T=512`, `D=768`, rows `8192`, heads `12`, head size `64`, GB10 hardware fingerprint (`load_width=128`, `plane_size=32`, `max_units_per_cube=1024`, `SMs=48`, deterministic).
- Candidate result:
  - `warps_per_cube_2`: fastest index, mean `269.684us`, median `269.730us`.
  - `warps_per_cube_4`: mean `270.575us`, median `269.746us`.
  - `warps_per_cube_8`: mean `273.801us`, median `269.682us`.
  - `warps_per_cube_1`: mean `271.581us`, median `271.826us`.
- Interpretation: the tuner selected `warps_per_cube=2`, but the direct candidate deltas are tiny and near noise. Need warm-cache compare plus `nsys` to see whether the launch config actually changes from `block=(128,1,1)` to `block=(64,1,1)` and whether aggregate key-prepare time improves.
- Next command: second standard remote compare without clearing cache.

## Remote Compare 2 Warm Cache

- Host: `caizus@10.100.1.253`.
- Command: standard compare with regenerated remote baseline, output captured in `target/rwkv-test/remote-key-prepare-warps-compare-warm.log`.
- Build: release binary was already up to date; Cargo finished in `0.19s`.
- Activation result: passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: clean, `timing_summary compared=76 passed=76 failed=0 actual_total_ms=85.836 baseline_total_ms=175.887 speedup=2.05x`.
- Module timing signal: `cells/*/time_mixer` `54.928ms`, `2.19x`; `cells/*/channel_mixer` `19.866ms`, `1.30x`.
- Interpretation: warm-cache compare remains correctness-clean and speedup-positive. The total is similar to the retained 64-thread GatedReadout state, so keep/revert depends on targeted `nsys` attribution for `key_prepare_forward_64_kernel`.
- Next command: run remote `nsys` profiler with `repeat=1,warmup=1`, output `target/rwkv-test/nsys-key-prepare-warps-gb10`, and query the key-prepare launch config/time.

## Remote Nsys

- Host: `caizus@10.100.1.253`.
- Command: `nsys profile --trace=cuda,nvtx --stats=true --force-overwrite=true -o target/rwkv-test/nsys-key-prepare-warps-gb10 target/release/rwkv-test compare-rwkv-nn --color never --baseline ... --repeat 1 --warmup 1`.
- Result: exited nonzero because profiler mode uses `repeat=1` against a `repeat=3` baseline; expected timing-profile mismatch. Activation still passed in the profiled run, and the emitted rough total stayed positive (`actual_total_ms=84.398`, speedup `2.08x`). Use only CUDA profiler data.
- Generated:
  - `target/rwkv-test/nsys-key-prepare-warps-gb10.nsys-rep`
  - `target/rwkv-test/nsys-key-prepare-warps-gb10.sqlite`
- `cuda_gpu_kern_sum` key row: `key_prepare_forward_64_kernel_f_`, `36` launches, `9.487ms` total, `263.532us` avg.
- Previous retained 64-thread/reference profiles were about `9.437ms` to `9.444ms` total for the same kernel family.
- Interpretation so far: profiler summary does not show a targeted improvement. Need launch-config query to confirm whether the selected `warps_per_cube=2` produced `block=(64,1,1)` and to compare registers/shared memory against the old `block=(128,1,1)` profile before deciding keep/revert.
- Next command: query `target/rwkv-test/nsys-key-prepare-warps-gb10.sqlite` for `key_prepare_forward_64_kernel_f_` launch geometry.

## SQLite Launch Query

- New tuned profile: `key_prepare_forward_64_kernel_f_`, `36` launches, `9.487ms` total, `263.532us` avg, `grid=(49152,1,1)`, `block=(64,1,1)`, `regs=24`, no shared memory.
- Previous `nsys-value-residual-gate-vector-axis` profile: `36` launches, `9.444ms` total, `262.340us` avg, `grid=(24576,1,1)`, `block=(128,1,1)`, `regs=24`, no shared memory.
- Previous `nsys-gated-readout-forward-combine-clean` profile: `36` launches, `9.437ms` total, `262.128us` avg, `grid=(24576,1,1)`, `block=(128,1,1)`, `regs=24`, no shared memory.
- Interpretation: the tuner did select `warps_per_cube=2`, but it doubled the block count and did not improve targeted kernel time. Register/shared-memory pressure is unchanged, so the likely reason is launch/occupancy packing noise or lower work-per-block efficiency rather than resource-pressure relief.
- Decision: reject this performance change. Revert the live code to fixed `HEAD64_WARPS_PER_CUBE=4` and keep this note as negative evidence. Do not keep a tuner that selects a slightly slower candidate on the target GB10 path.
- Next edit: revert only `crates/rwkv-nn/src/kernels/train/time_mixer/key_prepare/forward.rs` to the previous fixed 4-warps path.

## Revert Edit

- Reverted `crates/rwkv-nn/src/kernels/train/time_mixer/key_prepare/forward.rs` to fixed `HEAD64_WARPS_PER_CUBE=4`.
- Removed the `key-prepare-forward-64` `LocalTuner`, autotune key, candidate set, and multi-output `AutotuneOutput` implementation from this branch.
- Next command: local compile check `cargo check -p rwkv-nn --features cuda,fusion`; no local GPU execution.

## Revert Compile Check

- Command: `cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed in `12.42s`.
- No local GPU execution was performed.
- Next command: `cargo +nightly fmt --all`, then sync the reverted source and note to remote.

## Revert Format

- Command: `cargo +nightly fmt --all`.
- Result: passed.
- Next command: sync reverted `key_prepare/forward.rs` and this note to remote, then run remote compile check.

## Revert Remote Sync And Compile

- Command: scoped `rsync -avR` of reverted `crates/rwkv-nn/src/kernels/train/time_mixer/key_prepare/forward.rs` and this note, then remote `cargo check -p rwkv-nn --features cuda,fusion`.
- Result: rsync exited `0`; remote compile passed in `2.48s`.
- Next command: run one post-revert standard remote compare so the remote release binary is rebuilt from the restored fixed-4 source and the run directory is left in a known speedup-positive state.

## Post-Revert Remote Compare

- Host: `caizus@10.100.1.253`.
- Command: standard compare with regenerated remote baseline, output captured in `target/rwkv-test/remote-key-prepare-warps-post-revert-compare.log`.
- Build: release rebuild completed in `2m 25s`, proving the remote release binary no longer contains the rejected `key-prepare-forward-64` tuner code.
- Activation result: passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: positive but not clean, `timing_summary compared=76 passed=75 failed=1 actual_total_ms=86.865 baseline_total_ms=175.887 speedup=2.02x`.
- Failed row: known marginal unrelated row `timing/cells/cell_0000/channel_mixer.time.json`, `1.693ms` actual vs `1.668ms` baseline, `0.99x`.
- Module timing signal: `cells/*/time_mixer` `55.378ms`, `2.17x`; `cells/*/channel_mixer` `20.126ms`, `1.28x`.
- Final decision: reject and revert the key-prepare warps-per-cube tuner. The attempted dispatch dimension is documented, but on GB10 it selected `warps_per_cube=2`, doubled the block count, and made targeted profiler time slightly worse. The live remote copy is restored to fixed `HEAD64_WARPS_PER_CUBE=4` and remains activation-valid with total speedup above `1.0`.
