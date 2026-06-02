# Weight Decay Transform GB10

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-weight-decay-transform-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this branch inherits broad unrelated workspace changes plus retained kernel-tuning source. This attempt owns only a new WeightPrepare weight-decay transform kernel/wiring plus this note unless later entries explicitly expand scope.
- Prior-note/source search command:
  - `rg -n "weight_decay|softplus|decay|loramlp_d|weight decay|decay precursor|softplus transform|exp" .agents/notes/kernel-tuning crates/rwkv-nn/src/kernels/train/time_mixer crates/rwkv-nn/src/modules/time_mixer -S`
  - `rg -n "key_prepare|warps_per_cube|row_tile|WKV7|gated_readout|row-pack|rowpack|LayerNorm|ordered-256|channel_mixer|lm_head|LocalTuner|remote|10\\.100\\.1\\.253" .agents/notes/kernel-tuning crates/rwkv-nn/src/kernels/train -S`
- Matched prior evidence:
  - `2026-05-16-remote-current-profile-gb10.md` shows current remote GB10 state is activation-valid and timing-clean (`54/54`, `76/76`, speedup `2.01x`) and attributes repeated generic BF16/f32 elementwise clusters around WeightPrepare.
  - `2026-05-16-wire-time-mixer-gates-gb10.md` already wired custom learning-rate and value-residual gates. This branch must not repeat those.
  - Closed duplicate guards: residual-add/Burn-add A/B, channel-mixer forced Cube/Burn-reference/fusion/line-size, LocalTuner bypass, WKV7 row-tile restriction, lm-head target-logit/atomic/online-softmax/prune-256, GatedReadout warp32/row-pack, key-prepare warps, LayerNorm ordered-256.
  - No prior `weight_decay_transform` or softplus-transform custom kernel note exists.
- Machine/GPU: primary validation on remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`; local GPU must not be used.
- Remote run directory: `/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, regenerated remote baseline under `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Kernel/stage: WeightPrepare weight-decay transform after `param_weight_decay_lora.forward_without_bias(weight_decay_input)`.
- Hypothesis: replace the Burn expression `-softplus(-(bias + lora_input), 1.0) - 0.5` with one project-owned CubeCL kernel that adds the 1D bias and computes the transform with f32 intermediates, returning BF16. This should remove several generic BF16/f32 elementwise/cast launches before `key_prepare` without changing math intent.
- Candidate parameters: vector width/line size `[1,2,4,8,16,32,64]` under a hardware/shape `LocalTuner`; key includes runtime, hardware fingerprint, dtype, rows, `d_model`, max vector width, alias/in-place, and deterministic boundary.
- Expected keep/revert boundary:
  - Keep only if local compile/format checks pass, remote compile passes, remote activation remains `54/54 PASS`, and standard remote compare stays speedup `>1.0` without broad regressions.
  - If activation drifts, if the transform math mismatches Burn softplus beyond trace tolerance, or if remote nsys shows the new kernel does not reduce the targeted generic cluster, revert code and keep this note as negative evidence.
- Next command: inspect existing `learning_rate_gate` / `value_residual_gate` forward patterns and implement the new kernel with the same narrow trait/module style.

## Implementation Edit

- Inspected `learning_rate_gate` and `value_residual_gate` forward patterns.
- Added `crates/rwkv-nn/src/kernels/train/time_mixer/weight_decay_transform/` with:
  - `io.rs` for `[D]` base and `[B,T,D]` input validation.
  - `kernel.rs` for regular and power-of-two broadcast-index kernels.
  - `forward.rs` with `WeightDecayTransformForwardAutotuneKey`, `LocalTuner`, line-size candidates, CubeCL hardware fingerprint fields, alias/deterministic fields, and fusion registration.
  - `mod.rs` with the public wrapper and Burn reference expression.
- Wired `WeightPrepare::forward` to call `param_weight_decay_lora.forward_without_bias(...)`, then `weight_decay_transform(bias_1d, input)` instead of Burn `-softplus(-lora_result) - 0.5`.
- Added `weight_decay_transform` to `time_mixer` kernel modules and TimeMixer trait bounds.
- Next command: run local compile check only, `cargo check -p rwkv-nn --features cuda,fusion`; no local GPU execution.

## Local Compile Check 1

- Command: `cargo check -p rwkv-nn --features cuda,fusion`.
- Result: failed before codegen. `CausalCell<B>` calls `TimeMixer::forward`, whose new bound requires `B: WeightDecayTransformBackend`.
- Next edit: extend the `CausalCell` implementation bound/import with `WeightDecayTransformBackend`; do not change kernel math.

## Trait Bound Edit

- Scope: compile-only wiring change; no kernel math change.
- Decision: add `WeightDecayTransformBackend` to the central `TrainBackend` trait/import in `crates/rwkv-nn/src/kernels/train/mod.rs` so existing `CausalCell<B: TrainBackend>` and higher-level training paths inherit the new primitive capability consistently.
- Next command: rerun local compile check only, `cargo check -p rwkv-nn --features cuda,fusion`; no local GPU execution.

## Local Compile Check 2

- Command: `cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed in dev check profile.
- Next command: local format check only, `cargo +nightly fmt --all -- --check`; no local GPU execution.

## Local Format Check 1

- Command: `cargo +nightly fmt --all -- --check`.
- Result: failed only on rustfmt layout in the new `weight_decay_transform` files.
- Next edit: apply the rustfmt-suggested local formatting in `crates/rwkv-nn/src/kernels/train/time_mixer/weight_decay_transform/{forward.rs,mod.rs}` without running a broad workspace formatter.

## Local Format Edit

- Scope: rustfmt-suggested import and method-call wrapping only in the new weight-decay transform module.
- Next command: rerun local format check, `cargo +nightly fmt --all -- --check`; no local GPU execution.

## Local Format Check 2

- Command: `cargo +nightly fmt --all -- --check`.
- Result: passed.
- Next command: remote preflight on `caizus@10.100.1.253` to confirm repo path, GPU, branch, and toolchain before syncing this branch.

## Remote Preflight 1

- Host: `caizus@10.100.1.253`.
- Command: `ssh -i ~/.ssh/id_ed25519_dgx_spark_windows -o BatchMode=yes caizus@10.100.1.253 'set -e; hostname; pwd; test -d ~/Projects/Packages/rwkv-rs-stable; cd ~/Projects/Packages/rwkv-rs-stable; git branch --show-current; rustc --version; cargo --version; nvidia-smi --query-gpu=name,compute_cap,utilization.gpu --format=csv,noheader'`.
- Result: invalid preflight. Host is `spark-35ac`, but `~/Projects/Packages/rwkv-rs-stable` is not currently a Git repository (`fatal: not a git repository`).
- Decision: do not sync or run builds against that path until the actual remote repo path is located.
- Next command: inspect remote `~/Projects/Packages` and nearby Git roots.

## Remote Path Probe

- Host: `caizus@10.100.1.253`.
- Command: `find ~/Projects -maxdepth 4 -type d \( -name .git -o -name rwkv-rs-stable -o -name rwkv-rs-test \) -print` plus directory listings.
- Result: remote code mirrors exist at `/home/caizus/Projects/Packages/rwkv-rs-stable` and `/home/caizus/Projects/Packages/rwkv-rs-test`, but the stable mirror has no `.git` directory. This matches an rsync mirror rather than a branch checkout.
- Remote binary provenance plan: after scoped rsync, treat `/home/caizus/Projects/Packages/rwkv-rs-stable` as a mirror of local branch `kernel-tuning-weight-decay-transform-gb10-20260516` and record the local branch/dirty-tree state.
- Next command: remote preflight 2 without Git assumptions; confirm toolchain and GPU.

## Remote Preflight 2

- Host: `caizus@10.100.1.253`, remote path `/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Command: `ssh -i ~/.ssh/id_ed25519_dgx_spark_windows -o BatchMode=yes caizus@10.100.1.253 'set -e; cd /home/caizus/Projects/Packages/rwkv-rs-stable; hostname; rustc --version; cargo --version; nvidia-smi --query-gpu=name,compute_cap,utilization.gpu --format=csv,noheader; test -d /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 && echo baseline_present=yes || echo baseline_present=no'`.
- Result: host `spark-35ac`; Rust/Cargo `1.95.0`; GPU `NVIDIA GB10`, compute capability `12.1`, utilization `0 %`; companion baseline directory present.
- Next command: scoped rsync local repo mirror to remote stable path, excluding `.git`, `target`, `weights`, and `results`.

## Remote Sync 1

- Command: `rsync -az --delete --exclude '.git/' --exclude 'target/' --exclude 'weights/' --exclude 'results/' -e 'ssh -i ~/.ssh/id_ed25519_dgx_spark_windows -o BatchMode=yes' ./ caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable/`.
- Result: completed.
- Remote source provenance: mirror of local branch `kernel-tuning-weight-decay-transform-gb10-20260516` with existing dirty-tree constraint recorded above.
- Next command: remote compile check, `cargo check -p rwkv-nn --features cuda,fusion`.

## Remote Compile Check 1

- Host/path: `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Command: `cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed in dev check profile.
- Next command: remote standard acceptance compare with `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --repeat 3 --warmup 1 --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.

## Remote Compare 1

- Host/path: `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Command: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --repeat 3 --warmup 1 --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Result: command exited nonzero because one timing row failed.
- Correctness: `activation_summary compared=54 passed=54 failed=0 missing=0 extra=0 worst_abs=3.612041e-2 worst_rel=2.050613e4 worst_cosine=0.99999923`.
- Timing summary: `timing_summary compared=76 passed=75 failed=1 missing=0 extra=0 ignored=1 actual_total_ms=81.037 baseline_total_ms=175.887 speedup=2.17x`.
- Only failing row: `timing/cells/cell_0000/channel_mixer.time.json`, `actual_ms=1.690`, `baseline_ms=1.668`, `speedup=0.99x`. This is unrelated to the edited weight-decay transform path and is close enough to require a repeat before deciding.
- Decision: do not claim keep yet because the expected boundary was `76/76` timing pass. Rerun the same standard compare with the release binary already built and tuner/cache warm to distinguish noise from a stable regression.
- Next command: remote standard compare repeat with the same args.

## Remote Compare 2

- Host/path: `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Command: same standard compare as Remote Compare 1.
- Result: passed.
- Correctness: `activation_summary compared=54 passed=54 failed=0 missing=0 extra=0 worst_abs=3.612041e-2 worst_rel=2.050613e4 worst_cosine=0.99999923`.
- Timing summary: `timing_summary compared=76 passed=76 failed=0 missing=0 extra=0 ignored=1 actual_total_ms=80.515 baseline_total_ms=175.887 speedup=2.18x`.
- Timing context: previous remote current profile before this branch was `actual_total_ms=87.653 baseline_total_ms=175.887 speedup=2.01x`, so this branch's standard run improved the total by about `7.1 ms` on the same remote baseline.
- Decision: correctness/timing gate is passed on GB10. Still run profiler attribution before keep/drop decision because the hypothesis targets the generic WeightPrepare elementwise cluster.
- Next command: short remote `nsys` profile of the already-built release binary for attribution only.

## Remote Nsys Profile 1

- Host/path: `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Command: `nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --output target/rwkv-test/nsys-weight-decay-transform-gb10 target/release/rwkv-test compare-rwkv-nn --color never --repeat 1 --warmup 0 --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Result: profiler report generated at `target/rwkv-test/nsys-weight-decay-transform-gb10.nsys-rep`; command exited nonzero because the profiler run intentionally used `repeat=1,warmup=0` while the baseline timing files use `repeat=3,warmup=1`.
- Correctness inside profiler run: activation still passed `54/54`.
- Timing inside profiler run: invalid for acceptance; all timing rows fail with profile mismatch and profiler overhead. Do not use these `.time.json` values for speedup.
- Next command: export profiler report to sqlite and summarize CUDA kernel time/launch counts.

## Remote Nsys Export And Summary 1

- Export command: `nsys export --type sqlite --output target/rwkv-test/nsys-weight-decay-transform-gb10.sqlite target/rwkv-test/nsys-weight-decay-transform-gb10.nsys-rep`.
- Result: sqlite exported at `target/rwkv-test/nsys-weight-decay-transform-gb10.sqlite`.
- Current-vs-new targeted kernel summary from sqlite:
  - Previous `nsys-current-gb10.sqlite`: `kernel_scalar_binop_c_bf16_n_8` 216 launches / `11.323 ms`; `unary_float_f_bf16_n_8` 108 launches / `3.820 ms`; BF16/F32 cast pair 36 launches each.
  - New `nsys-weight-decay-transform-gb10.sqlite`: `weight_decay_transform_forward_kernel_f__n_4` 24 launches / `1.568 ms`; `kernel_scalar_binop_c_bf16_n_8` no longer appears in the targeted summary; `unary_float_f_bf16_n_8` is down to 24 launches / `0.055 ms`; BF16/F32 cast pair 24 launches each.
- Interpretation: the custom kernel is replacing the intended Burn scalar/unary softplus transform chain. The nsys run used a different repeat/warmup boundary, so launch counts are used as attribution evidence rather than acceptance timing.
- Next command: targeted `ncu` on `weight_decay_transform_forward_kernel` for SOL/occupancy-style metrics.

## Remote NCU Probe 1

- Command: `ncu --version`.
- Result: failed, `zsh:1: command not found: ncu`.
- Next command: probe common Nsight Compute/CUDA binary paths without changing remote environment.

## Remote NCU Probe 2

- Command: print `PATH`, `command -v nsys`, `command -v ncu`, and search common `/usr/local/cuda*` / `/opt/nvidia` paths for `ncu`.
- Result: `ncu` is installed but not on PATH. Valid candidates include `/usr/local/cuda-13.0/bin/ncu` and `/opt/nvidia/nsight-compute/2025.3.1/ncu`; `nsys` is `/usr/local/bin/nsys`.
- Next command: run targeted Nsight Compute with the explicit `/usr/local/cuda-13.0/bin/ncu` path.

## Remote NCU Attempt 1

- Command: `/usr/local/cuda-13.0/bin/ncu --target-processes all --kernel-name-base demangled --kernel-name regex:weight_decay_transform_forward_kernel --launch-count 1 --set speedOfLight --csv --log-file target/rwkv-test/ncu-weight-decay-transform-gb10.csv target/release/rwkv-test compare-rwkv-nn --color never --repeat 1 --warmup 0 --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Result: invalid for metrics. Nsight Compute launched but wrote `No metrics to collect found in sections`; app returned error code `1` because the compare run used profiler repeat/warmup mismatch.
- Decision: retry with an explicit section and a wider demangled kernel regex.
- Next command: targeted Nsight Compute retry using `--section SpeedOfLight` and `regex:.*weight_decay_transform.*`.

## Remote NCU Attempt 2

- Command: targeted Nsight Compute retry using `--section SpeedOfLight` and unquoted `regex:.*weight_decay_transform.*`.
- Result: invalid shell invocation; remote `zsh` treated `*` as a glob and failed with `no matches found`.
- Next command: rerun the same retry with the regex argument quoted for remote zsh.

## Remote NCU Attempt 3

- Command: `/usr/local/cuda-13.0/bin/ncu --target-processes all --kernel-name-base demangled --kernel-name 'regex:.*weight_decay_transform.*' --launch-count 1 --section SpeedOfLight --csv --log-file target/rwkv-test/ncu-weight-decay-transform-gb10-retry.csv target/release/rwkv-test compare-rwkv-nn --color never --repeat 1 --warmup 0 --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Result: invalid for metrics. Nsight Compute attached to the process, but metric collection failed with `ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0`.
- Correctness inside profiled run: activation still passed `54/54`.
- Timing inside profiled run: invalid for acceptance due profiler overhead plus `repeat=1,warmup=0` mismatch.
- Decision: do not block this keep decision on ncu because the remote user cannot access performance counters without system-level configuration. Record this as an environment blocker for occupancy/SOL metrics.

## Keep Decision

- Keep state: keep the weight-decay transform kernel and wiring on branch `kernel-tuning-weight-decay-transform-gb10-20260516`.
- Acceptance evidence:
  - Local compile: `cargo check -p rwkv-nn --features cuda,fusion` passed.
  - Local format: `cargo +nightly fmt --all -- --check` passed.
  - Remote compile: `cargo check -p rwkv-nn --features cuda,fusion` passed on `10.100.1.253`.
  - Remote standard compare repeat: activation `54/54`, timing `76/76`, `actual_total_ms=80.515 baseline_total_ms=175.887 speedup=2.18x`.
- Attribution evidence: targeted nsys summary shows the previous BF16 scalar/unary softplus transform chain was replaced by `weight_decay_transform_forward_kernel_f__n_4`; `kernel_scalar_binop_c_bf16_n_8` disappeared from the targeted summary and `unary_float_f_bf16_n_8` dropped from `108` launches / `3.820 ms` to `24` launches / `0.055 ms`.
- Remaining profiler gap: ncu SOL/occupancy metrics are blocked by `ERR_NVGPUCTRPERM` on `10.100.1.253`; rerun ncu after enabling GPU performance counters or under an account/container with permission.
- Next action candidate: continue with the next largest owned kernels from nsys (`wkv7_pretrain_forward_output`, `mix6_forward`, `channel_mixer_relu_square`, `gated_readout_combine`, `key_prepare_forward_64`) under new branch+note attempts, avoiding already-recorded duplicate experiments.
