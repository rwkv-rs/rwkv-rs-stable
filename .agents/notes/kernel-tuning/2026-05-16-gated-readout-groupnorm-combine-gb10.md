# GatedReadout GroupNorm Combine GB10

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-gated-readout-groupnorm-combine-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this branch inherits broad unrelated workspace changes plus the retained post-gates/weight-decay kernel tree. This attempt owns only the GatedReadout GroupNorm + combine forward boundary and this note unless a later entry explicitly expands scope.
- Prior-note search commands:
  - `rg -n "gated_readout|GatedReadout|group_norm|GroupNorm|combine|warp32|rowpack|row-pack|reduce_kernel_in_bf16|bonus|normalized" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md crates/rwkv-nn/src/modules/time_mixer/gated_readout.rs crates/rwkv-nn/src/kernels/train/time_mixer/gated_readout_combine -S`
  - `rg -n "GroupNorm|group_norm|gated_readout_combine|reduce_kernel_in_bf16|gated-readout" /root/.codex/memories/MEMORY.md`
- Matched prior evidence:
  - `2026-05-16-gated-readout-forward-combine-gb10.md` kept the 64-thread/two-warp gated-readout combine implementation.
  - `2026-05-16-gated-readout-warp32-gb10.md` rejected warp32 after direct profiler attribution showed the target combine kernel got slower.
  - `2026-05-16-gated-readout-rowpack-gb10.md` is closed; row-pack did not become the retained implementation.
  - `2026-05-16-post-weight-next-surface-gb10.md` maps the repeated `reduce_kernel_in_bf16* -> scalar/binop -> binop` group immediately before `gated_readout_combine_forward_kernel_f_` to Burn `GroupNorm` inside `GatedReadout::forward`.
  - No prior note or memory entry matched a fused GroupNorm + gated-readout combine implementation.
- Machine/GPU: primary validation on remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`; do not use the local GPU.
- Remote run directory: `/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, `num_heads=12`, `head_size=64`, regenerated remote baseline under `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Hypothesis: fuse the Burn GroupNorm over WKV7 output with the existing gated-readout combine so the forward path removes the generic GroupNorm reductions and the separate combine launch while preserving the group-size-64 f32 reduction order closely enough for trace-backed BF16 accuracy.
- Candidate parameters: fixed first candidate, one block per `(row, head)`, `head_size=64`, two reductions in one kernel for mean/square mean plus the existing bonus reduction. No autotune candidate set until one correctness-valid implementation exists.
- Expected keep/revert boundary:
  - Keep only if remote activation passes and standard remote timing remains `>1.0x`, with profiler evidence showing the targeted GroupNorm generic kernels disappear or shrink and the fused kernel does not regress the GatedReadout block.
  - Revert if activation drifts, if GroupNorm affine/beta semantics mismatch Burn, if compile fails, or if profiler attribution shows the extra reductions make the fused kernel slower than Burn GroupNorm plus the retained 64-thread combine.
- Next command: inspect Burn `GroupNorm` fields and existing gated-readout combine primitive/fusion IO to plan the smallest source edit before changing code.

## Source Inspection

- Command: inspected Burn `burn-nn-0.21.0` `GroupNorm` implementation and the existing `gated_readout_combine` IO/forward/kernel files.
- Result:
  - Burn `GroupNorm` has public `gamma`, `beta`, `num_groups`, `num_channels`, `epsilon`, and `affine` fields.
  - Current `GatedReadout::forward` reshapes WKV output to `[rows, embedded_dim]`, applies affine GroupNorm with `num_groups=12`, `num_channels=768`, `epsilon=64e-5`, then reshapes back before `gated_readout_combine`.
  - Existing combine already maps one block to `(row, head)` with `HEAD_SIZE=64`, `BLOCK_SIZE=64`, and `NUM_WARPS=2`.
- Code change planned: replace the combine primitive input `normalized` with raw `wkv_output` plus `norm_gamma`, `norm_beta`, and `norm_epsilon`; compute GroupNorm mean/variance/affine inside `gated_readout_combine_forward_kernel` before the existing bonus/gate expression.
- Next command: edit `gated_readout.rs` and `gated_readout_combine/{io.rs,mod.rs,forward.rs,kernel.rs}` for the fused first candidate.

## Code Change 1

- Changed `GatedReadout::forward` to pass raw `wkv7_forward_output`, GroupNorm `gamma`, `beta`, and `epsilon` into `gated_readout_combine` instead of precomputing Burn GroupNorm.
- Extended `GatedReadoutCombineForwardInputs` / primitive inputs with `wkv_output`, `norm_gamma`, `norm_beta`, and `norm_epsilon`.
- Updated the reference and Autodiff wrapper to compute GroupNorm then the existing bonus/gate expression.
- Updated the Cube forward/fusion wrapper and kernel to compute per `(row, head)` GroupNorm mean/variance/affine and gated-readout combine in one kernel.
- Next command: run a local CPU-side compile check without using the local GPU: `cargo check -p rwkv-nn --features cuda,fusion`.

## Local Compile Check 1

- Command: `cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed, but emitted three unused-variable warnings in `crates/rwkv-nn/src/modules/time_mixer/gated_readout.rs` for the no-longer-needed `gate_input.dims()` destructuring.
- Decision: fix the warnings before syncing to remote.
- Next command: remove the unused dims destructuring from `gated_readout.rs`, then rerun the same compile check.

## Code Change 2

- Removed the unused `gate_input.dims()` destructuring from `GatedReadout::forward`.
- Next command: rerun `cargo check -p rwkv-nn --features cuda,fusion`.

## Local Compile Check 2

- Command: `cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed with no warnings from the changed files.
- Decision: sync only this attempt's changed files and note to remote `10.100.1.253`, then compile there before any GPU compare.
- Next command: explicit-file `rsync -avR` for:
  - `.agents/notes/kernel-tuning/2026-05-16-gated-readout-groupnorm-combine-gb10.md`
  - `crates/rwkv-nn/src/modules/time_mixer/gated_readout.rs`
  - `crates/rwkv-nn/src/kernels/train/time_mixer/gated_readout_combine/io.rs`
  - `crates/rwkv-nn/src/kernels/train/time_mixer/gated_readout_combine/mod.rs`
  - `crates/rwkv-nn/src/kernels/train/time_mixer/gated_readout_combine/forward.rs`
  - `crates/rwkv-nn/src/kernels/train/time_mixer/gated_readout_combine/kernel.rs`

## Remote Sync 1

- Host/path: `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Command: explicit-file `rsync -avR` for this note plus the six GatedReadout/GatedReadoutCombine source files.
- Result: sync completed; `33,497` bytes total source size, speedup `1.18`.
- Next command: remote compile check `cargo check -p rwkv-nn --features cuda,fusion` in `/home/caizus/Projects/Packages/rwkv-rs-stable`.

## Remote Compile Check 1

- Host/path: `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Command: `cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed.
- Next command: remote standard CUDA compare with regenerated GB10 baseline: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1`, teeing output to `target/rwkv-test/gated-readout-groupnorm-combine-compare.log`.

## Remote Compare 1

- Host/path: `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Command: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1 | tee target/rwkv-test/gated-readout-groupnorm-combine-compare.log`.
- Result: release build completed and compare passed activation.
  - `activation_summary compared=54 passed=54 failed=0`.
  - `timing_summary compared=76 passed=75 failed=1 actual_total_ms=72.812 baseline_total_ms=175.887 speedup=2.42x`.
  - The only failed timing row is the previously known `timing/cells/cell_0000/channel_mixer.time.json` at `0.94x`; this is outside the GatedReadout GroupNorm/combine boundary.
  - `cells/*/time_mixer` improved to `42.356 ms` actual vs `120.381 ms` baseline, `2.84x`.
- Decision: correctness and standard timing gates pass strongly enough to profile before keep/revert.
- Next command: remote `nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --output target/rwkv-test/nsys-gated-readout-groupnorm-combine target/release/rwkv-test compare-rwkv-nn --color never --repeat 1 --warmup 0 --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`, then query the sqlite for `gated_readout_combine_forward_kernel`, `reduce_kernel_in_bf16`, and `kernel_binop_c_bf16` groups.

## Remote Nsys Capture 1

- Host/path: `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Command: `nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --output target/rwkv-test/nsys-gated-readout-groupnorm-combine target/release/rwkv-test compare-rwkv-nn --color never --repeat 1 --warmup 0 --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 | tee target/rwkv-test/gated-readout-groupnorm-combine-nsys.log`.
- Result: capture completed and generated `target/rwkv-test/nsys-gated-readout-groupnorm-combine.nsys-rep`.
- Invalid timing boundary: the compare timing emitted inside nsys uses `repeat=1,warmup=0` and profiler instrumentation; it reports profile mismatches and must not be used as speedup evidence.
- Next command: export the `.nsys-rep` to sqlite and query kernel groups.

## Remote Nsys Query 1

- Host/path: `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Command: `nsys export --type sqlite --force-overwrite=true --output target/rwkv-test/nsys-gated-readout-groupnorm-combine.sqlite target/rwkv-test/nsys-gated-readout-groupnorm-combine.nsys-rep`, then Python sqlite query for relevant kernel groups.
- Result:
  - Export succeeded and `target/rwkv-test/nsys-gated-readout-groupnorm-combine.sqlite` exists.
  - Relevant groups:
    - `wkv7_pretrain_forward_output_kernel_f_bf16`: `14.591 ms / 24`.
    - `mix6_forward_kernel_f__n_1`: `9.634 ms / 24`.
    - `channel_mixer_relu_square_forward_kernel_f__n_2`: `9.502 ms / 24`.
    - `gated_readout_combine_forward_kernel_f_`: `8.095 ms / 24`, `337.280 us` avg, `grid=(8192,12,1)`, `block=(64,1,1)`, `regs=25`, dynamic smem `8`.
    - `kernel_binop_c_bf16_n_8`: `5.892 ms / 48`.
    - `layer_norm_forward_kernel_f_`: `4.338 ms / 52`.
  - The targeted `reduce_kernel_in_bf16*` and `kernel_binop_c_bf16_n_1` groups are absent from the relevant grouped output; before this branch, the post-weight profile showed `reduce_kernel_in_bf16*` at `3.781 ms / 48` and `kernel_binop_c_bf16_n_1` at `4.047 ms / 48`.
  - The fused combine kernel itself is slightly slower than the previous `7.832 ms / 24`, but it removes the preceding Burn GroupNorm generic work and the standard compare improved to `2.42x`.
- Decision: keep this candidate as positive GB10 evidence unless formatting or a follow-up clean compare contradicts it.
- Next command: local formatting check for the touched crate: `cargo +nightly fmt --package rwkv-nn --check`.

## Format Check 1

- Command: `cargo +nightly fmt --package rwkv-nn --check`.
- Result: failed on one rustfmt line-wrap difference in `crates/rwkv-nn/src/kernels/train/time_mixer/gated_readout_combine/kernel.rs`.
- Next command: apply the rustfmt line-wrap change, then rerun the same format check.

## Code Change 3

- Applied the rustfmt line-wrap change in `gated_readout_combine/kernel.rs`; no semantic change.
- Next command: rerun `cargo +nightly fmt --package rwkv-nn --check`.

## Format Check 2

- Command: `cargo +nightly fmt --package rwkv-nn --check`.
- Result: passed.
- Decision: sync the formatted `kernel.rs` and updated note back to remote so the remote copy matches this branch.
- Next command: explicit-file `rsync -avR` for this note and `crates/rwkv-nn/src/kernels/train/time_mixer/gated_readout_combine/kernel.rs`.

## Remote Sync 2

- Host/path: `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Command: explicit-file `rsync -avR` for this note and `crates/rwkv-nn/src/kernels/train/time_mixer/gated_readout_combine/kernel.rs`.
- Result: sync completed; `15,632` bytes total source size, speedup `2.14`.
- Current decision: keep the fused GatedReadout GroupNorm + combine candidate on this branch. It is remote activation-clean, standard timing-positive on GB10, and profiler-confirmed to remove the targeted Burn GroupNorm generic reduction/binop groups.

## Completion Audit Snapshot

- Objective requirement: use branch/worktree plus notes for every new tuning attempt. Current state: satisfied for this attempt on branch `kernel-tuning-gated-readout-groupnorm-combine-gb10-20260516`; note was written before code/profiler/benchmark commands and records keep/revert boundaries.
- Objective requirement: kernel implementation selection should be hardware/shape aware with key dimensions such as runtime, GPU capability, dtype, rows, `d_model`, block/warp/vector/in-place/deterministic. Current state: partially covered across existing autotune-key work and skill rules, but this specific fused candidate is a fixed first implementation and has not yet added an autotune candidate set or hardware dispatch key.
- Objective requirement: Burn/CubeCL autotune mechanism should be understood and tuner overhead not counted incorrectly. Current state: skill rules and previous notes cover `LocalTuner::execute` cache-hit/miss boundaries; this fused candidate does not use `LocalTuner`, so no new host-side tuner overhead is introduced.
- Objective requirement: CUDA lower-level evidence should include occupancy/register/shared-memory/memory/warp metrics from ncu, not only `.time.json` or nsys timing. Current state: not yet satisfied for this fused candidate; nsys has launch/register/shared-memory timing, but not achieved occupancy or warp/memory metrics.
- Objective requirement: final speedup `>1.0` on both local and `10.100.1.253`. Current state: remote GB10 passes (`2.42x` standard compare); local run is intentionally not measured because the user said the local GPU is busy and may be inaccurate.
- Objective requirement: precision issues must be debugged, not blindly reverted. Current state: this candidate has no activation failure on remote; prior LayerNorm ordered-256 precision diagnosis is recorded separately.
- Missing work: ncu metrics for this candidate, local validation when the local GPU is available, and additional non-duplicate iterations for remaining surfaces such as WKV7/mix6/channel-mixer/lm-head.
- Next command: remote ncu on `gated_readout_combine_forward_kernel_f_` using sections `SpeedOfLight`, `Occupancy`, `MemoryWorkloadAnalysis`, `SchedulerStats`, and `WarpStateStats`, writing `target/rwkv-test/ncu-gated-readout-groupnorm-combine.csv`.

## Remote Ncu Attempt 1

- Host/path: `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Command: `/usr/local/cuda-13.0/bin/ncu --target-processes all --kernel-name-base demangled --kernel-name regex:gated_readout_combine_forward_kernel --launch-count 1 --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats --csv --log-file target/rwkv-test/ncu-gated-readout-groupnorm-combine.csv target/release/rwkv-test compare-rwkv-nn --color never --repeat 1 --warmup 0 --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Result: command exited `1`. The profiled program still produced `activation_summary compared=54 passed=54 failed=0`, but emitted invalid timing mismatch rows because `repeat=1,warmup=0` under ncu instrumentation.
- Decision: no ncu metric conclusion yet. Need inspect `target/rwkv-test/ncu-gated-readout-groupnorm-combine.csv` to distinguish GPU counter permission failure from kernel-name/section issues.
- Next command: remote `sed -n '1,160p' target/rwkv-test/ncu-gated-readout-groupnorm-combine.csv`.

## Remote Ncu Log Inspection 1

- Host/path: `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Command: `sed -n '1,180p' target/rwkv-test/ncu-gated-readout-groupnorm-combine.csv`.
- Result: ncu connected to the process, then failed with `ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0`.
- Decision: ncu achieved-occupancy / warp / memory metrics remain blocked by host permissions on `10.100.1.253`; this is not a kernel-name or section selection issue. Keep nsys plus standard compare as the available remote evidence until GPU counter permissions are enabled.
- Keep/revert state: keep the fused GroupNorm + combine candidate as positive GB10 evidence. Missing ncu metrics are an environment blocker, not a failed kernel result.

## Next Surface Audit

- WKV7 preflight: current notes already close row-tile-only tuning (`row_tile=16` negative), and `2026-05-16-remote-time-mixer-evidence.md` says WKV7 is not the dominant remaining GB10 surface after resolving matmul families. A real WKV7 time-split/state-handoff/scan design is a separate algorithmic project, not the next small branch from whole-module timing.
- Decision: before opening another implementation branch, query the latest `nsys-gated-readout-groupnorm-combine.sqlite` for full top kernel groups so the next attempt targets the current post-GroupNorm-fusion surface rather than stale pre-fusion evidence.
- Next command: remote Python sqlite query over `target/rwkv-test/nsys-gated-readout-groupnorm-combine.sqlite`, top kernel groups by total duration.

## Full Top Kernel Query

- Host/path: `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Command: Python sqlite query over `target/rwkv-test/nsys-gated-readout-groupnorm-combine.sqlite`, grouping all kernels by resolved demangled name plus launch metadata.
- Result: current post-fusion top groups are:
  - `matmul_entry_lhs_bf16_lhs_size_1_rhs_bf16_rhs_size_1_acc_bf16_acc_size_8`, `grid=(4096,16,1)`, `2` launches, `33.597 ms` total, likely the lm-head projection.
  - Other Burn/Cubek BF16 matmuls: `17.379 ms / 96`, `13.804 ms / 24`, `12.741 ms / 24`, `6.163 ms / 94`, plus smaller groups.
  - `wkv7_pretrain_forward_output_kernel_f_bf16`: `14.591 ms / 24`.
  - `mix6_forward_kernel_f__n_1`: `9.634 ms / 24`.
  - `channel_mixer_relu_square_forward_kernel_f__n_2`: `9.502 ms / 24`.
  - `lm_head_l2wrap_ce_forward_row_kernel_f__i_i32`: `8.559 ms / 2`.
  - `gated_readout_combine_forward_kernel_f_`: `8.095 ms / 24`.
  - `key_prepare_forward_64_kernel_f_`: `6.325 ms / 24`.
  - `kernel_binop_c_bf16_n_8`: `5.892 ms / 48`.
  - `layer_norm_forward_kernel_f_`: `4.338 ms / 52`.
- Interpretation:
  - The largest remaining surface is still Burn/Cubek BF16 matmul, especially the two-launch `grid=(4096,16,1)` shape. Existing notes already identify the likely lm-head projection and warn that true projection+loss fusion requires a broader API/matmul-epilogue design rather than a small row-loss tweak.
  - WKV7/mix6/channel-mixer/key-prepare/lm-head-row/layernorm all have duplicate-guarded negative or constrained directions recorded. Any next implementation branch must name a materially different boundary, not another row-tile, line-size, forced-Cube, atomic, target-logit, or block-size retry.
- Current next candidate if continuing implementation: a design branch for lm-head projection plus loss API/epilogue feasibility, because it targets the largest live kernel family and is not the same as the closed row-loss variants. This should start as a read-only design branch unless the API boundary is proven safe.
