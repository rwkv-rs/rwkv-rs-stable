# Gated Readout Forward Combine on GB10

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-gated-readout-forward-combine-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated uncommitted changes plus previous kernel notes/diagnostics. This attempt is limited to a new gated-readout combine kernel, its module wiring, and this note.
- Prior-note search command:
  - `rg -n "gated_readout|GatedReadout|bonus|out_gated|forward-only|forward combine|GatedReadout combine" .agents/notes/kernel-tuning crates/rwkv-nn/src -S`
- Matched prior evidence:
  - `2026-05-16-wire-time-mixer-gates-gb10.md`: current kept remote high point is `actual_total_ms=91.595`, baseline `175.887`, speedup `1.92x`, activation/timing pass.
  - `2026-05-16-gated-readout-combine-analysis.md`: full backward is high risk, but remote nsys shows a measurable post-WKV7 generic chain before output projection.
  - `2026-05-16-wkv7-remote-launch-design.md`: WKV7 launch retuning is closed as evidence-only; the next cleaner boundary is the GatedReadout combine chain.
  - Duplicate guards still apply: do not rerun residual-add A/B, channel-mixer forced Cube/Burn-reference/fusion, LocalTuner bypass, WKV7 forced `row_tile=16`, or lm-head row-kernel variants.
- Machine/GPU: remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1` for acceptance. Do not run local GPU.
- Remote run directory: `/home/caizus/Projects/Packages/rwkv-rs-stable` is a synchronized non-git copy; branch ownership remains in the local checkout.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, `num_heads=12`, `head_size=64`, regenerated remote baseline under `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Kernel/stage: TimeMixer `GatedReadout::forward` after WKV7 and output-gate LoRA/group norm, before output projection.
- Hypothesis: add a project-owned forward combine primitive:
  - Input tensors: normalized WKV7 output `[B,T,D]`, output gate `[B,T,D]`, WKV7 receptance/key/value `[B,T,H,64]`, bonus parameter `[H,64]`.
  - Output: `out_gated[b,t,e] = (normalized[b,t,e] + sum_i(receptance[b,t,h,i] * replacement_key[b,t,h,i] * bonus_param[h,i]) * value[b,t,h,lane]) * gate[b,t,e]`.
  - Non-autodiff CubeBackend uses the fused forward kernel for `compare-rwkv-nn`.
  - Autodiff backends keep the Burn reference graph so training gradients are not replaced by an unverified duplicate backward.
- Candidate parameters: fixed `head_size=64`, one work unit per `(B*T, head)` computes the head reduction once and writes all 64 lanes. No autotune candidates until this first version proves correctness and timing.
- Expected keep/revert boundary:
  - Keep only if remote activation passes and remote timing stays above the current kept `1.92x` or improves targeted time-mixer/GatedReadout evidence without new failures.
  - Revert or close as negative if activation drifts, compile breaks Autodiff training paths, or timing regresses against the kept remote branch.
- Next command: implement the forward combine primitive and module wiring, then compile locally without running GPU.

## Implementation edit

- Added `crates/rwkv-nn/src/kernels/train/time_mixer/gated_readout_combine/`:
  - `io.rs` input contract and primitive containers.
  - `kernel.rs` CubeCL forward kernel: one cube per `(row, head)`, 64 units reduce `receptance * replacement_key * bonus` once per head and write 64 output lanes.
  - `forward.rs` CubeBackend/Fusion dispatch.
  - `mod.rs` public helper, CubeBackend implementation, and Autodiff reference fallback.
- Updated `TrainBackend` and `TimeMixer::forward` bounds to include `GatedReadoutCombineBackend`.
- Replaced `GatedReadout::forward`'s Burn bonus/out-gated expression with `gated_readout_combine(...)`.
- Next command: `cargo check -p rwkv-nn --features cuda,fusion`.

## Local compile check 1

- Command: `cargo check -p rwkv-nn --features cuda,fusion`.
- Result: failed before codegen.
- Errors:
  - `Shape::dims()` in `forward.rs` needed an explicit const dimension.
  - Autodiff reference fallback needed explicit `TensorPrimitive::<Self>::Float(...)` / typed tensors.
- Decision: fix type annotations only, no algorithm change.

## Local compile check 2

- Command: `cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed in `13.68s`.
- No local GPU execution was performed.
- Next command: format the touched Rust files, then sync the branch files to remote `10.100.1.253` for CUDA compare.

## Format

- Command: `cargo +nightly fmt --all`.
- Result: passed.
- Note: the checkout already contains broad dirty changes, so `git diff` includes older train-boundary edits outside this attempt. This attempt's owned files are the new `gated_readout_combine` module, gated-readout wiring, `TrainBackend`/TimeMixer bounds needed to expose the new backend capability, and this note.
- Next command: rerun `cargo check -p rwkv-nn --features cuda,fusion` after formatting.

## Local compile check 3

- Command: `cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed in `21.40s`.
- No local GPU execution was performed.
- Next command: sync this branch's owned files to remote run directory and run remote CUDA compare.

## Remote sync attempt 1

- Intended command: sync only this branch's owned files to `/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Actual command accidentally included `./` as an rsync source after the file list. That began enumerating the whole local checkout, including `.git` and `target/`.
- Action: interrupted the rsync with `pkill -f "rsync -avR"`. The command exited with rsync code `20`.
- Result validity: invalid sync attempt; do not draw build or timing conclusions from it.
- Cleanup boundary: remove the accidentally copied remote `.git` directory because the remote run directory was explicitly non-git before this attempt. Do not delete remote `target/` because it may contain pre-existing build/profiler artifacts needed for comparison.
- Next command: remove remote `.git` if present, then rerun rsync with an explicit owned-file list and no `./` source.

## Remote sync attempt 2

- Cleanup command: removed `/home/caizus/Projects/Packages/rwkv-rs-stable/.git`; verified it no longer exists.
- Corrected rsync command used an explicit file list and no `./` source:
  - `.agents/notes/kernel-tuning/2026-05-16-gated-readout-forward-combine-gb10.md`
  - `crates/rwkv-nn/src/kernels/train/mod.rs`
  - `crates/rwkv-nn/src/kernels/train/time_mixer/mod.rs`
  - `crates/rwkv-nn/src/kernels/train/time_mixer/gated_readout_combine`
  - `crates/rwkv-nn/src/modules/time_mixer/mod.rs`
  - `crates/rwkv-nn/src/modules/time_mixer/gated_readout.rs`
- Result: rsync exited `0`.
- Next command: remote `cargo check -p rwkv-nn --features cuda,fusion`.

## Remote compile check

- Host: `caizus@10.100.1.253`.
- Command: `cd /home/caizus/Projects/Packages/rwkv-rs-stable && cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed in `13.14s`.
- Next command: remote standard compare with regenerated remote baseline:
  - `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1`.

## Remote compare 1

- Host: `caizus@10.100.1.253`.
- Wrapper command issue: the ssh wrapper used `status=$?`, but `status` is read-only in zsh. The wrapper exited before propagating the compare code. Treat the wrapper exit as invalid, but the compare log itself was complete.
- Log: `/home/caizus/Projects/Packages/rwkv-rs-stable/target/rwkv-test/remote-gated-readout-forward-combine-compare.log`.
- Build: release build completed in `2m 25s`.
- Activation result: passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: marginal. `timing_summary compared=76 passed=75 failed=1 actual_total_ms=90.367 baseline_total_ms=175.887 speedup=1.95x`.
- Failed row: `timing/cells/cell_0000/channel_mixer.time.json`, `1.881ms` actual vs `1.668ms` baseline, `0.89x`.
- Module timing signal:
  - `cells/*/time_mixer`: `60.082ms` actual vs `120.381ms` baseline, `2.00x`.
  - `cells/*/channel_mixer`: `19.918ms` actual vs `25.759ms` baseline, `1.29x`.
- Interpretation: total speedup improves over the kept `1.92x` branch, but the single channel-mixer edge failure means this is not yet a clean keep. This edge is outside the new GatedReadout code and matches the previously recorded remote channel-mixer flake/failure family.
- Next command: rerun the same remote compare with a corrected `rc=$?` wrapper to check whether the single channel-mixer failure is stable.

## Remote compare 2

- Host: `caizus@10.100.1.253`.
- Command: same standard compare with corrected `rc=$?` wrapper.
- Activation result: passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: still not clean and worse than compare 1: `timing_summary compared=76 passed=75 failed=1 actual_total_ms=98.789 baseline_total_ms=175.887 speedup=1.78x`.
- Failed row: `timing/cells/cell_0003/time_mixer.time.json`, `11.097ms` actual vs `10.038ms` baseline, `0.90x`.
- Module timing signal:
  - `cells/*/time_mixer`: `67.690ms` actual vs `120.381ms` baseline, `1.78x`.
  - `cells/*/channel_mixer`: `19.889ms` actual vs `25.759ms` baseline, `1.30x`.
- Decision: not a clean keep. It is accuracy-clean, but repeated timing has one failing row and the second run regressed below the kept `1.92x` total. Need profiler evidence before deciding whether to revise or revert.
- Next command: run remote `nsys` with `repeat=1,warmup=1` and query the new `gated_readout_combine` kernel plus adjacent generic kernels.

## Remote nsys

- Host: `caizus@10.100.1.253`.
- Command: `nsys profile --trace=cuda,nvtx --stats=true --force-overwrite=true -o target/rwkv-test/nsys-gated-readout-forward-combine target/release/rwkv-test compare-rwkv-nn --color never --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 1 --warmup 1`.
- Result: exited nonzero because actual `repeat=1` mismatched baseline `repeat=3`, expected for profiler mode. Activation passed; use only CUDA profiler data.
- Generated:
  - `target/rwkv-test/nsys-gated-readout-forward-combine.nsys-rep`
  - `target/rwkv-test/nsys-gated-readout-forward-combine.sqlite`
- Kernel summary from the new profile:
  - `gated_readout_combine_forward_kernel_f_`: 36 launches, `11.904ms` total, `0.330665ms` avg, grid `8192x12x1`, block `64x1x1`, 21 registers/thread.
  - `kernel_binop_c_bf16_n_8`: 216 launches, `16.743ms` total. In the previous post-gates profile this was 360 launches, `31.778ms`, so the combine kernel removed a real generic-binop surface.
  - `reduce_kernel_in_bf16...out_size_1`: 72 launches, `4.283ms`; previous profile had 108 launches, `6.116ms`.
  - WKV7 and mix6 timings are roughly unchanged.
- Local sequence after WKV7 confirms the new kernel replaces the old bonus/out-gated generic chain:
  - new sequence has `gated_readout_combine_forward_kernel_f_` at about `0.308ms`, followed by output projection.
  - old sequence had several `kernel_binop` and reduction launches in the same region.
- Measurement caveat: the accidental rsync copied local `target/autotune` paths before it was interrupted. The nsys matmul distribution changed materially versus the previous post-gates profile (`lhs_size_8` matmul time rose sharply), so the compare regressions may be polluted by imported autotune cache rather than only by this kernel.
- Next command: remove generated remote autotune cache directories that could have been polluted by the accidental rsync, then rerun compare enough times for GB10 to retune and produce a clean measurement.

## Remote autotune-cache cleanup

- Command: `rm -rf target/autotune crates/rwkv-nn/target/autotune crates/rwkv-test/target/autotune` under `/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Result: removed generated autotune caches; follow-up `find . -path "*/target/autotune"` printed nothing.
- Rationale: these are generated tuning caches and were plausibly polluted by the accidental full-checkout rsync. Build artifacts and profiler reports were left intact.
- Next command: run one remote compare as retune/warm-cache setup, then run a second remote compare for the clean decision.

## Remote compare 3 after cache cleanup

- Host: `caizus@10.100.1.253`.
- Command: standard compare after deleting generated autotune caches.
- Role: retune/warm-cache setup run, not final decision.
- Activation result: passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: `timing_summary compared=76 passed=75 failed=1 actual_total_ms=82.210 baseline_total_ms=175.887 speedup=2.14x`.
- Failed row: `timing/cells/cell_0000/channel_mixer.time.json`, `1.698ms` actual vs `1.668ms` baseline, `0.98x`.
- Module timing signal:
  - `cells/*/time_mixer`: `52.598ms` actual vs `120.381ms` baseline, `2.29x`.
  - `cells/*/channel_mixer`: `19.417ms` actual vs `25.759ms` baseline, `1.33x`.
- Interpretation: after clearing polluted autotune caches, the candidate is much faster than the kept `1.92x` branch and the only failure is a tiny unrelated channel-mixer edge. Need the planned second warm-cache compare for the keep decision.
- Next command: run the same standard remote compare once more without clearing cache.

## Remote compare 4 clean cache

- Host: `caizus@10.100.1.253`.
- Command: standard compare after the retune run, without clearing cache.
- Activation result: passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: `timing_summary compared=76 passed=75 failed=1 actual_total_ms=85.705 baseline_total_ms=175.887 speedup=2.05x`.
- Failed row: `timing/cells/cell_0000/channel_mixer.time.json`, `1.711ms` actual vs `1.668ms` baseline, `0.97x`.
- Module timing signal:
  - `cells/*/time_mixer`: `54.828ms` actual vs `120.381ms` baseline, `2.20x`.
  - `cells/*/channel_mixer`: `19.828ms` actual vs `25.759ms` baseline, `1.30x`.
- Decision: the GatedReadout combine is a real positive candidate after cleaning the autotune cache, but the branch is still not a 76/76 clean keep because the known cell-0 channel-mixer edge fails by `0.043ms`. This failure is outside the edited boundary.
- Next command: rerun `nsys` after cache cleanup so profiler attribution matches the clean measurement state, not the polluted pre-clean profile.

## Remote nsys after cache cleanup

- Host: `caizus@10.100.1.253`.
- Rationale: previous `nsys` proved the fused kernel was active, but the accidental rsync may have polluted autotune cache before that profile. Reprofile after cache cleanup and warm-cache compare so CUDA launch attribution matches the clean `2.05x-2.14x` measurement state.
- Command to run: `nsys profile --trace=cuda,nvtx --stats=true --force-overwrite=true -o target/rwkv-test/nsys-gated-readout-forward-combine-clean target/release/rwkv-test compare-rwkv-nn --color never --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 1 --warmup 1`.
- Expected boundary: compare timing under `nsys`/`repeat=1` is invalid for acceptance. Use only activation sanity and CUDA kernel launch/time attribution.

### Result

- Activation sanity passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing emitted under profiler was profile-mismatched (`repeat=1` vs baseline `repeat=3`) and is invalid for acceptance; the rough module totals were still consistent with the clean-cache standard compares: `actual_total_ms=88.262`, baseline `175.887`, `speedup=1.99x`.
- Generated profiler artifacts:
  - `/home/caizus/Projects/Packages/rwkv-rs-stable/target/rwkv-test/nsys-gated-readout-forward-combine-clean.nsys-rep`
  - `/home/caizus/Projects/Packages/rwkv-rs-stable/target/rwkv-test/nsys-gated-readout-forward-combine-clean.sqlite`
- `cuda_gpu_kern_sum` key rows:
  - `gated_readout_combine_forward_kernel_f_`: 36 launches, `11.793ms` total, `0.328ms` avg.
  - `kernel_binop_c_bf16_n_8`: 216 launches, `16.402ms` total.
  - `reduce_kernel_in_bf16_in_size_8_out_bf16_out_size_8_acc_f32`: 72 launches, `5.139ms` total.
  - `wkv7_pretrain_forward_output_kernel_f_bf16`: 36 launches, `22.732ms` total.
  - `mix6_forward_kernel_f__n_1`: 36 launches, `14.198ms` total.
  - `channel_mixer_relu_square_forward_kernel_f__n_2`: 36 launches, `13.818ms` total.
- Interpretation: this clean-cache profile confirms the combine kernel is active and the old generic binop/reduce launch surface remains reduced. The acceptance blocker is still the known marginal `cell_0000/channel_mixer` row, not an activation or GatedReadout correctness issue.
- Next command: query the clean sqlite for launch config and adjacent kernel sequence around `gated_readout_combine_forward_kernel_f_`.

### SQLite Launch Query

- Launch config: `gated_readout_combine_forward_kernel_f_` launched 36 times with `grid=(8192,12,1)`, `block=(64,1,1)`, `registers/thread=21`, `static_smem=0`, `dynamic_smem=8`, total `11.793ms`, avg `327.595us`.
- First local sequence around the new kernel:
  - preceding time-mixer generic tail includes `kernel_binop_c_bf16_n_8` launches around `67-68us`, a `reduce_kernel...` around `22us`, and a `kernel_binop_c_bf16_n_1` around `104us`.
  - `gated_readout_combine_forward_kernel_f_`: `305.056us`.
  - following output projection matmul: `matmul_entry_lhs_bf16_lhs_size_1_rhs_bf16_rhs_size_1_acc_bf16_acc_size_8`, `161.856us`.
  - next stages are layer norm and channel mixer.
- Decision: close this candidate as positive but not fully clean because the unrelated known `cell_0000/channel_mixer` row keeps standard compare at `75/76`. The next GatedReadout-specific implementation branch should test a 32-thread one-warp version that computes two lanes per thread and removes the inter-warp shared-memory reduction/synchronization from this kernel. This is a different kernel parameter/implementation boundary and needs a fresh branch/note before editing.
