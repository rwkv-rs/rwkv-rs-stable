# LM Head Target Logit GB10

- Date: 2026-05-16 17:15 +0800.
- Branch/worktree: `kernel-tuning-lm-head-target-logit-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits plus retained/reverted kernel tuning changes. This attempt owns only `crates/rwkv-nn/src/kernels/train/lm_head_l2wrap_ce/kernel.rs` and this note unless later entries explicitly expand scope.
- Prior-note search commands:
  - `rg -n "lm_head_l2wrap_ce|forward row|online softmax|logsumexp|target-logit|target logit|atomic|GB10|remote|10\\.100\\.1\\.253|loss row" .agents/notes/kernel-tuning /root/.codex/memories/MEMORY.md -S`
  - `rg -n "lm_head_l2wrap_ce_forward_row_kernel|target_logit|local_target|block_reduce_sum_f32\\(local_target" crates/rwkv-nn/src/kernels/train/lm_head_l2wrap_ce -S`
- Matched prior evidence:
  - `2026-05-16-lm-head-forward-target-logit.md` tried direct target-logit loading locally. It passed activation but did not improve local timing and was reverted. This branch is allowed only because the hardware, baseline, and current source boundary change to remote GB10.
  - `2026-05-16-lm-head-forward-online-softmax-gb10.md` tried one-pass online softmax on GB10; it passed activation but profiler showed the row kernel regressed (`14.185ms / 3` vs previous `12.787ms / 3`), so it was reverted.
  - `2026-05-16-remote-next-after-wkv7-gb10.md`: current remote source is activation/timing clean (`54/54`, `76/76`, `2.36x`), and current nsys has `lm_head_l2wrap_ce_forward_row_kernel_f__i_i32` at `13.879ms / 3`, `grid=(8192,1,1)`, `block=(1024,1,1)`, regs `53`.
  - Current remote autotune log selects `block_1024` for the loss row kernel; block-size-only tuning is not the issue.
- Machine/GPU: remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`; local GPU must not be used.
- Remote run directory: `/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Shape/dtype: CUDA BF16 `rwkv_lm`, logits `[16,512,65536]`, targets `[16,512]`, rows `8192`, vocab `65536`.
- Baseline path: `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Kernel/stage: `lm_head_l2wrap_ce_forward_row_kernel`, forward row loss only.
- Hypothesis: the current row kernel scans the target logit during the second full-vocab pass and then performs a block sum reduction over a mostly zero `local_target`. Loading the target logit directly from `row_start + target` on `UNIT_POS == 0` removes one shared-memory reduction and branch from the scan. This is small and may not move a memory-bound kernel, but GB10 has not measured this exact candidate.
- Candidate parameters: keep existing `block_{256,512,1024}` autotune candidates. Only change target-logit extraction inside the selected row kernel algorithm.
- Accuracy risk: no mathematical change when `target < vocab_size`; invalid targets still produce zero loss. Activation compare must remain `54/54`.
- Expected keep/revert boundary:
  - Keep only if remote compile passes, remote activation passes, standard compare remains clean or no worse, and short nsys shows `lm_head_l2wrap_ce_forward_row_kernel` improves versus the current `13.879ms / 3` profile.
  - If profiler time is flat or worse, revert even if standard compare is noisy-positive, because the candidate targets only this row kernel.
- Next edit: change `lm_head_l2wrap_ce_forward_row_kernel` to remove `local_target` from the vocab scan and let `UNIT_POS == 0` load `target_logit` directly.

## Implementation Edit

- Edited `crates/rwkv-nn/src/kernels/train/lm_head_l2wrap_ce/kernel.rs`.
- Removed `local_target` accumulation inside the second vocab scan.
- Removed `block_reduce_sum_f32(local_target, ...)`.
- `UNIT_POS == 0` now directly loads `target_logit` from `inputs.logits[row_start + target]` when `target < vocab_size`.
- Next command: rustfmt the edited kernel file and run local compile check `cargo check -p rwkv-nn --features cuda,fusion` without local GPU execution.

## Local Non-GPU Check

- Command: `rustup run nightly rustfmt crates/rwkv-nn/src/kernels/train/lm_head_l2wrap_ce/kernel.rs && cargo check -p rwkv-nn --features cuda,fusion`.
- Result: passed.
- No local GPU execution was performed.
- Next command: scoped rsync of the edited kernel file and this note to `10.100.1.253`, then remote compile check `cargo check -p rwkv-test --features cuda`.

## Remote Sync And Compile

- Command: scoped `rsync -azR` of:
  - `crates/rwkv-nn/src/kernels/train/lm_head_l2wrap_ce/kernel.rs`
  - `.agents/notes/kernel-tuning/2026-05-16-lm-head-target-logit-gb10.md`
- Destination: `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable/`.
- Remote compile command: `cargo check -p rwkv-test --features cuda`.
- Result: sync completed and remote compile passed.
- Next command: standard remote compare with `repeat=3,warmup=1`, capturing output in `target/rwkv-test/remote-lm-head-target-logit-gb10-compare.log`.

## Remote Standard Compare

- Command: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline ~/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1`, captured in `target/rwkv-test/remote-lm-head-target-logit-gb10-compare.log`.
- Binary/build provenance: remote release binary rebuilt after the row-kernel edit.
- Activation result: passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: positive but not fully clean, `timing_summary compared=76 passed=75 failed=1 missing=0 extra=0 ignored=1 actual_total_ms=75.161 baseline_total_ms=175.887 speedup=2.34x`.
- Failed row: known unrelated `timing/cells/cell_0000/channel_mixer.time.json`, `1.812ms` vs `1.668ms`, `0.92x`.
- Targeted timing row: `loss/l2wrap_cross_entropy` was `4.410ms` vs baseline `5.957ms`, `1.35x`. Previous attribution branch measured `4.593ms` in standard compare, so this may be a small improvement or run variance.
- Decision: run nsys before keep/revert; standard compare alone is insufficient for this row-kernel-only change.
- Next command: short remote `nsys` profile with `repeat=1,warmup=1`, export sqlite, and compare `lm_head_l2wrap_ce_forward_row_kernel_f__i_i32` against the current pre-edit `13.879ms / 3` reference.

## Remote Nsys Profile

- Command: `nsys profile --trace=cuda,nvtx --sample=none --force-overwrite=true --output=target/rwkv-test/nsys-lm-head-target-logit-gb10 target/release/rwkv-test compare-rwkv-nn --color never --baseline ... --repeat 1 --warmup 1`.
- Result: exited nonzero because profile mode uses `repeat=1,warmup=1` against a `repeat=3,warmup=1` baseline. This is expected.
- Activation sanity inside profile run: passed, `activation_summary compared=54 passed=54 failed=0`.
- Profile-mode timing is invalid for acceptance due mismatch, but rough total stayed positive: `actual_total_ms=79.527 baseline_total_ms=175.887 speedup=2.21x`.
- Generated artifact: `target/rwkv-test/nsys-lm-head-target-logit-gb10.nsys-rep`.
- Next command: export the nsys report to sqlite and summarize the row-loss kernel.

## Nsys Export And Row-Kernel Query

- Command: `nsys export --type sqlite --force-overwrite=true --output target/rwkv-test/nsys-lm-head-target-logit-gb10.sqlite ...`, then remote Python sqlite query for lm-head row/finalize and adjacent BF16 matmul groups.
- Result:
  - `lm_head_l2wrap_ce_forward_row_kernel_f__i_i32`: `12.851ms / 3`, avg `4283.573us`, `grid=(8192,1,1)`, `block=(1024,1,1)`, regs `40`, dynamic smem `128`.
  - `lm_head_l2wrap_ce_forward_finalize_kernel_f_`: `0.044ms / 3`, avg `14.784us`.
  - Pre-edit current attribution reference from `2026-05-16-remote-next-after-wkv7-gb10.md`: `13.879ms / 3`, avg `4626.443us`, regs `53`.
- Interpretation:
  - The direct target-logit load improves the targeted row kernel by about `1.028ms / 3` launches, roughly `7.4%` for the row kernel in this profile.
  - Register count drops from `53` to `40`, which is consistent with removing the `local_target` accumulation and reduction.
  - Standard compare still had one unrelated channel-mixer edge failure and a slightly worse total, so run one more standard compare with the same binary before the keep decision.
- Next command: repeat standard remote compare without rebuilding, same `repeat=3,warmup=1`, to check whether the non-target total/edge row is noise.

## Remote Compare Repeat

- Command: `target/release/rwkv-test compare-rwkv-nn --color never --baseline ~/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1`, captured in `target/rwkv-test/remote-lm-head-target-logit-gb10-compare-repeat.log`.
- Activation result: passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: positive but still one known edge failure, `timing_summary compared=76 passed=75 failed=1 missing=0 extra=0 ignored=1 actual_total_ms=75.061 baseline_total_ms=175.887 speedup=2.34x`.
- Failed row: `timing/cells/cell_0000/channel_mixer.time.json`, `1.684ms` vs `1.668ms`, `0.99x`.
- Targeted timing row: `loss/l2wrap_cross_entropy` was `4.534ms`, still above baseline and within the current run-to-run range.
- Decision: keep the direct target-logit kernel on the GB10 branch. The standard compare is activation-clean and speedup-positive, the only timing failure is the already known unrelated channel-mixer edge, and the nsys row-kernel measurement improved from `13.879ms / 3` to `12.851ms / 3` with lower register pressure.
- Keep/revert state: keep `crates/rwkv-nn/src/kernels/train/lm_head_l2wrap_ce/kernel.rs` with direct target-logit loading. Missing lower-level ncu counters remain blocked on `10.100.1.253` by `ERR_NVGPUCTRPERM` from prior attempts.
