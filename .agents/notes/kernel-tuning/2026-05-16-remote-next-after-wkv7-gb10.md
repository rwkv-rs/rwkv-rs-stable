# Remote Next Surface After WKV7 GB10

- Date: 2026-05-16 16:56 +0800.
- Branch/worktree: `kernel-tuning-remote-next-after-wkv7-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits plus retained/reverted kernel tuning changes. This attempt owns only this attribution note unless a later entry explicitly opens an implementation branch.
- Prior-note and memory search commands:
  - `rg -n "shared-lanes|shared_lanes|wkv7|state-scan|residual|ordered-256|LayerNorm|10\\.100\\.1\\.253|GB10" .agents/notes/kernel-tuning /root/.codex/memories/MEMORY.md 2>/dev/null`
  - `rg -n "Decision: pending|pending\\.|Next implementation|Next action|next implementation|next .*branch|do not implement|open a separate|not yet|unresolved|Remaining" .agents/notes/kernel-tuning/2026-05-16-*.md`
- Matched prior evidence:
  - `2026-05-16-wkv7-shared-state-lanes-gb10.md`: shared-lanes candidates were activation-clean but much slower; they were reverted, and GB10 still selects `row_tile_64`. WKV7 remains about `22.083ms / 36` launches with `grid=(12,16,1)`, `block=(64,1,1)`, `regs/thread=108`, `dynamic_smem=1536`.
  - `2026-05-16-wkv7-state-scan-design-gb10.md`: full time/chunk scan or state-handoff is a larger algorithmic project; shared-state lanes were the smaller follow-up and are now rejected.
  - `2026-05-16-key-prepare-warps-gb10.md`: warps-per-cube tuning selected `block=64` but slightly worsened targeted time and was reverted.
  - `2026-05-16-gated-readout-groupnorm-combine-gb10.md` and related GatedReadout notes: groupnorm/combine is kept; warp32, rowpack, and sumsq-variance variants are closed.
  - `2026-05-16-lm-head-projection-loss-fusion-analysis.md`: projection+loss fusion is too broad without a tensor-core-grade fused operator; current safer result is only timing-contract exposure of `lm_head/projection`.
  - `2026-05-16-channel-mixer-matmul-fusion-analysis.md`: channel-mixer matmul/activation fusion would require extending Cubek/TMA matmul or repeating known negative forced-Cube/Burn-reference paths.
  - `2026-05-16-layernorm-runtime-dispatch.md` plus skill rules: GB10 can use LayerNorm `256`, while local deterministic path must remain `1024`; ordered-256 is device-boundary failed and must not be reintroduced without a guard.
  - Remote `ncu` counter collection is blocked by `ERR_NVGPUCTRPERM`, so this attempt can use `nsys` kernel time, launch geometry, registers, and shared memory, but not achieved occupancy, warp efficiency, or memory-throughput counters.
- Machine/GPU: remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`; local GPU must not be used.
- Remote run directory: `/home/caizus/Projects/Packages/rwkv-rs-stable`, synchronized non-git mirror.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, heads `12`, head size `64`.
- Baseline path: `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Hypothesis: after rejecting WKV7 shared-lanes, the correct next action is to refresh remote current-source attribution before opening another implementation branch. Several obvious surfaces are duplicate-closed or too broad; a fresh current profile should decide whether any smaller project-owned boundary remains.
- Candidate parameters: none. This is a profiling/attribution attempt only.
- Expected keep/revert boundary: keep this note as evidence. If top remaining project-owned surfaces are already duplicate-closed or require broad Cubek/TMA operator work, do not edit kernels. If a non-duplicate small boundary appears, open a fresh implementation branch and note before changing code.
- Next command: sync the current local source mirror to `10.100.1.253` excluding `.git`, `target`, `weights`, and `results`; run remote `cargo check -p rwkv-test --features cuda`, standard compare, then a short `nsys` profile and sqlite summary.

## Remote Preflight

- Host: `spark-35ac`.
- Toolchain: `rustc 1.95.0`, `cargo 1.95.0`.
- GPU: `NVIDIA GB10`, compute capability `12.1`, utilization `0%`.
- Baseline: `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000` exists.
- Nsight Systems: `2025.3.2`.
- Next command: sync the current local mirror to the remote run directory, excluding generated and heavy paths.

## Remote Sync

- Command: `rsync -az --delete -e 'ssh -i ~/.ssh/id_ed25519_dgx_spark_windows ...' --exclude '.git/' --exclude 'target/' --exclude 'weights/' --exclude 'results/' ./ caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable/`.
- Result: completed successfully.
- Remote source provenance: synchronized mirror of local branch `kernel-tuning-remote-next-after-wkv7-gb10-20260516`, including the current dirty-tree source and this note.
- Next command: remote compile check `cargo check -p rwkv-test --features cuda`.

## Remote Compile Check

- Command: `cargo check -p rwkv-test --features cuda` on `10.100.1.253`.
- Result: passed in dev check profile.
- Next command: standard remote compare with regenerated GB10 baseline, `repeat=3`, `warmup=1`, capturing output in `target/rwkv-test/remote-next-after-wkv7-compare.log`.

## Remote Standard Compare

- Command: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline ~/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1`, captured in `target/rwkv-test/remote-next-after-wkv7-compare.log`.
- Binary/build provenance: remote release binary was already up to date after sync.
- Activation result: passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: clean, `timing_summary compared=76 passed=76 failed=0 missing=0 extra=0 ignored=1 actual_total_ms=74.629 baseline_total_ms=175.887 speedup=2.36x`.
- Module timing signal:
  - `cells/*/time_mixer`: `43.397ms` vs `120.381ms`, `2.77x`.
  - `cells/*/channel_mixer`: `20.001ms` vs `25.759ms`, `1.29x`.
  - `loss/l2wrap_cross_entropy`: `4.593ms` vs `5.957ms`, `1.30x`.
- Decision: current remote source is activation-clean and timing-clean. Use this release binary for profiler attribution.
- Next command: short remote `nsys` profile with `repeat=1,warmup=1`, then export/query sqlite. The profiled compare timing is instrumentation-only and not an acceptance gate.

## Remote Nsys Profile

- Command: `nsys profile --trace=cuda,nvtx --sample=none --force-overwrite=true --output=target/rwkv-test/nsys-next-after-wkv7-gb10 target/release/rwkv-test compare-rwkv-nn --color never --baseline ... --repeat 1 --warmup 1`.
- Result: exited nonzero because the profile run uses `repeat=1,warmup=1` against a `repeat=3,warmup=1` baseline. This is expected for profiler mode.
- Activation sanity inside profile run: passed, `activation_summary compared=54 passed=54 failed=0`.
- Profile-mode rough timing: invalid for acceptance due timing-profile mismatch, but still positive at `actual_total_ms=75.978 baseline_total_ms=175.887 speedup=2.31x`.
- Generated artifact: `target/rwkv-test/nsys-next-after-wkv7-gb10.nsys-rep`.
- Next command: export the nsys report to sqlite and summarize CUDA kernel groups by demangled name and launch geometry.

## Nsys Export

- Command: `nsys export --type sqlite --force-overwrite=true --output target/rwkv-test/nsys-next-after-wkv7-gb10.sqlite target/rwkv-test/nsys-next-after-wkv7-gb10.nsys-rep`.
- Result: export completed successfully.
- Generated artifact: `target/rwkv-test/nsys-next-after-wkv7-gb10.sqlite`.
- Next command: query the sqlite for top CUDA kernel groups and launch metadata.

## Nsys Kernel Ranking

- Command: remote Python sqlite query grouped by demangled kernel name plus launch geometry.
- Top current groups:
  - lm-head projection Cubek/TMA matmul: `47.868ms / 3`, `grid=(4096,16,1)`, `block=(32,12,1)`, regs `73`, dynamic smem `27648`.
  - recurring Cubek/TMA matmul groups: `26.765ms / 144`, `20.955ms / 36`, `19.047ms / 36`.
  - `wkv7_pretrain_forward_output_kernel_f_bf16`: `22.114ms / 36`, `grid=(12,16,1)`, `block=(64,1,1)`, regs `108`, dynamic smem `1536`.
  - `mix6_forward_kernel_f__n_1`: `14.584ms / 36`, `grid=(24576,1,1)`, `block=(32,8,1)`, regs `32`.
  - `channel_mixer_relu_square_forward_kernel_f__n_2`: `14.170ms / 36`, `grid=(49152,1,1)`, `block=(32,8,1)`, regs `16`.
  - `lm_head_l2wrap_ce_forward_row_kernel_f__i_i32`: `13.879ms / 3`, `grid=(8192,1,1)`, `block=(1024,1,1)`, regs `53`.
  - `gated_readout_combine_forward_kernel_f_`: `11.865ms / 36`, `grid=(8192,12,1)`, `block=(64,1,1)`, regs `25`.
  - `key_prepare_forward_64_kernel_f_`: `9.523ms / 36`, `grid=(24576,1,1)`, `block=(128,1,1)`, regs `24`.
  - `kernel_binop_c_bf16_n_8`: `8.686ms / 72`.
  - `layer_norm_forward_kernel_f_`: `6.724ms / 78`, `block=(256,1,1)`.
  - `value_residual_gate_forward_kernel_f__n_2`: `6.519ms / 33`.
- Interpretation:
  - Top absolute time is still Cubek/TMA matmul, especially `lm_head/projection`, which is already exposed as optional timing but implementation fusion is a broad operator-boundary project.
  - WKV7 remains the largest project-owned custom kernel, but row-tile and shared-lanes are closed; state scan is a larger algorithmic project.
  - `mix6`, channel-mixer, lm-head row, GatedReadout, key-prepare, LayerNorm, and value-residual all have existing notes. Before opening another branch, inspect current autotune logs to see whether any remaining top custom kernel is using an unexpected candidate or stale cache.
- Next command: read current remote autotune logs for `mix6`, `channel_mixer`, `value_residual_gate`, `gated_readout`, `lm_head_l2wrap_ce`, `layer_norm`, and `wkv7` without changing code or cache.

## Remote Autotune Log Audit

- Command: read `target/autotune/**/*.json.log` entries matching the current train-kernel names on `10.100.1.253`.
- Current selected candidates:
  - `channel_mixer_mix_forward`: `line_size_8`, median `76.384us`.
  - `channel_mixer_relu_square_forward`: `line_size_2`, median `433.634us`; `line_size_4` is nearly tied at `435.042us`, `line_size_8` is worse/noisier.
  - `layer_norm_forward`: `block_256`, median `102.881us`; this is the verified GB10 policy.
  - `lm_head_l2wrap_ce_forward`: `block_1024`, median `4.242ms`; `block_512` and `block_256` are slower.
  - `learning_rate_gate_forward`: `line_size_8`, median `88.672us`.
  - `mix6_forward`: `line_size_1`, median `411.554us`; wider line sizes are close but not faster.
  - `value_residual_gate_forward`: `line_size_2`, median `220.868us`; wider candidates are near-tied but not selected.
  - `weight_decay_transform_forward`: `line_size_4`, median `81.505us`; `line_size_8` nearly tied at `81.825us`.
  - `wkv7_pretrain_output_forward`: `row_tile_64`, median `597.990us`.
- Stale-cache caveat:
  - A `key_prepare-forward-64` autotune log still exists from the rejected key-prepare branch, but current local and remote source both use fixed `HEAD64_WARPS_PER_CUBE = 4`, and the current nsys launch is `block=(128,1,1)`. Treat that log as stale generated residue, not a live dispatch result.
- Interpretation:
  - The current top custom kernels are not stale-cache artifacts. Their candidate choices match prior kept/rejected notes.
  - A broad solution would target Cubek/TMA matmul or a new lm-head projection+loss operator, but that crosses the current small-kernel tuning boundary.
  - The remaining small remote-only gap is the post-projection `lm_head_l2wrap_ce` row kernel. Online-softmax was tested on GB10 and rejected, but the simpler direct target-logit load was only tested locally. This materially changes the hardware/baseline boundary and is still scoped to the row kernel.
- Decision: close this attribution branch as evidence. Open a fresh implementation branch for a remote GB10 direct target-logit row-kernel candidate, with a strict profiler-based keep/revert rule. Do not touch WKV7, residual add, LayerNorm ordered-256, channel-mixer matmul, or Cubek matmul in that branch.
