# Remote Post Target-Logit Attribution GB10

- Date: 2026-05-16 17:15 +0800.
- Branch/worktree: `kernel-tuning-remote-post-target-logit-attribution-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits plus retained/reverted kernel tuning changes. This attempt owns only this attribution note unless a later entry explicitly opens an implementation branch.
- Prior-note and memory search commands:
  - `rg -n "post-target|target-logit.*attribution|remote.*after.*target|next.*target-logit|lm-head-target-logit" .agents/notes/kernel-tuning -S`
  - `rg -n "10\\.100\\.1\\.253|GB10|target-logit|remote current|nsys|ncu|ERR_NVGPUCTRPERM|WKV7|channel_mixer|lm_head|LayerNorm|LocalTuner" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md /root/.codex/memories/MEMORY.md -S`
- Matched prior evidence:
  - `2026-05-16-lm-head-target-logit-gb10.md` kept direct target-logit loading on remote GB10: activation `54/54`, total speedup about `2.34x`, and row kernel improved from `13.879ms / 3` to `12.851ms / 3`.
  - `2026-05-16-remote-next-after-wkv7-gb10.md` already profiled the pre-target-logit source and selected direct target-logit as the next small remote-only implementation surface.
  - WKV7 shared-lanes, WKV7 row-tile retries, key-prepare warps, GatedReadout warp/rowpack variants, channel-mixer forced Cube/Burn/reference/fusion, LayerNorm ordered-256, and lm-head online-softmax/atomic/prune variants are duplicate-closed under their recorded boundaries.
  - Remote `ncu` counters on `10.100.1.253` are blocked by `ERR_NVGPUCTRPERM`; this attempt can use `nsys` timing, launch geometry, registers, and shared memory, but cannot claim achieved occupancy, warp execution efficiency, or memory throughput.
- Machine/GPU: remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`; local GPU must not be used.
- Remote run directory: `/home/caizus/Projects/Packages/rwkv-rs-stable`, synchronized non-git mirror.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, heads `12`, head size `64`.
- Baseline path: `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Hypothesis: after the kept direct target-logit change, the correct next step is a fresh remote attribution pass on the current source before opening another implementation branch. If the remaining top custom kernels are all duplicate-closed or require broad Cubek/TMA matmul work, record that and stop instead of forcing a small edit.
- Candidate parameters: none. This is a profiling/attribution attempt only.
- Expected keep/revert boundary: keep this note as evidence. If a non-duplicate small boundary appears, open a new implementation branch and note before changing source. If the only high-impact path is broad matmul/operator design, do not edit kernels in this branch.
- Next command: remote preflight on `10.100.1.253`, sync the current source mirror excluding `.git`, `target`, `weights`, and `results`, run remote standard compare, then short `nsys` profile and sqlite summary.

## Remote Preflight

- Command: `ssh -i ~/.ssh/id_ed25519_dgx_spark_windows ... 'hostname; rustc --version; cargo --version; nvidia-smi --query-gpu=...; nsys --version; test -d baseline'`.
- Host: `spark-35ac`.
- Toolchain: `rustc 1.95.0`, `cargo 1.95.0`.
- GPU: `NVIDIA GB10`, compute capability `12.1`, utilization `0%`.
- Nsight Systems: `2025.3.2`.
- Baseline: `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000` exists.
- Next command: sync the current local mirror to the remote run directory, excluding `.git/`, `target/`, `weights/`, and `results/`.

## Remote Sync

- Command: `rsync -az --delete -e 'ssh -i ~/.ssh/id_ed25519_dgx_spark_windows ...' --exclude '.git/' --exclude 'target/' --exclude 'weights/' --exclude 'results/' ./ caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable/`.
- Result: completed successfully.
- Remote source provenance: synchronized mirror of local branch `kernel-tuning-remote-post-target-logit-attribution-gb10-20260516`, including the current dirty-tree source and this note.
- Next command: remote compile check `cargo check -p rwkv-test --features cuda`.

## Remote Compile Check

- Command: `cargo check -p rwkv-test --features cuda` on `10.100.1.253`.
- Result: passed in dev check profile.
- Next command: standard remote compare with `repeat=3,warmup=1`, capturing output in `target/rwkv-test/remote-post-target-logit-attribution-compare.log`.

## Remote Standard Compare

- Command: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline ~/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1`, captured in `target/rwkv-test/remote-post-target-logit-attribution-compare.log`.
- Binary/build provenance: remote release binary was already up to date after sync.
- Activation result: passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: clean, `timing_summary compared=76 passed=76 failed=0 missing=0 extra=0 ignored=1 actual_total_ms=72.438 baseline_total_ms=175.887 speedup=2.43x`.
- Module timing signal:
  - `cells/*/time_mixer`: `42.408ms` vs `120.381ms`, `2.84x`.
  - `cells/*/channel_mixer`: `19.710ms` vs `25.759ms`, `1.31x`.
  - `loss/l2wrap_cross_entropy`: `4.384ms` vs `5.957ms`, `1.36x`.
  - `lm_head`: `0.094ms` vs `0.573ms`, `6.12x`.
- Decision: current remote source is activation-clean and timing-clean after target-logit. Use this release binary for profiler attribution.
- Next command: short remote `nsys` profile with `repeat=1,warmup=1`, then export/query sqlite. The profiled compare timing is instrumentation-only and not an acceptance gate.

## Remote Nsys Profile

- Command attempt: `nsys profile ...; status=$?; ...` on remote zsh.
- Result: invalid wrapper command because `status` is a read-only zsh variable. Do not use this as profiler evidence.
- Next command: rerun the same `nsys` profile boundary with shell variable `rc` instead.

## Remote Nsys Profile Rerun

- Command: `nsys profile --trace=cuda,nvtx --sample=none --force-overwrite=true --output=target/rwkv-test/nsys-post-target-logit-attribution-gb10 target/release/rwkv-test compare-rwkv-nn --color never --baseline ... --repeat 1 --warmup 1`.
- Result: exited nonzero because the profile run uses `repeat=1,warmup=1` against a `repeat=3,warmup=1` baseline. This is expected for profiler mode.
- Profile-mode activation/timing sanity: the log still reports speedup-positive total, `actual_total_ms=74.224 baseline_total_ms=175.887 speedup=2.37x`, but all timing rows fail because of the profile mismatch. This timing table is not an acceptance gate.
- Generated artifact: `target/rwkv-test/nsys-post-target-logit-attribution-gb10.nsys-rep`.
- Next command: export the nsys report to sqlite and summarize CUDA kernel groups by demangled name and launch metadata.

## Nsys Export And Kernel Ranking

- Export command: `nsys export --type sqlite --force-overwrite=true --output target/rwkv-test/nsys-post-target-logit-attribution-gb10.sqlite target/rwkv-test/nsys-post-target-logit-attribution-gb10.nsys-rep`.
- First SQLite query attempt: invalid because shell quoting broke a Python f-string. The sqlite file was already exported; do not use the failed query.
- Corrected query result, top current groups:
  - lm-head projection Cubek/TMA matmul: `46.921ms / 3`, `grid=(4096,16,1)`, `block=(32,12,1)`, regs `73`, dynamic smem `27648`.
  - recurring Cubek/TMA matmul groups: `26.159ms / 144`, `22.075ms / 36`, `19.582ms / 36`.
  - `wkv7_pretrain_forward_output_kernel_f_bf16`: `22.412ms / 36`, `grid=(12,16,1)`, `block=(64,1,1)`, regs `108`, dynamic smem `1536`.
  - `channel_mixer_relu_square_forward_kernel_f__n_2`: `14.572ms / 36`, `grid=(49152,1,1)`, `block=(32,8,1)`, regs `16`.
  - `mix6_forward_kernel_f__n_1`: `14.479ms / 36`, `grid=(24576,1,1)`, `block=(32,8,1)`, regs `32`.
  - `lm_head_l2wrap_ce_forward_row_kernel_f__i_i32`: `13.083ms / 3`, `grid=(8192,1,1)`, `block=(1024,1,1)`, regs `40`, dynamic smem `128`.
  - `gated_readout_combine_forward_kernel_f_`: `11.680ms / 36`, `grid=(8192,12,1)`, `block=(64,1,1)`, regs `25`, dynamic smem `8`.
  - `key_prepare_forward_64_kernel_f_`: `9.326ms / 36`, `grid=(24576,1,1)`, `block=(128,1,1)`, regs `24`.
  - `value_residual_gate_forward_kernel_f__n_2`: `6.583ms / 33`.
  - `layer_norm_forward_kernel_f_`: `6.186ms / 78`, `block=(256,1,1)`, regs `40`.
- Interpretation: current post-target-logit source is clean and the top remaining time is split between broad Cubek/TMA matmul and previously investigated custom kernels. Before choosing another branch, inspect autotune logs to confirm no current candidate selection is stale or surprising.
- Next command: read current remote autotune logs for the top custom kernels.

## Remote Autotune Log Audit

- Command: remote Python scan of `target/autotune/**/*.json.log` for `channel_mixer`, `mix6`, `value_residual_gate`, `gated_readout`, `lm_head_l2wrap_ce`, `layer_norm`, `wkv7`, and `key_prepare`.
- Current selected candidates:
  - `channel_mixer_mix_forward`: `line_size_8`, median `76.384us`.
  - `channel_mixer_relu_square_forward`: `line_size_2`, median `433.634us`; `line_size_4` is close at `435.042us`.
  - `layer_norm_forward`: `block_256`, median `102.881us`; this is the verified GB10 policy.
  - `lm_head_l2wrap_ce_forward`: `block_1024`, median `4.242ms`; `block_512` and `block_256` are slower.
  - `mix6_forward`: `line_size_1`, median `411.554us`; wider line sizes are near but not faster.
  - `value_residual_gate_forward`: `line_size_2`, median `220.868us`; `line_size_4` is close at `221.572us`.
  - `wkv7_pretrain_output_forward`: `row_tile_64`, median `597.990us`.
- Stale-cache caveat:
  - The `key_prepare-forward-64` autotune log from the rejected warps-per-cube branch still exists and records `warps_per_cube_2`, but current source uses fixed `HEAD64_WARPS_PER_CUBE = 4`, and current nsys launch is `block=(128,1,1)`. Treat this log as stale generated residue, not a live dispatch result.
- Interpretation:
  - Current top custom kernels are not stale-cache artifacts. Their selected candidates match prior kept/rejected notes.
  - The only stale candidate log found is for key-prepare, and it is proven non-live by source plus launch geometry.
  - The remaining high-impact surface is broad Cubek/TMA matmul, especially `lm_head/projection`. Existing projection+loss analysis says a fused implementation would require a new operator/API and GEMM-quality tiling/backward design, not another post-logits row-loss tweak.
  - WKV7 remains the largest project-owned custom kernel, but row-tile-only and shared-lanes attempts are rejected. A real WKV7 improvement needs a separate algorithmic state-scan/time-split design, not another candidate-set tweak.
- Decision: close this attribution branch as current remote evidence. Do not open another small implementation branch from this profile unless the boundary is materially different from the recorded negatives. The next honest implementation track is a broader design branch, either WKV7 state-scan/time-split or lm-head projection-loss operator design.
- Keep/revert state: no source changes in this branch beyond this note. Current remote source remains activation-clean and timing-clean at `2.43x` standard compare on `10.100.1.253`.
- Next command: sync this updated note to the remote mirror for run provenance.
