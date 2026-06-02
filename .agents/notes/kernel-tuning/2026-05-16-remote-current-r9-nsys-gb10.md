# Remote Current R9 Nsys GB10

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-remote-current-r9-nsys-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits plus retained/reverted kernel tuning changes. This attempt owns only this read-only profiler note unless a later entry explicitly opens a source or cache-policy edit.
- User constraint: debug first on remote `10.100.1.253`; do not use local GPU timing.
- Prior-note and memory search commands already run:
  - `rg -n "remote current|nsys-current|current-binary|R9|projection|activation_summary|timing_summary|matmul|WKV7|Mix6|channel_mixer|lm_head|gated_readout|key_prepare|value_residual|LayerNorm" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md /root/.codex/memories/MEMORY.md -S`
  - `find .agents/notes/kernel-tuning -maxdepth 1 -type f -printf '%T@ %f\n' | sort -n | tail -20`
- Matched prior evidence:
  - `2026-05-16-projection-matmul-autotune-audit-gb10.md` shows the current remote binary passes the R9 projection baseline cleanly: activation `54/54`, timing `77/77`, total speedup `1.51x`, projection `1.01x`.
  - Existing remote `nsys` sqlite artifacts mostly predate the current remote release binary. Some later artifacts also contain rejected code boundaries such as WKV7 direct-value or residual fusion.
  - Remote `ncu` counters remain blocked by `RmProfilingAdminOnly: 1`; `nsys` launch/register/shared-memory metadata is the available profiler evidence on this host.
  - Duplicate guards close WKV7 row-tile/shared-lanes/segment/low-rank/direct-value, key-prepare warps, GatedReadout rowpack/warp32/sumsq, channel-mixer forced Cube/Burn/fusion/line-size, lm-head row-kernel variants, residual-add plain wiring, and projection+loss naive fusion.
- Machine/GPU: remote `caizus@10.100.1.253`, host `spark-35ac`, `NVIDIA GB10`, compute capability `12.1`.
- Remote run directory: `/home/caizus/Projects/Packages/rwkv-rs-stable`, synchronized non-git mirror.
- Baseline path: `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_r9_20260516/rwkv_lm/bf16/case_000000`.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, heads `12`, head size `64`, vocab `65536`.
- Hypothesis: after the current-binary R9 clean pass, a fresh short `nsys` profile is needed to rank the actual remaining kernel surfaces without relying on stale or rejected-code profiler artifacts.
- Candidate parameters: none. This is a read-only profiler refresh; do not edit code, clear caches, or force candidates in this branch.
- Expected keep/revert boundary: keep this note as current profiler evidence. If the top surfaces remain duplicate-closed or broad Cubek/TMA matmul, do not open a kernel branch from the profile alone. If a new non-duplicate project-owned surface appears, open a separate branch and prewrite a note before editing.
- Next command: run one remote `nsys profile` with `repeat=1,warmup=1` against the R9 baseline using the current release binary, export sqlite, and rank kernel groups. This profiler run is not an acceptance gate; the R9 compare above is the acceptance evidence.

## Remote Nsys Profile

- Command: remote `nsys profile --trace=cuda,nvtx --sample=none --force-overwrite=true --output=target/rwkv-test/nsys-current-r9-gb10 target/release/rwkv-test compare-rwkv-nn --repeat 1 --warmup 1 --baseline ...`, followed by sqlite export to `target/rwkv-test/nsys-current-r9-gb10.sqlite`.
- Result:
  - Profile command exit was `1` because the profiled compare intentionally used `repeat=1,warmup=1` against an R9 `repeat=9,warmup=3` timing baseline, so all timing rows report a timing profile mismatch. Activation/timing acceptance is not drawn from this run.
  - The instrumented run still reported total `actual_total_ms=94.378`, baseline `130.484`, speedup `1.38x`; `lm_head/projection` was `14.997ms` under the profiler run, with profile mismatch noted.
  - `nsys export` succeeded and generated `target/rwkv-test/nsys-current-r9-gb10.sqlite`.
- Validity: profiler artifact is valid for CUDA kernel ranking and launch metadata. It is not a timing acceptance gate.
- Next command: query the sqlite for grouped CUDA kernel totals, launch geometry, register counts, and shared-memory metadata.

## Kernel Ranking

- Command: remote Python sqlite grouping over `target/rwkv-test/nsys-current-r9-gb10.sqlite`, grouped by kernel name and launch geometry.
- Top current groups:
  - `lm_head/projection` Cubek/TMA BF16 matmul: `46.174ms / 3`, `grid=(4096,16,1)`, `block=(32,12,1)`, `regs=73`, dynamic shared memory `27648`.
  - Recurring Cubek/TMA BF16 matmuls: `26.447ms / 144`, `21.327ms / 36`, `18.860ms / 36`.
  - `wkv7_pretrain_forward_output_kernel_f_bf16`: `23.171ms / 36`, `grid=(12,16,1)`, `block=(64,1,1)`, `regs=127`, dynamic shared memory `1536`.
  - `mix6_forward_kernel_f__n_1`: `14.484ms / 36`, `block=(32,8,1)`, `regs=32`.
  - `channel_mixer_relu_square_forward_kernel_f__n_2`: `14.220ms / 36`, `block=(32,8,1)`, `regs=16`.
  - `lm_head_l2wrap_ce_forward_row_kernel_f__i_i32`: `13.470ms / 3`, `block=(1024,1,1)`, `regs=40`, dynamic shared memory `128`.
  - `gated_readout_combine_forward_kernel_f_`: `12.164ms / 36`, `grid=(8192,12,1)`, `block=(64,1,1)`, `regs=25`.
  - `key_prepare_forward_64_kernel_f_`: `9.401ms / 36`, `block=(128,1,1)`, `regs=24`.
  - `kernel_binop_c_bf16_n_8`: `8.477ms / 72`, matching the known residual-add attribution.
  - `value_residual_gate_forward_kernel_f__n_2`: `6.506ms / 33`.
  - `layer_norm_forward_kernel_f_`: `6.205ms / 78`, `block=(256,1,1)`.
- Interpretation:
  - Current top surfaces are the same families as prior evidence: broad Cubek/TMA matmuls, WKV7 output, Mix6, channel-mixer ReLU-square, lm-head row loss, GatedReadout, key-prepare, residual binops, value-residual, and LayerNorm.
  - No new small project-owned kernel surface appears from the current-binary profile.
  - WKV7 remains the largest project-owned custom kernel, but its row-tile/shared-lanes/segment/low-rank/direct-value design families are duplicate-closed. The current profile alone does not justify reopening them.
  - The fresh profile still supports keeping `lm_head/projection` as broad Cubek/TMA work; the projection row passes R9 compare and the generic matmul cache selected the fastest TMA candidate.
- Next command: read current remote autotune logs for the top custom kernels to make sure the fresh profile did not expose an unexpected stale candidate selection.

## Current Autotune Log Audit

- Command: remote Python parser over `target/autotune/0.10.0/device-0-0-cuda/rwkv_nn-*.json.log`, printing recent entries and fastest mean timings.
- Result:
  - `channel_mixer_mix_forward`: latest hardware-fingerprint key ranks `line_size_4` and `line_size_8` as a near tie (`83.125us` vs `83.637us`); current nsys suffix shows `n_4`.
  - `channel_mixer_relu_square_forward`: `line_size_2` remains the fastest mean (`428.042us`), matching current nsys `n_2`.
  - `layer_norm_forward`: `block_256` remains clearly fastest (`93.620us` vs `123.294us` for `512`) under the GB10 deterministic policy.
  - `lm_head_l2wrap_ce_forward`: `block_1024` remains fastest (`4529.818us` vs `4742.971us` for `512`).
  - `mix6_forward`: `line_size_1` remains fastest (`408.685us`), matching current nsys `n_1`.
  - `value_residual_gate_forward`: `line_size_2` remains fastest by a small margin (`220.747us`), matching current nsys `n_2`.
  - `weight_decay_transform_forward`: latest means rank `line_size_4` fastest (`89.187us`) with `line_size_8` near (`90.403us`); current nsys suffix for the retained run is `n_2`, but this small kernel is only `2.493ms / 36` and prior evidence kept the transform for removing the Burn softplus chain rather than for a line-size micro-win.
  - `wkv7_pretrain_output_forward`: `row_tile_64` remains clearly fastest (`606.959us` vs `944.969us` for `32` and `1843.061us` for `16`).
  - Stale `key_prepare-forward-64` autotune log still reports a rejected `warps_per_cube_2`, but current source and nsys use fixed four warps (`block=(128,1,1)`). Treat it as stale residue, not live dispatch.
- Interpretation:
  - Current custom-kernel choices are either live and consistent with nsys or already explained stale residue.
  - Near ties exist in several line-size tuners, but prior branches already tested the tempting surfaces and total compare did not improve. This audit does not justify another line-size-only branch.
  - No cache-clearing action is warranted.
- Decision: close this branch as current profiler evidence unless a later task gets privileged `ncu` counters or introduces a materially new algorithm/operator boundary.
- Next command: run `git diff --check` for this note and sync it to the remote mirror.

## Validation And Sync

- Command: `git diff --check -- .agents/notes/kernel-tuning/2026-05-16-remote-current-r9-nsys-gb10.md`.
- Result: passed.
- Remote sync command: scoped `rsync -azR` of this note to `/home/caizus/Projects/Packages/rwkv-rs-stable` on `10.100.1.253`.
- Result: passed.
- Continuation: because remote `ncu` counters are blocked, use the fresh `nsys` launch/register/shared-memory metadata plus CUDA device limits to estimate theoretical occupancy constraints for the top kernels. This does not replace achieved occupancy, warp execution efficiency, or memory throughput, but it identifies whether launch size, registers, or shared memory can plausibly cap occupancy.
- Next command: query CUDA device properties on `10.100.1.253` with a temporary `/tmp` CUDA program, then compute per-kernel theoretical occupancy from the `nsys-current-r9-gb10.sqlite` top groups.

## Theoretical Occupancy Estimate

- Device property command: compiled and ran a temporary `/tmp/rwkv_cuda_props.cu` with `/usr/local/cuda-13.0/bin/nvcc`.
- GB10 CUDA device limits:
  - compute capability `12.1`, SM count `48`, warp size `32`.
  - max threads/SM `1536`, max blocks/SM `24`, registers/SM `65536`.
  - shared memory/SM `102400`, default shared memory/block `49152`, opt-in shared memory/block `101376`.
- Occupancy-estimate command: remote Python query over `target/rwkv-test/nsys-current-r9-gb10.sqlite`, using nsys `block`, `grid`, `registersPerThread`, and static/dynamic shared memory. This is a coarse theoretical estimate; it ignores allocation granularity and does not provide achieved occupancy or warp efficiency.
- Top project/custom kernel estimates:
  - `wkv7_pretrain_forward_output_kernel_f_bf16`: `23.171ms / 36`, theory `33.3%`, effective wave occupancy `16.7%`, only `4.00` blocks/SM available from the grid, `active_blocks/SM=8`, `regs/thread=127`, smem/block `1536`. This confirms WKV7 is both register-constrained and grid-underfilled on GB10.
  - `mix6_forward_kernel_f__n_1`: `14.484ms / 36`, theory/wave `100%`, `regs/thread=32`, no shared memory. The bottleneck is not theoretical occupancy from registers/smem.
  - `channel_mixer_relu_square_forward_kernel_f__n_2`: `14.220ms / 36`, theory/wave `100%`, `regs/thread=16`, no shared memory.
  - `lm_head_l2wrap_ce_forward_row_kernel_f__i_i32`: `13.470ms / 3`, theory/wave `66.7%`, one `1024`-thread block per SM limit, `regs/thread=40`, smem/block `128`.
  - `gated_readout_combine_forward_kernel_f_`: `12.164ms / 36`, theory/wave `100%`, `regs/thread=25`, smem/block `8`.
  - `key_prepare_forward_64_kernel_f_`: `9.401ms / 36`, theory/wave `100%`, `regs/thread=24`, no shared memory.
  - `kernel_binop_c_bf16_n_8`: `8.477ms / 72`, theory/wave `100%`; already attributed to residual adds, not a fresh generic-binop target.
  - `value_residual_gate_forward_kernel_f__n_2`: `6.506ms / 33`, theory/wave `100%`.
  - `layer_norm_forward_kernel_f_`: `6.205ms / 78`, theory/wave `100%`, `block=(256,1,1)`, `regs/thread=40`, smem/block `32`.
- Interpretation:
  - WKV7 remains the only large custom kernel where nsys metadata plus device limits show a clear occupancy structural issue: high register pressure and only `192` blocks per launch (`4` blocks/SM) under the current shape.
  - The other major custom kernels are not theoretically occupancy-limited by launch geometry, registers, or shared memory. Without privileged `ncu`, the remaining questions for those are memory throughput / scheduler efficiency, which this estimate cannot measure.
  - This reinforces the previous decision: do not retry line-size/block-size tweaks for Mix6, ChannelMixer, key-prepare, GatedReadout, or LayerNorm from this evidence. A real WKV7 improvement needs a materially different algorithm that increases parallel work or reduces per-row register state without repeating the rejected segment/direct-value families.
- Next command: sync the updated note to the remote mirror.

## Final Sync

- Command: `git diff --check -- .agents/notes/kernel-tuning/2026-05-16-remote-current-r9-nsys-gb10.md`.
- Result: passed.
- Remote sync command: scoped `rsync -azR` of this note to `/home/caizus/Projects/Packages/rwkv-rs-stable` on `10.100.1.253`.
- Result: passed.
- Final decision for this branch: keep this note as the current remote R9 profiler and theoretical occupancy evidence. No source code or cache files were changed.

## WKV7 Skill Guard Update

- Follow-up duplicate check after the occupancy estimate:
  - `row_tile=16` is already negative because it repeats input loads and regressed timing despite increasing block count.
  - `shared_lanes_{4,8}` is already negative because shared-state reductions/synchronization were much slower than `row_tile_64`.
  - direct-value, output-factor, dense segment recompute, and low-rank segment scan are already closed.
- Edited `.agents/skills/kernel-tuning/SKILL.md` to record the current GB10 WKV7 structural occupancy result:
  - `192` blocks/launch, about `4 blocks/SM`.
  - `regs/thread=127`.
  - theoretical occupancy `33.3%`, effective wave occupancy about `16.7%`.
  - A future WKV7 branch must name a materially different algorithm that increases parallel work or reduces per-row register state without repeating the closed families.
- Next command: run `git diff --check` for this note and the updated skill, then sync both to `10.100.1.253`.

## Skill Sync

- Command: `git diff --check -- .agents/notes/kernel-tuning/2026-05-16-remote-current-r9-nsys-gb10.md .agents/skills/kernel-tuning/SKILL.md`.
- Result: passed.
- Remote sync command: scoped `rsync -azR` of this note and `.agents/skills/kernel-tuning/SKILL.md` to `/home/caizus/Projects/Packages/rwkv-rs-stable` on `10.100.1.253`.
- Result: passed.
- Final state: no kernel source or cache files changed in this continuation; only the current profiler note and kernel-tuning skill guard were updated.
