# Remote Current Profile GB10

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-remote-current-profile-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this branch inherits broad unrelated workspace changes plus the retained kernel-tuning source from prior branches. This attempt is profiling/attribution only; it must not edit kernels. Any implementation change found from this evidence gets a new branch and note.
- Prior-note/source search command:
  - `rg -n "key_prepare|warps_per_cube|row_tile|WKV7|gated_readout|row-pack|rowpack|LayerNorm|ordered-256|channel_mixer|lm_head|LocalTuner|remote|10\\.100\\.1\\.253" .agents/notes/kernel-tuning crates/rwkv-nn/src/kernels/train -S`
  - `rg -n "kernel-tuning|LayerNorm|LocalTuner|gated_readout|goal-gap|10\\.100\\.1\\.253|GB10" /root/.codex/memories/MEMORY.md`
- Matched prior evidence:
  - Remote `10.100.1.253` / GB10 is the primary timing source for current work; local GPU timing is intentionally avoided because the user is using it for other work.
  - `key_prepare` warps-per-cube tuner is a recorded negative GB10 attempt and was reverted to fixed `HEAD64_WARPS_PER_CUBE=4`.
  - WKV7 row-tile-only tuning is closed; forced `row_tile=16` regressed, and a real time/chunk split would be a separate algorithmic design.
  - LayerNorm ordered-256 is closed until a trace/device-side accuracy guard exists; runtime hardware/shape dispatch currently permits GB10 `block=256` while protecting local deterministic behavior.
  - GatedReadout warp32 and row-pack are closed negative/neutral attempts; retained code is the 64-thread/two-warp combine.
  - Duplicate guards remain closed for residual-add/Burn-add A/B, channel-mixer forced Cube/Burn-reference/fusion, LocalTuner bypass, lm-head target-logit/atomic/online-softmax/prune-256, key-prepare warps, and WKV7 row-tile restriction.
- Machine/GPU: remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`; do not run local GPU.
- Remote run directory: `/home/caizus/Projects/Packages/rwkv-rs-stable`, synchronized non-git copy.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, heads `12`, head size `64`, regenerated remote baseline under `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Hypothesis: after reverting failed key-prepare and GatedReadout row-pack attempts, the next correct step is to refresh `nsys` attribution for the current live remote source/binary and rank remaining large surfaces. Do not pick another implementation branch from stale profiles.
- Candidate parameters: none in this profiling branch.
- Expected keep/revert boundary: keep this note as evidence only. If the top ranked surface is a project-owned unfused operation with no duplicate negative note, open a fresh implementation branch before editing. If the top surfaces are Cubek matmul internals or already closed candidates, document that and choose the next viable boundary.
- Planned command: sync this note to remote if needed, run a standard remote compare to confirm current activation/timing provenance, then run `nsys` with `repeat=1,warmup=1` and query kernel totals plus launch geometry from the generated sqlite.

## Remote Access Attempt 1

- Command: `rsync -avR .agents/notes/kernel-tuning/2026-05-16-remote-current-profile-gb10.md caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable/ ...`.
- Result validity: invalid/no remote data. SSH without the explicit key failed with `Permission denied (publickey,password)`, and rsync exited `255`.
- Follow-up search result: prior notes record the working access pattern as `ssh -i ~/.ssh/id_ed25519_dgx_spark_windows -o BatchMode=yes -o ConnectTimeout=10 caizus@10.100.1.253`.
- Next command: retry the note sync and remote preflight using the explicit key.

## Remote Preflight

- Command: explicit-key rsync of this note, then remote `hostname`, `nvidia-smi`, repo path, and profiler artifact listing.
- Result: sync succeeded. Host is `spark-35ac`; GPU is `NVIDIA GB10`, compute capability `12.1`, utilization `0%`; remote run directory is `/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Existing newest profiler artifact before this branch is `target/rwkv-test/nsys-gated-readout-rowpack.sqlite` from May 16 01:43, followed by the key-prepare and value-residual artifacts. Because row-pack was later reverted, do not use that artifact as the current live-state profile.
- Next command: run a standard remote compare with regenerated baseline to rebuild/confirm the current post-revert release binary before profiling.

## Remote Standard Compare

- Host: `caizus@10.100.1.253`.
- Command: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1`, output saved to `target/rwkv-test/remote-current-profile-compare.log`.
- Build provenance: release binary was already up to date; Cargo finished in `0.19s`.
- Activation result: passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: clean, `timing_summary compared=76 passed=76 failed=0 actual_total_ms=87.653 baseline_total_ms=175.887 speedup=2.01x`.
- Module timing signal: `cells/*/time_mixer` `56.009ms`, `2.15x`; `cells/*/channel_mixer` `20.199ms`, `1.28x`; `loss/l2wrap_cross_entropy` `4.519ms`, `1.32x`.
- Interpretation: current post-revert remote state is activation-valid and speedup-positive. Proceed to `nsys` attribution on this release binary.
- Next command: run remote `nsys profile --trace=cuda,nvtx --stats=true --force-overwrite=true -o target/rwkv-test/nsys-current-gb10 target/release/rwkv-test compare-rwkv-nn --color never --baseline ... --repeat 1 --warmup 1`, then query the generated sqlite.

## Remote Nsys

- Host: `caizus@10.100.1.253`.
- Command: `nsys profile --trace=cuda,nvtx --stats=true --force-overwrite=true -o target/rwkv-test/nsys-current-gb10 target/release/rwkv-test compare-rwkv-nn --color never --baseline ... --repeat 1 --warmup 1`, output saved to `target/rwkv-test/remote-current-profile-nsys.log`.
- Result: profiler run completed and generated:
  - `target/rwkv-test/nsys-current-gb10.nsys-rep`
  - `target/rwkv-test/nsys-current-gb10.sqlite`
- Activation inside profiler run: passed (`54/54`). Timing comparison rows intentionally fail due to `repeat=1` vs baseline `repeat=3`; use only profiler attribution from this run.
- `cuda_gpu_kern_sum` top groups:
  - Cubek BF16 matmul, `120.738ms / 327`, `42.9%`.
  - `wkv7_pretrain_forward_output_kernel_f_bf16`, `21.671ms / 36`, avg `601.970us`.
  - `kernel_binop_c_bf16_n_8`, `16.422ms / 216`.
  - `mix6_forward_kernel_f__n_1`, `14.568ms / 36`.
  - `channel_mixer_relu_square_forward_kernel_f__n_2`, `14.139ms / 36`.
  - Cubek size-8 BF16 matmul, `13.102ms / 174`.
  - `lm_head_l2wrap_ce_forward_row_kernel_f__i_i32`, `12.743ms / 3`.
  - `gated_readout_combine_forward_kernel_f_`, `11.676ms / 36`.
  - `kernel_scalar_binop_c_bf16_n_8`, `11.323ms / 216`.
  - `key_prepare_forward_64_kernel_f_`, `9.492ms / 36`.
- Interpretation: current top custom surfaces remain WKV7, mix6, channel-mixer relu-square, lm-head row, gated-readout combine, and key-prepare. Several of these are duplicate-closed; need launch-geometry query and source attribution before picking a fresh branch.
- Next command: query `target/rwkv-test/nsys-current-gb10.sqlite` for top kernel groups with demangled names, grid/block, registers, and shared memory, especially splitting Cubek matmul shapes.

## SQLite Launch Query Attempt 1

- Command: remote Python `sqlite3` query over `target/rwkv-test/nsys-current-gb10.sqlite`, grouping by demangled name and launch geometry.
- Result validity: invalid query. SQL string quoting around `printf('id:%d', ...)` was mangled across the remote shell boundary and failed with `sqlite3.OperationalError: unrecognized token: ":"`.
- Next command: rerun the query without fallback `printf` string formatting.

## SQLite Launch Query

- Command: remote Python `sqlite3` query over `target/rwkv-test/nsys-current-gb10.sqlite`, grouping by demangled name and launch geometry.
- Top launch groups:
  - Cubek lm-head projection matmul: `46.670ms / 3`, `grid=(4096,16,1)`, `block=(32,12,1)`, `regs=73`, dynamic smem `27648`.
  - Cubek repeated BF16 matmul: `25.225ms / 144`, `grid=(16,48,1)`, `block=(32,12,1)`, `regs=73`, dynamic smem `27648`.
  - `wkv7_pretrain_forward_output_kernel_f_bf16`: `21.671ms / 36`, `grid=(12,16,1)`, `block=(64,1,1)`, `regs=108`, dynamic smem `1536`.
  - Channel-mixer Cubek matmuls: `21.257ms / 36` and `19.146ms / 36`, `regs=122`, dynamic smem `26624`.
  - `mix6_forward_kernel_f__n_1`: `14.568ms / 36`, `grid=(24576,1,1)`, `block=(32,8,1)`, `regs=32`, no smem.
  - `channel_mixer_relu_square_forward_kernel_f__n_2`: `14.139ms / 36`, `grid=(49152,1,1)`, `block=(32,8,1)`, `regs=16`, no smem.
  - `lm_head_l2wrap_ce_forward_row_kernel_f__i_i32`: `12.743ms / 3`, `grid=(8192,1,1)`, `block=(1024,1,1)`, `regs=53`, dynamic smem `128`.
  - `gated_readout_combine_forward_kernel_f_`: `11.676ms / 36`, `grid=(8192,12,1)`, `block=(64,1,1)`, `regs=21`, dynamic smem `8`.
  - `key_prepare_forward_64_kernel_f_`: `9.492ms / 36`, `grid=(24576,1,1)`, `block=(128,1,1)`, `regs=24`, no smem.
- Interpretation:
  - Cubek matmul and channel-mixer matmul internals remain the largest total surface, but prior notes already reject forced Cube/Burn-reference/fusion inside the current workspace scope.
  - WKV7 row-tile-only tuning is closed; a real recurrence-scan design needs a separate algorithm plan.
  - `mix6_forward_kernel_f__n_1` is a notable current project-owned surface. Since `mix6` already has a hardware-rich `LocalTuner`, `n_1` may indicate candidate eligibility is constrained by tensor layout rather than that line-size `1` is truly fastest.
- Next command: inspect remote `mix6` autotune log and local source shape/layout rules to determine whether a non-duplicate axis-eligibility branch exists, analogous to the already kept value-residual-gate axis fix.

## Mix6 Dispatch Check

- Remote autotune log: `target/autotune/0.10.0/device-0-0-cuda/rwkv_nn-kernels-train-time_mixer-mix6-forward-rwkv7-time-mixer-mix6-forward.json.log`.
- Key fields: CUDA BF16, `num_elements=8388608`, `B=16`, `T=512`, `D=768`, rows `8192`, GB10 hardware fingerprint, `max_line_size=8`, deterministic.
- Candidate result:
  - `line_size_1`: selected fastest, mean `411.457us`, median `411.554us`.
  - `line_size_8`: mean `416.940us`, median `416.817us`.
  - `line_size_4`: mean `417.776us`, median `417.617us`.
  - `line_size_2`: mean `419.740us`, median `416.770us`.
  - `line_size_16/32/64`: skipped by candidate validity.
- Source/layout inspection:
  - Public mix6 scale tensors are `[1,1,D]`, so using embedded axis `2` for the scale tensors is valid.
  - The current key does not accidentally force `max_line_size=1`; it exposes candidates up to `8`.
- Decision: do not open a mix6 vector-axis or line-size branch. The current `n_1` launch is a real GB10 tuner decision, not the value-residual-style axis bug. A different mix6 attempt would need a new algorithm/kernel layout hypothesis, not another vector-width retry.
- Next command: inspect current channel-mixer relu-square autotune log and source boundary because it is the next large project-owned non-matmul surface in the current profile.

## Channel-Mixer Elementwise Dispatch Check

- Remote autotune logs:
  - `channel-mixer-mix-forward`: key has CUDA BF16, rows `8192`, innermost `768`, `max_line_size=8`, selected `line_size_8` with median `76.384us`.
  - `channel-mixer-relu-square-forward`: key has CUDA BF16, rows `8192`, innermost `3072`, `max_line_size=8`, in-place `true`, selected `line_size_2` with median `433.634us`.
- Source inspection:
  - `channel_mixer_mix_forward_kernel` is already fused token-shift + scale mix.
  - `channel_mixer_relu_square_forward_kernel` is a simple in-place ReLU-square over the matmul projection output.
  - Candidate eligibility is not constrained to `1`; the current `n_2` relu-square launch is the tuner winner on GB10.
- Decision: do not open a channel-mixer line-size or Burn-reference branch. Prior notes already reject forced matmul/Burn-reference/fusion attempts, and current logs show the elementwise tuner is functioning.
- Next command: inspect timeline neighborhoods for the remaining generic BF16 binop/scalar-binop groups (`kernel_binop_c_bf16_n_8`, `kernel_scalar_binop_c_bf16_n_8`) to determine whether any unfused project-owned chain remains outside duplicate-closed kernels.

## Generic BF16 Timeline Attribution

- Command: remote Python timeline query around generic BF16 binop/scalar-binop/reduce kernels in `target/rwkv-test/nsys-current-gb10.sqlite`.
- Evidence:
  - The large repeated generic clusters occur after the WeightPrepare LoRA matmuls and before `key_prepare`/WKV7, and again around gated readout/group-norm/channel-mixer boundaries.
  - The cluster after `weight_decay_lora.forward(weight_decay_input)` matches source-level `let weight_decay = -softplus(-weight_decay_lora_result, 1.0) - 0.5;` in `crates/rwkv-nn/src/modules/time_mixer/weight_prepare.rs`.
  - `learning_rate_gate` and `value_residual_gate` are already wired custom kernels; their nearby generic work is mostly the remaining weight-decay transform and LoRA/matmul-adjacent operations.
- Source search:
  - No prior `weight_decay`/`softplus` custom kernel note exists under `.agents/notes/kernel-tuning`.
  - CubeCL supports `exp()` and `ln()` in kernels, so a project-owned fused transform can compute `-(ln(1 + exp(-(bias + input)))) - 0.5` with f32 intermediates and BF16 output.
- Decision: close this profile branch as current-state attribution and open a fresh implementation branch for a `weight_decay_transform` forward kernel. This is a new boundary: it does not repeat learning-rate/value-residual gate wiring, channel-mixer line-size, residual-add, WKV7 row-tile, or lm-head row variants.
