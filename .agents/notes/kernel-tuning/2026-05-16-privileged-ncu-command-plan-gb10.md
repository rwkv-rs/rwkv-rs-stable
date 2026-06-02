# Privileged NCU Command Plan GB10

- Date: 2026-05-16.
- Branch/worktree: originally `kernel-tuning-current-goal-audit-after-r9-20260516`; refreshed on `kernel-tuning-privileged-ncu-plan-current-gb10-20260516` after the current top-surface guard audit; continued on `kernel-tuning-ncu-permission-runbook-gb10-20260516` for the explicit 253-only debugging constraint.
- Scope: command plan only. Do not run these commands in the current `caizus` permission boundary because `10.100.1.253` reports `RmProfilingAdminOnly: 1` and previous ncu CSVs contain `ERR_NVGPUCTRPERM`.
- User constraint: continue remote-first on `10.100.1.253`; do not use the local GPU for timing until the user makes it available.
- Prior evidence:
  - `2026-05-16-current-goal-audit-after-r9.md` records remote R9 acceptance: activation `54/54`, timing `76/77`, total speedup `1.48x`, and only `lm_head/projection` at `0.99x`.
  - `2026-05-16-remote-current-after-lowrank-gb10.md` records the current top remote `nsys` surfaces and launch metadata.
  - `2026-05-16-completion-audit-after-guards.md` records current remote acceptance against the ordinary GB10 baseline: activation `54/54`, total speedup `2.42x`, plus the still-open local timing and ncu-counter gaps.
  - `2026-05-16-current-top-surface-guards-gb10.md` records current duplicate guards for `value_residual_gate`, Mix6, and channel-mixer line-size forcing.
  - `.agents/skills/kernel-tuning/SKILL.md` records the GB10 ncu blocker and says not to repeat counter runs until root enables counters or runs ncu privileged.
- Baseline for all commands:
  - Host: `10.100.1.253` / `spark-35ac`.
  - Repo: `/home/caizus/Projects/Packages/rwkv-rs-stable`.
  - Binary: rebuild or confirm `target/release/rwkv-test` from the current synced source before profiling.
  - Baseline path for current whole-model acceptance and profiler reproduction: `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
  - Projection-specific R9 baseline, only when investigating projection timing rows: `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_r9_20260516/rwkv_lm/bf16/case_000000`.
  - Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, head size `64`, vocab `65536`.
- Required ncu sections:
  - `SpeedOfLight`
  - `Occupancy`
  - `SchedulerStats`
  - `WarpStateStats`
  - `MemoryWorkloadAnalysis`
- Validity rules:
  - Treat compare timing under ncu as invalid because ncu serializes/instruments kernels.
  - Use ncu only for kernel counters: achieved occupancy, theoretical occupancy, warp execution / eligible warps, memory throughput, DRAM/L2 behavior, register pressure, shared memory, and stalls.
  - Keep the standard `compare-rwkv-nn --repeat 3 --warmup 1` result against `test_gen/rwkv_lm/bf16/case_000000` as the current whole-model acceptance timing source.
  - Use the R9 `--repeat 9 --warmup 3` projection baseline only for projection-row attribution, not as the whole-model acceptance baseline.
  - If ncu reports stale binary, missing kernel, or permission failure, mark the run invalid and do not draw a kernel conclusion.

## 253-Only Debugging Continuation

- User update: use `10.100.1.253` first because the local GPU is reserved for other work and local timing may be inaccurate.
- Continuation preflight commands:
  - local `git status --short --branch` confirmed branch `kernel-tuning-ncu-permission-runbook-gb10-20260516` and a broad dirty tree; this attempt owns only this note unless a later entry explicitly expands scope.
  - remote `hostname; pwd; nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader; grep RmProfilingAdminOnly /proc/driver/nvidia/params; command -v ncu; command -v nsys; ls -d ~/Projects/Packages/rwkv-rs-stable ~/Projects/Packages/rwkv-rs-test`.
  - remote result: host `spark-35ac`, GPU `NVIDIA GB10`, driver `580.95.05`, `RmProfilingAdminOnly: 1`, `nsys` present, `ncu` not in default `PATH` but expected at `/usr/local/cuda-13.0/bin/ncu`, stable/test mirrors present.
  - remote `git status --short --branch` in `/home/caizus/Projects/Packages/rwkv-rs-stable` failed with `fatal: not a git repository`; this remote is an rsync mirror without `.git`, so branch provenance is recorded locally and remote evidence must cite command logs, binary path, source mirror path, and note sync.
  - remote log inventory found current artifacts through `target/rwkv-test/nsys-current-r9-gb10.sqlite` and earlier clean compare logs including `remote-current-after-gb10-results-compare.log`.
- Next command under normal user permissions: run a fresh remote standard compare against `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`, captured as `target/rwkv-test/remote-253-only-current-compare.log`. This is allowed because it does not require hardware performance counters.
- Remote standard compare result:
  - Command: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1`, captured in `target/rwkv-test/remote-253-only-current-compare.log`.
  - Build provenance: Cargo reported `Finished release profile [optimized] target(s) in 0.21s`, then ran `target/release/rwkv-test`.
  - Activation: `activation_summary compared=54 passed=54 failed=0 missing=0 extra=0`.
  - Timing: `timing_summary compared=76 passed=76 failed=0 missing=0 extra=0 ignored=1 actual_total_ms=73.373 baseline_total_ms=175.887 speedup=2.40x`.
  - Decision: remote 253 ordinary compare remains clean and above `1.0`; this does not satisfy the blocked `ncu` counter requirement.
- Next normal-user command: run a short fresh `nsys` profile on the same remote binary and baseline, then summarize CUDA kernel time from the generated sqlite. This is the allowed profiler path while counters remain blocked.
- Remote `nsys` result:
  - Command: `nsys profile --trace=cuda,nvtx --sample=none --force-overwrite=true --output=target/rwkv-test/nsys-253-only-current-gb10 target/release/rwkv-test compare-rwkv-nn --color never --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 1 --warmup 1`.
  - Exit status: `1`, expected for this profiler command because the compare output marks all timing rows as `timing profile mismatch: actual repeat=1 warmup=1 baseline repeat=3 warmup=1`. Activation still passed `54/54`; profiler `.time.json` rows are invalid for acceptance.
  - Generated report: `/home/caizus/Projects/Packages/rwkv-rs-stable/target/rwkv-test/nsys-253-only-current-gb10.nsys-rep`.
  - Export command: `nsys export --type sqlite --force-overwrite=true --output target/rwkv-test/nsys-253-only-current-gb10.sqlite target/rwkv-test/nsys-253-only-current-gb10.nsys-rep`.
  - Generated sqlite: `/home/caizus/Projects/Packages/rwkv-rs-stable/target/rwkv-test/nsys-253-only-current-gb10.sqlite`.
- Remote `nsys` top CUDA kernel groups, grouped by demangled name plus launch geometry:
  - `matmul_entry_lhs_bf16_lhs_size_1_rhs_bf16_rhs_size_1_acc_bf16_acc_size_8`: `3` launches, `47.023ms`, grid `4096x16x1`, block `32x12x1`, `73` regs/thread, `27648` dynamic shared memory.
  - `matmul_entry_lhs_bf16_lhs_size_1_rhs_bf16_rhs_size_1_acc_bf16_acc_size_8`: `144` launches, `26.261ms`, grid `16x48x1`, block `32x12x1`, `73` regs/thread, `27648` dynamic shared memory.
  - `wkv7_pretrain_forward_output_kernel_f_bf16`: `36` launches, `22.294ms`, grid `12x16x1`, block `64x1x1`, `108` regs/thread, `1536` dynamic shared memory.
  - `matmul_entry_lhs_bf16_lhs_size_1_rhs_bf16_rhs_size_1_acc_bf16_acc_size_8`: `36` launches, `21.047ms`, grid `64x48x1`, block `32x8x1`, `122` regs/thread, `26624` dynamic shared memory.
  - `matmul_entry_lhs_bf16_lhs_size_1_rhs_bf16_rhs_size_1_acc_bf16_acc_size_8`: `36` launches, `20.651ms`, grid `16x48x1`, block `32x8x1`, `122` regs/thread, `26624` dynamic shared memory.
  - `mix6_forward_kernel_f__n_1`: `36` launches, `14.862ms`, grid `24576x1x1`, block `32x8x1`, `32` regs/thread.
  - `channel_mixer_relu_square_forward_kernel_f__n_2`: `36` launches, `14.510ms`, grid `49152x1x1`, block `32x8x1`, `16` regs/thread.
  - `lm_head_l2wrap_ce_forward_row_kernel_f__i_i32`: `3` launches, `14.007ms`, grid `8192x1x1`, block `1024x1x1`, `40` regs/thread, `128` dynamic shared memory.
  - `gated_readout_combine_forward_kernel_f_`: `36` launches, `11.931ms`, grid `8192x12x1`, block `64x1x1`, `25` regs/thread, `8` dynamic shared memory.
  - `key_prepare_forward_64_kernel_f_`: `36` launches, `9.388ms`, grid `24576x1x1`, block `128x1x1`, `24` regs/thread.
  - `kernel_binop_c_bf16_n_8`: `72` launches, `8.788ms`, grid `3072x1x1`, block `32x8x1`, `16` regs/thread.
  - `value_residual_gate_forward_kernel_f__n_2`: `33` launches, `6.466ms`, grid `12288x1x1`, block `32x8x1`, `16` regs/thread.
  - `layer_norm_forward_kernel_f_`: `78` launches, `6.350ms`, grid `8192x1x1`, block `256x1x1`, `40` regs/thread, `32` dynamic shared memory.
- Interpretation:
  - The allowed remote profiler path still points at the already-known top surfaces: Cubek/TMA matmuls, WKV7 output under-fill, wide elementwise/vector kernels, row loss, GatedReadout, KeyPrepare, residual binop, and LayerNorm `block_256`.
  - `ncu` is still required for achieved occupancy, warp efficiency, memory throughput, and stall reasons; `nsys` provides launch geometry, registers, shared memory, and elapsed CUDA kernel groups only.

## Admin Counter Unblock Options

- Source: NVIDIA's ERR_NVGPUCTRPERM guidance (`https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters`) says the error means the process lacks permission for GPU performance counters; Linux options are to run the profiled process with elevated privileges or enable non-admin counter access via the NVIDIA module parameter `NVreg_RestrictProfilingToAdminUsers=0`.
- Current machine check: `/proc/driver/nvidia/params` reports `RmProfilingAdminOnly: 1`, so normal-user `ncu` counter runs are expected to fail.
- Preferred low-disruption path for this host:
  1. Ask an admin to run the exact `ncu` command in this note under a privileged context from `/home/caizus/Projects/Packages/rwkv-rs-stable`, or grant the profiled process the needed administrative capability.
  2. If cluster policy allows non-admin counters, add a modprobe config such as `/etc/modprobe.d/nvidia-profiler.conf` containing `options nvidia NVreg_RestrictProfilingToAdminUsers=0`, rebuild initrd if required by the distro, reboot or reload the NVIDIA modules during a maintenance window, then verify `grep RmProfilingAdminOnly /proc/driver/nvidia/params` returns `0`.
  3. If using a container boundary later, the host must already allow counters or the container must be launched with the required administrative capability; changing the container alone is insufficient when the host restriction remains enabled.
- Do not attempt temporary module reload on this shared remote without explicit admin coordination because it requires stopping users of `/dev/nvidia*` and unloading NVIDIA modules.

## Commands To Run Under Privileged Counter Access

```bash
cd /home/caizus/Projects/Packages/rwkv-rs-stable
cargo build --release -p rwkv-test --features cuda
```

```bash
cd /home/caizus/Projects/Packages/rwkv-rs-stable
/usr/local/cuda-13.0/bin/ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*(layer_norm_forward|lm_head_l2wrap_ce_forward_row|wkv7_pretrain_forward_output|mix6_forward|channel_mixer_relu_square|gated_readout_combine|key_prepare_forward_64|value_residual_gate_forward).*' \
  --launch-count 240 \
  --section SpeedOfLight \
  --section Occupancy \
  --section SchedulerStats \
  --section WarpStateStats \
  --section MemoryWorkloadAnalysis \
  --csv \
  --log-file target/rwkv-test/ncu-top-owned-gb10.csv \
  target/release/rwkv-test compare-rwkv-nn \
    --color never \
    --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 \
    --repeat 1 \
    --warmup 1
```

```bash
cd /home/caizus/Projects/Packages/rwkv-rs-stable
/usr/local/cuda-13.0/bin/ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*matmul_entry_lhs_bf16_lhs_size_1_rhs_bf16_rhs_size_1_acc_bf16_acc_size_8.*' \
  --launch-count 12 \
  --section SpeedOfLight \
  --section Occupancy \
  --section SchedulerStats \
  --section WarpStateStats \
  --section MemoryWorkloadAnalysis \
  --csv \
  --log-file target/rwkv-test/ncu-lm-head-projection-matmul-gb10.csv \
  target/release/rwkv-test compare-rwkv-nn \
    --color never \
    --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_r9_20260516/rwkv_lm/bf16/case_000000 \
    --repeat 1 \
    --warmup 1
```

## Expected Analysis Targets

- `lm_head/projection` Cubek/TMA matmul:
  - Existing current `nsys`: largest projection-like Cubek/TMA group about `46.174ms / 3`, grid `(4096,16,1)`, block `(32,12,1)`, `73` registers/thread, dynamic shared memory `27648`.
  - Need ncu to determine whether near-parity or small projection deltas are limited by occupancy, tensor-core throughput, memory, or scheduler stalls.
- `wkv7_pretrain_forward_output_kernel_f_bf16`:
  - Existing current `nsys`: about `23.171ms / 36`, grid `(12,16,1)`, block `(64,1,1)`, `127` registers/thread, dynamic shared memory `1536`.
  - Prior designs failed because they increased global f32 tensors or register pressure; ncu should confirm under-fill vs register/latency limits before any new algorithm branch.
- `mix6_forward_kernel_f__n_1`, `channel_mixer_relu_square_forward_kernel_f__n_2`, `key_prepare_forward_64_kernel_f_`, `value_residual_gate_forward_kernel_f__n_2`, `weight_decay_transform_forward_kernel_f__n_*`, and gate kernels:
  - Existing notes say they are memory-heavy, fixed-policy, or candidate deltas are near noise.
  - ncu should confirm whether memory throughput is already near saturation before rejecting further vector/warp candidates.
- `layer_norm_forward_kernel_f_`:
  - Existing remote policy selects `block_256`; local deterministic path keeps larger block sizes due to drift.
  - ncu should record occupancy/register/shared-memory counters for the accepted GB10 path, but correctness still comes from trace activation.

## Keep/Stop Boundary

- Keep this note as a runbook while `ncu` counters are blocked.
- Do not execute these commands as the normal `caizus` user until the profiling restriction changes; repeated `ERR_NVGPUCTRPERM` would add no evidence.
- After privileged ncu data exists, open a fresh branch for exactly one new interpretation or implementation hypothesis, cite the generated CSV path, and record whether the issue is occupancy, register pressure, shared memory, memory coalescing, or numerical guardrails.
