# Remote GB10 Current nsys Diagnostic

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-remote-gb10-nsys-current-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits and retained kernel-tuning notes. This attempt owns only this note and any remote profiler artifacts explicitly named here unless a later entry opens a code edit.
- User constraint: debug first on remote `10.100.1.253`; do not use the local GPU because it may be used for other work and timing may be inaccurate.
- Prior-note search commands:
  - `rg -n "ordered-256|LayerNorm|channel_mixer|current goal|ncu|10\\.100\\.1\\.253|R9|local regenerated|projection" .agents/notes/kernel-tuning`
  - `rg -n "kernel-tuning|LayerNorm|ordered-256|channel_mixer|10\\.100\\.1\\.253|GB10|ncu|current-goal|R9|residual" /root/.codex/memories/MEMORY.md`
- Matched prior evidence:
  - `2026-05-16-current-goal-audit-after-r9.md` records the current remote R9 result: activation `54/54`, timing `76/77`, total speedup `1.48x`, only `lm_head/projection` at `0.99x`.
  - `.agents/skills/kernel-tuning/SKILL.md` records that `ncu` on `10.100.1.253` is blocked by `RmProfilingAdminOnly: 1`; do not repeat normal-user `ncu` counter runs in this boundary.
  - `2026-05-16-layernorm-ordered256-accuracy-rootcause.md` and `2026-05-16-layernorm-ordered256-reconstruct.md` record that intended ordered-256 LayerNorm math matches the `1024` reduction, while reconstructed device kernels still drift. Treat this as implementation/boundary failure, not inherent math.
  - Residual/value-residual duplicate guards are closed; do not rerun the same residual or value-residual vector-width attempts.
- Machine/GPU: remote `caizus@10.100.1.253`, expected host `spark-35ac`, `NVIDIA GB10`, compute capability `12.1`, repo `/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, head size `64`, vocab `65536`.
- Hypothesis: since counter-level `ncu` is unavailable on the remote host, the next useful remote-only step is to refresh `nsys` launch/kernel attribution for the current synchronized tree and pair it with the existing R9 compare result. This should identify the next non-duplicate implementation surface without claiming achieved occupancy or memory throughput.
- Commands planned:
  1. Remote preflight: host, GPU, repo path, existing R9 logs, binary/log mtimes, current `nsys` availability.
  2. If no current `nsys` report exists for the R9/current tree, run a short remote `nsys profile` around `target/release/rwkv-test compare-rwkv-nn --color never --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_r9_20260516 --repeat 1 --warmup 1`.
  3. Extract `cuda_gpu_kern_sum` and, if available, launch metadata from the report.
- Profiler caveat: `repeat=1,warmup=1` under `nsys` is for current kernel attribution only. It is not an acceptance timing run; R9 acceptance remains the existing `repeat=9,warmup=3` compare evidence unless a fresh standard compare is explicitly recorded.
- Expected keep/revert boundary: no code changes in this attempt. Keep the report if it names a next target; otherwise close as evidence that remote counter access is the real blocker.

## 2026-05-16 Remote Preflight Result

- Host/GPU: `spark-35ac`, `NVIDIA GB10`, compute capability `12.1`, driver `580.95.05`.
- `nsys`: `/usr/local/bin/nsys`, Nsight Systems `2025.3.2.474-253236389321v0`.
- Binary/log mtimes:
  - `target/release/rwkv-test`: `2026-05-16 04:52:19 -0500`.
  - `target/rwkv-test/remote-projection-r9-compare.log`: `2026-05-16 05:34:14 -0500`.
  - `target/rwkv-test/remote-projection-r9-rerun-compare.log`: `2026-05-16 05:34:56 -0500`.
- Existing current-ish profiler artifacts:
  - `target/rwkv-test/nsys-current-after-lowrank-gb10.{log,nsys-rep,sqlite}` at `2026-05-16 05:12-05:13 -0500`, after the current binary build and before the R9 compare rerun.
- Decision: inspect the existing `nsys-current-after-lowrank-gb10.sqlite` first. Do not run a new profile unless the sqlite is missing the kernel summary or is clearly a different source boundary.

## 2026-05-16 Existing nsys Summary

- Source: existing remote `target/rwkv-test/nsys-current-after-lowrank-gb10.sqlite`. No new profiler run was launched.
- Profile caveat: this report used `repeat=1,warmup=1`; its compare timing rows are intentionally invalid against the `repeat=3,warmup=1` baseline. Use only the CUDA kernel attribution and launch metadata.
- Activation in the profiled run: `compared=54 passed=54 failed=0`.
- Top CUDA kernel groups:
  - lm-head projection Cubek/TMA matmul: `46.376ms / 3`, avg `15458.635us`, `grid=(4096,16,1)`, `block=(32,12,1)`, regs/thread `73`, dynamic smem `27648`.
  - recurring Cubek/TMA matmul groups: `25.405ms / 144`, `21.061ms / 36`, `18.911ms / 36`, `8.143ms / 141`.
  - `wkv7_pretrain_forward_output_kernel_f_bf16`: `23.166ms / 36`, avg `643.502us`, `grid=(12,16,1)`, `block=(64,1,1)`, regs/thread `108`, dynamic smem `1536`.
  - `mix6_forward_kernel_f__n_1`: `14.579ms / 36`, avg `404.976us`, `grid=(24576,1,1)`, `block=(32,8,1)`, regs/thread `32`.
  - `channel_mixer_relu_square_forward_kernel_f__n_2`: `14.216ms / 36`, avg `394.880us`, `grid=(49152,1,1)`, `block=(32,8,1)`, regs/thread `16`.
  - `lm_head_l2wrap_ce_forward_row_kernel_f__i_i32`: `12.800ms / 3`, avg `4266.773us`, `grid=(8192,1,1)`, `block=(1024,1,1)`, regs/thread `40`, dynamic smem `128`.
  - `gated_readout_combine_forward_kernel_f_`: `11.721ms / 36`, avg `325.592us`, `grid=(8192,12,1)`, `block=(64,1,1)`, regs/thread `25`, dynamic smem `8`.
  - `key_prepare_forward_64_kernel_f_`: `9.467ms / 36`, avg `262.974us`, `grid=(24576,1,1)`, `block=(128,1,1)`, regs/thread `24`.
  - `kernel_binop_c_bf16_n_8`: `8.550ms / 72`.
  - `layer_norm_forward_kernel_f_`: `6.541ms / 78`, avg `83.856us`, `grid=(8192,1,1)`, `block=(256,1,1)`, regs/thread `40`, dynamic smem `32`.
  - `value_residual_gate_forward_kernel_f__n_2`: `6.524ms / 33`, avg `197.711us`, `grid=(12288,1,1)`, `block=(32,8,1)`, regs/thread `16`.
- CUDA runtime summary:
  - `cuEventSynchronize`: `238.212ms / 520`, profiler boundary only.
  - `cuLaunchKernel`: `45.381ms / 1320`.
  - `cuMemAllocAsync`, `cuModuleLoadData`, and host allocation remain visible but are not kernel implementation targets for this attempt.
- Interpretation:
  - The ranking matches the previous current-after-lowrank attribution, so this was not stale.
  - Projection is the largest cost, but the R9 projection timing contract already shows it is near parity (`0.99x`) while total remains `1.48x`; improving it means broad Cubek/TMA matmul work rather than another `rwkv-nn` row-loss tweak.
  - WKV7, GatedReadout warp32/row-pack/sumsq, key-prepare warps, channel-mixer forced Cube/Burn/reference/fusion/line-size, lm-head row variants, value-residual vector width, and LayerNorm ordered-256 are duplicate-closed under recorded notes.
  - With normal-user `ncu` blocked on 253, there is no new counter-level evidence to distinguish occupancy/register/memory causes for the remaining candidates on this host.
- Decision: no source edit on this branch. Treat this attempt as a current remote attribution refresh and sync the note to the remote mirror.

## 2026-05-16 Completion Audit Refresh

- Objective restated as concrete deliverables:
  1. Every new kernel tuning attempt uses a fresh branch/worktree and a prewritten note; failed attempts are reverted but their branches/notes remain.
  2. Tuned kernel dispatch keys include backend/runtime, GPU hardware identity or available hardware fingerprint, dtype, `d_model`, rows, candidate block/warp/vector parameters, in-place/alias state, and deterministic numeric boundary.
  3. LayerNorm must select by runtime/hardware/shape/numeric policy rather than a universal hard-coded block size.
  4. Burn/CubeCL `LocalTuner::execute` mechanics are understood: key construction, in-memory and persistent caches, warmup/profiling on miss, candidate representation, host lookup cost, and accuracy-guard limits.
  5. CUDA diagnosis must include warp/block launch shape, register/shared-memory pressure, memory/coalescing reasoning, BF16 reduction accuracy analysis, and `ncu` counters where permissions allow.
  6. Poor results must be diagnosed as design, implementation, measurement, or numeric-boundary failures before revert.
  7. Final acceptance requires trace-backed activation passing and speedup `>1.0` on both local machine and `10.100.1.253`.
- Prompt-to-artifact checklist:

| Requirement | Current evidence | Status |
| --- | --- | --- |
| Branch plus prewritten note for new attempts | This branch `kernel-tuning-remote-gb10-nsys-current-20260516` has this note before the remote current-state inspection. Existing notes record prior branches and keep/revert decisions. | Covered for current workflow |
| Failed attempts retained | LayerNorm ordered-256, WKV7 segment/low-rank, key-prepare warps, GatedReadout variants, channel-mixer variants, and lm-head variants all have retained notes and negative/keep decisions. | Covered |
| Runtime/backend in keys | Source grep confirms tuned train-kernel keys use `runtime: R::name(...)`. | Covered |
| GPU arch / compute capability | CubeCL `HardwareProperties` exposes hardware fingerprint fields, but not CUDA compute capability. `cubecl-cuda` computes `arch_version` internally, then stores SM count, warp/plane, cube/shared/vector limits, and tensor-core fields in public properties. Current keys include these public fields. | Partially covered by available fingerprint; literal CC is not publicly exposed |
| dtype, `d_model`, rows | LayerNorm, WKV7, lm-head, mix6, learning/value/weight/channel tuned keys include dtype and shape fields appropriate to their axis. | Covered for live tuned families |
| block size / warps / vector width | Candidate names/groups encode `block_*`, `line_size_*`, row tiles, reduce block and `bt_tile` combinations; warp count is derived from block size where not a separate live candidate. Closed warps-only attempts are documented. | Covered where live candidates exist |
| in-place / alias and deterministic boundary | Tuned keys include `is_in_place`/alias-equivalent state and `deterministic`; LayerNorm adds `deterministic_min_block_size`. | Covered |
| LayerNorm hardware/shape policy | `layer_norm/forward.rs` uses `LayerNormForwardAutotuneKey`, block candidates, and a GB10 BF16 D768 rows=8192 hardware-fingerprint allow path for `256`; other deterministic BF16 paths stay at the safer power-of-two boundary. | Covered for current boundary |
| LocalTuner mechanism | `2026-05-16-cubecl-local-tuner-findings.md` documents `LocalTuner::init`, `execute`, cache hit/miss, persistent checksum, 3 warmup + 10 profiled samples, and the lack of trace-baseline accuracy guard. | Covered |
| Host-side tuner overhead | Notes correctly separate miss tuning cost from steady-state compare timing. There is no evidence that cache-hit lookup dominates current GB10 millisecond-scale kernels. | Covered enough for current hot path; not a completed micro-overhead proof |
| CUDA launch/register/shared-memory metadata on 253 | This note records current `nsys` launch metadata for projection, WKV7, Mix6, channel-mixer, lm-head row, GatedReadout, key-prepare, LayerNorm, and value-residual. | Covered via nsys metadata |
| Remote achieved occupancy / warp efficiency / memory throughput | `.agents/skills/kernel-tuning/SKILL.md` and prior notes record `RmProfilingAdminOnly: 1`; normal `caizus` cannot run `ncu` counters. | Blocked by environment |
| BF16 reduction numeric diagnosis | Ordered-256 LayerNorm notes show CPU intended math matches `1024`, while device candidates drift; ordinary `256` can create BF16 cell-boundary deltas. | Covered |
| Poor-result root-cause handling | Negative notes distinguish design-lower-bound failures, implementation/boundary bugs, tuner/cache issues, measurement noise, and broad operator scope. | Covered |
| Remote speedup `>1.0` | R9 projection baseline result: activation `54/54`, timing `76/77`, total `1.48x`; previous remote clean baseline without projection row has `76/76` and `>2x`. | Covered for remote total speedup |
| Local speedup `>1.0` | Notes contain local regenerated-baseline total speedup `2.44x`, but local row-level timing and baseline provenance remain incomplete; user currently asked not to use local GPU. | Partially covered; not clean final acceptance |

- Completion decision: do not call `update_goal`. The objective remains incomplete because remote `ncu` counter evidence is blocked by permissions and local clean acceptance remains unresolved under the user's current local-GPU constraint.
- Next valid actions:
  - If privileged 253 access is provided, run the saved privileged `ncu` plan and classify remaining kernels by occupancy, scheduler/warp, memory, register, or shared-memory limits.
  - If local GPU becomes available, rerun local acceptance against the intended baseline and address the row-level/provenance gap.
  - Without either condition, only design-level work remains; do not open another small kernel branch from the current remote `nsys` list because the remaining small surfaces are duplicate-closed.

## 2026-05-16 Skill Update For Hardware Fingerprint Limit

- Scope: update `.agents/skills/kernel-tuning/SKILL.md` only.
- Reason: the audit found an important precision in the hardware-key requirement. CubeCL CUDA internally computes `arch_version`, but public `HardwareProperties` does not expose CUDA compute capability major/minor to kernel code. Current project keys should honestly be described as using the available hardware fingerprint, not literal CUDA CC.
- Expected effect: future tuning notes should not claim literal compute capability is covered unless a public API or project extension actually exposes it.
- Next edit: add a concise Burn/CubeCL key-design rule to the project skill.
- Result: added the rule to `.agents/skills/kernel-tuning/SKILL.md`. It names the public hardware fingerprint fields and forbids claiming literal CC coverage unless a public API or project extension exposes it.

## 2026-05-16 Compute Capability API Feasibility Check

- Commands:
  - `rg -n "trait CubeRuntime|fn name\\(|type Device|struct CudaDevice|arch_version|COMPUTE_CAPABILITY" .../cubecl-runtime-0.10.0 .../cubecl-cuda-0.10.0 .../burn-cubecl-0.21.0`
  - `sed -n '1,80p' .../cubecl-cuda-0.10.0/src/device.rs`
  - `sed -n '45,90p' .../cubecl-cuda-0.10.0/src/runtime.rs`
  - `sed -n '330,370p' .../cubecl-cuda-0.10.0/src/runtime.rs`
- Evidence:
  - `CudaDevice` contains only `index: usize`.
  - `CudaRuntime::name(...)` returns the static runtime string `"cuda"`.
  - `CudaServer::init` queries `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR` and `_MINOR` into `arch_version`, then uses it for WMMA feature registration, tensor-core counts, and type support.
  - Public `HardwareProperties` does not store `arch_version`, CUDA major, or CUDA minor.
  - `CubeTensor` exposes `client` and `device`, but the generic `R::Device` path only gives the public device value; for CUDA that is just `Cuda(index)`.
- Decision: do not add a project-side literal compute capability key field in this branch. Doing it cleanly requires either an upstream/public CubeCL hardware property for CUDA CC or a project-specific CUDA-only dependency/query path with feature-boundary review. Current tuned keys should continue using the public hardware fingerprint until that broader API work is explicitly opened.
