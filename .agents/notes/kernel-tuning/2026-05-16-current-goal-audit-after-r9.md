# Current Goal Audit After R9 Projection Baseline

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-current-goal-audit-after-r9-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits plus retained/reverted kernel-tuning changes. This attempt owns only this audit note unless a later entry explicitly opens an implementation or tooling edit.
- User constraint: debug first on remote `10.100.1.253`; do not use the local GPU because the user is running other work there and timing may be inaccurate.
- Prior search commands:
  - `rg -n "lm_head/projection|projection matmul|matmul_specialized|matmul_entry|Cubek|TMA|lhs_size|rhs_size|projection baseline|R9|0\\.99x|15\\.24" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md /root/.codex/memories/MEMORY.md -S`
  - `rg -n "autotune key|LocalTuner|LayerNorm|deterministic|ncu|ERR_NVGPUCTRPERM|speedup|local|10\\.100\\.1\\.253|WKV7|shared-lanes|segment" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md /root/.codex/memories/MEMORY.md -S`
- Matched current evidence:
  - `2026-05-16-projection-baseline-statistics-gb10.md`: R9 projection baseline on GB10 gives activation `54/54`, timing `76/77`, total `1.48x`; only `lm_head/projection` is `0.99x`.
  - `.agents/skills/kernel-tuning/SKILL.md`: branch/note discipline, duplicate guards, LocalTuner mechanics, remote-first rule, LayerNorm numerical guardrails, and R9 projection measurement guard are recorded.
  - Remote `ncu` counters remain blocked by `ERR_NVGPUCTRPERM`; remote evidence can use `nsys` metadata and compare timing but not achieved occupancy / warp execution efficiency / memory throughput.
  - Local GPU timing is intentionally not current because the user said it is busy.
- Machine/GPU for any future runtime evidence: remote `caizus@10.100.1.253`, host `spark-35ac`, `NVIDIA GB10`, compute capability `12.1`, remote repo `/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Shape/dtype scope: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, heads `12`, head size `64`, vocab `65536`.
- Objective restated as deliverables:
  1. Each materially new kernel tuning attempt uses a fresh branch/worktree plus prewritten note; failed attempts are reverted but branches/notes retained.
  2. Kernel selection/autotune keys include runtime/backend, hardware fingerprint or GPU arch, dtype, `d_model`, rows, tunable parameters such as block size / warps / vector width, in-place or alias state, and deterministic numeric boundary.
  3. LayerNorm is not a hard-coded universal constant; GB10 `256` and local deterministic `1024` behavior are separated by hardware/shape/numeric policy.
  4. Burn/CubeCL `LocalTuner::execute` mechanics are understood: key/cache, persistent cache, warmup/profiling boundary, candidate parameters, and lack of production accuracy guard.
  5. CUDA lower-level design evidence explains warp/block, occupancy/register/shared-memory/coalescing, and BF16 numeric differences where profiler permissions allow.
  6. Poor results are investigated for algorithm vs implementation vs measurement cause before reverting.
  7. Final acceptance requires activation passing and speedup `>1.0` on both local and `10.100.1.253`.
- Commands planned for this audit:
  - Inspect current notes/skill and remote latest logs only. Do not run local GPU. Do not rerun remote benchmarks unless evidence is missing or stale.
- Expected decision boundary: if any deliverable remains incomplete, keep the goal open and choose the next non-duplicate concrete action. Do not mark the goal complete from remote-only success.

## 2026-05-16 Continuation After Compaction

- User constraint restated: continue debugging first on `10.100.1.253`; do not use the local GPU because it is reserved for other work and timing may be inaccurate.
- Duplicate check result:
  - `2026-05-16-value-residual-gate-vector-axis-gb10.md` already implemented and validated the `value_residual_gate` vector-axis eligibility fix. It changed dispatch from `n_1` to `n_2`, preserved activation, and kept remote total speedup above `1.0`, but did not prove a timing win.
  - `2026-05-16-value-residual-vector-width-gb10.md` is explicitly closed as a duplicate of that exact implementation boundary.
  - Therefore do not rerun or re-edit value-residual vector-width eligibility.
- Evidence gap: the current R9 projection baseline evidence in this note predates the latest audit continuation. Before choosing another kernel, confirm the remote current tree still passes activation and stays above `1.0x` against `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_r9_20260516`.
- Next command: inspect remote branch/status and existing R9 compare logs on `10.100.1.253`; run a fresh remote R9 compare only if the existing log is missing, stale, or clearly from a different source boundary.

## 2026-05-16 Remote R9 Evidence Refresh

- Remote repo check: `/home/caizus/Projects/Packages/rwkv-rs-stable` is a synchronized worktree without `.git`, so branch provenance must be tracked through the local branch/note plus remote file/log mtimes rather than remote git metadata.
- R9 baseline exists: `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_r9_20260516`.
- Existing R9 logs are present and complete:
  - `target/rwkv-test/remote-projection-r9-compare.log`
  - `target/rwkv-test/remote-projection-r9-rerun-compare.log`
- R9 result:
  - activation: `compared=54 passed=54 failed=0`
  - timing rerun: `compared=77 passed=76 failed=1 actual_total_ms=88.202 baseline_total_ms=130.484 speedup=1.48x`
  - only failed row: `timing/lm_head/projection.time.json`, `15.241ms` actual vs `15.158ms` baseline, `0.99x`.
- Source/binary timing check: remote `target/release/rwkv-test` mtime is `2026-05-16 04:52:19 -0500`, and R9 logs are `2026-05-16 05:34 -0500`, so the logs are newer than the binary. No fresh remote compare is needed for this audit.
- Decision: current remote GB10 acceptance is still valid under R9 baseline: activation passes and total speedup remains `>1.0`. Keep the goal open because local row-level timing remains unresolved and `ncu` counters remain permission-blocked.

## 2026-05-16 NCU Permission Follow-Up

- Current kernel implementation surface is mostly exhausted for small non-duplicate attempts:
  - WKV7 row-tile, shared-lanes, dense segment recompute, and low-rank segment-scan are closed.
  - Key-prepare warps-per-cube is closed and reverted to fixed 4 warps.
  - Mix6 line-size/axis retry is closed because GB10 tuner genuinely selected `line_size_1`.
  - LM-head row-kernel target-logit, atomic, online-softmax, and prune-256 variants are closed; projection+loss fusion is too broad without tensor-core matmul support.
  - Channel-mixer forced Cube, Burn reference, LocalTuner bypass, line-size retry, and matmul fusion are closed or out of scope.
- Correct next diagnostic step under the user's remote-only constraint: inspect why `ncu` counters are blocked on `10.100.1.253`, using non-destructive remote commands only. Do not run a local GPU workload.
- Next command: check remote `ncu`, `nvidia-smi`, kernel module profiling restriction parameter, device permissions, and passwordless sudo availability. Do not change system configuration without explicit user direction.

## 2026-05-16 NCU Permission Result

- Host: `spark-35ac`, user `caizus`.
- `ncu`: `/usr/local/cuda-13.0/bin/ncu`, Nsight Compute `2025.3.1.0`.
- GPU/driver: `NVIDIA GB10`, compute capability `12.1`, driver `580.95.05`.
- Device nodes: `/dev/nvidia0` and `/dev/nvidiactl` are world-readable/writable, so the block is not Unix device-file permissions.
- Driver setting: `/proc/driver/nvidia/params` reports `RmProfilingAdminOnly: 1`.
- Sudo check: `sudo -n true` exits `1`, so this session cannot make privileged driver changes non-interactively.
- Interpretation: remote `ERR_NVGPUCTRPERM` is explained by NVIDIA performance counters being restricted to admin users. Until root changes the driver setting or runs `ncu` under a privileged context, achieved occupancy, warp execution efficiency, SOL, and memory-throughput counters are unavailable on `10.100.1.253`.
- Decision: do not rerun `ncu` in the same permission boundary. Continue remote-only work with standard compare and `nsys` metadata, and treat counter-level validation as blocked by environment rather than by the kernel code.
- Next action if root access is provided later: enable non-admin profiling counters or run the targeted `ncu` commands under sudo/root, then collect SpeedOfLight, Occupancy, SchedulerStats, WarpStateStats, and MemoryWorkloadAnalysis for the remaining large project-owned kernels.

## 2026-05-16 Skill Update

- Scope: update `.agents/skills/kernel-tuning/SKILL.md` only.
- Reason: the remote `ncu` permission blocker is stable environment guidance for future tuning attempts on `10.100.1.253`; recording it prevents repeated profiler attempts in the same `RmProfilingAdminOnly: 1` boundary.
- Next edit: add a concise known GB10 profiling blocker entry to the skill.
- Result: added the known GB10 profiling blocker entry to `.agents/skills/kernel-tuning/SKILL.md`, including `ncu` path, `RmProfilingAdminOnly: 1`, no passwordless sudo, and the rule to avoid repeated `ncu` counter runs until privileged access changes.
- Next command: sync the updated skill and this note to the remote mirror.

## 2026-05-16 Prompt-To-Artifact Checklist

| Objective requirement | Current evidence | Status |
| --- | --- | --- |
| Use git branch/tree per materially new tuning attempt; failed attempts reverted but branches/notes kept. | Current branch is `kernel-tuning-current-goal-audit-after-r9-20260516`; prior notes list dedicated tuning branches and keep/revert decisions. Current worktree is dirty with broad unrelated changes, so this audit owns only this note and the skill blocker update. | Covered for current continuation; keep enforcing before any new kernel/profiler run. |
| Autotune/runtime keys include backend/runtime. | Source inspection shows train-kernel autotune keys use `runtime: R::name(...)` where candidates exist, including LayerNorm, lm-head, Mix6, and gate kernels. | Covered for inspected tuned kernels. |
| Include GPU hardware / compute capability or equivalent fingerprint. | Current keys include CubeCL hardware fingerprint fields: `load_width`, `plane_size`, `max_units_per_cube`, `max_cube_dim`, shared memory, vector size, SM count, tensor-core fields. | Covered through CubeCL-exposed hardware fingerprint; exact CUDA compute capability is not exposed directly in these key structs. |
| Include dtype. | Inspected keys include `dtype: DType`. | Covered. |
| Include `d_model` and rows `B*T` where relevant. | LayerNorm key has `d_model` and `rows`; Mix6 and backward keys have `embedded_dim`/`d_model` and rows; lm-head uses `num_tokens` and `vocab_size` because its reduction axis is vocab. | Covered by shape-specific fields. |
| Include tunable parameters: block size, warps, vector width. | Candidate names/groups encode `block_*`, `line_size_*`, row tiles, and rejected warps attempts are documented. | Covered where there is a live candidate set; fixed-shape kernels without live candidate sets are documented as not autotune-key gaps until a second implementation exists. |
| Include in-place / alias and deterministic numeric boundary. | Tuned keys include `is_in_place` and `deterministic`; LayerNorm also includes `deterministic_min_block_size`. | Covered for inspected tuned kernels. |
| LayerNorm should select by hardware/shape/policy, not one constant. | `layer_norm/forward.rs` has `LayerNormForwardAutotuneKey`, `BLOCK_SIZE_CANDIDATES`, `LocalTuner`, and `supports_gb10_bf16_d768_layer_norm(...)` that permits `256` only for the GB10 BF16 D768 rows=8192 fingerprint; fallback uses `d_model.next_power_of_two()` up to device max. | Covered. |
| Burn/CubeCL LocalTuner mechanism: key/cache/warmup/measurement/accuracy guard understood. | `.agents/skills/kernel-tuning/SKILL.md` records cache-hit vs miss, `LocalTuner::init`, persistent cache checksum, warmup/profiling cost, and that `autotune-checks` is not a production trace accuracy guard. | Covered in project skill. |
| CUDA design evidence: warp/block, register pressure, shared memory, coalescing, BF16 differences. | Notes contain `nsys` launch metadata for current remote kernels and negative notes analyzing WKV7, key_prepare, lm-head, LayerNorm drift, and value-residual vector-axis. Remote `ncu` counter collection is blocked by `RmProfilingAdminOnly: 1`; local ncu evidence exists from earlier but local GPU is currently reserved by user. | Partially covered; counter-level remote evidence blocked by environment. |
| Poor results investigated before revert. | Notes document why WKV7 segment designs, key_prepare warps, lm-head row variants, channel-mixer variants, value-residual vector-axis, and ordered-256 LayerNorm were kept/reverted. | Covered for recorded attempts. |
| If precision issue occurs, inspect whether code/math caused it. | Ordered-256 LayerNorm notes record CPU ordered math matching `1024` but device candidate still drifting on `value_from_first_cell` and `lm_head/embedded_context`, classifying it as a device implementation/boundary failure rather than blindly reverting. | Covered for the known precision issue. |
| Final acceptance: speedup `>1.0` on both local and `10.100.1.253`. | Remote R9 baseline result is activation `54/54` and total speedup `1.48x`; local regenerated-baseline result is activation `54/54` and total speedup `2.44x`, but local row-level timing still has failures and the older checked-in baseline gives only `0.92x`. | Partially covered; total speedup evidence exists on both machines, but clean row-level/local baseline-provenance acceptance remains incomplete. |

- Audit conclusion: the goal is not complete. Remote GB10 total acceptance is current and valid, and local regenerated-baseline total speedup is above `1.0`, but local row-level timing/baseline provenance remains incomplete and remote ncu counter evidence is blocked by driver permissions. Do not call `update_goal`.
- Next concrete action without running another broad benchmark: keep this audit as the current state record and use prior local regenerated-baseline evidence to target the remaining local channel-mixer/short-row provenance gap only under a fresh branch. Any new kernel idea must start with a fresh branch and prewritten note; current small non-duplicate surfaces are mostly exhausted.

## 2026-05-16 Privileged NCU Plan

- Added `.agents/notes/kernel-tuning/2026-05-16-privileged-ncu-command-plan-gb10.md`.
- Purpose: preserve exact privileged `ncu` commands and analysis targets for `10.100.1.253` once root enables non-admin performance counters or runs `ncu` under a privileged context.
- The plan targets current top surfaces from `nsys`: lm-head projection Cubek/TMA matmul, WKV7 output, Mix6, channel-mixer ReLU-square, lm-head row loss, gated-readout combine, key-prepare, value-residual gate, and LayerNorm.
- The plan explicitly says not to execute as normal `caizus` while `RmProfilingAdminOnly: 1` remains set.
- Remote sync result: note copied to `/home/caizus/Projects/Packages/rwkv-rs-stable/.agents/notes/kernel-tuning/2026-05-16-privileged-ncu-command-plan-gb10.md`.

## 2026-05-16 Local GPU Availability Check

- Purpose: the active goal still requires local speedup `>1.0`, but the user said the local GPU is being used for other work and timing may be inaccurate. Check only whether the GPU is busy; do not run any benchmark, profiler, or CUDA workload locally.
- Next command: run `nvidia-smi` query for GPU name, memory use, utilization, and compute processes.
- Result: local `NVIDIA GeForce RTX 5090`, utilization `4%`, memory `3088/32607 MiB`, and no compute apps reported. No benchmark or profiler was run in this audit step.
- Additional evidence correction: prior note `2026-05-16-local-regenerated-baseline.md` already contains a local regenerated-baseline compare:
  - activation `54/54`
  - total timing `actual_total_ms=39.984 baseline_total_ms=97.609 speedup=2.44x`
  - row-level timing still failed (`22/76` pass), with meaningful remaining local gaps around channel mixer and short row boundaries.
- Audit correction: local total speedup `>1.0` has evidence under the locally regenerated baseline, but local clean row-level acceptance is not achieved and should not be conflated with the older checked-in baseline result (`0.92x`).
