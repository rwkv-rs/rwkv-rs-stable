# Remote Next Surface GB10

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-remote-next-surface-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this branch inherits broad unrelated workspace changes and the kept 64-thread GatedReadout combine / TimeMixer gate wiring state. This attempt is evidence-first and may run remote profiler/compare commands, but it must not edit kernels until a non-duplicate implementation boundary is identified and recorded.
- Prior-note search command:
  - `rg -n "Next command|next useful|next .*branch|Decision:|Final decision|unresolved|candidate" .agents/notes/kernel-tuning/2026-05-16-*.md`
  - `rg -n "10\\.100\\.1\\.253|remote|GB10|channel_mixer|LayerNorm|ordered-256|residual|gated_readout|WKV7|lm_head|LocalTuner|autotune" .agents/notes/kernel-tuning /root/.codex/memories/MEMORY.md`
- Matched prior evidence:
  - `2026-05-16-gated-readout-warp32-gb10.md`: warp32 GatedReadout combine passed activation but made the targeted combine kernel slower (`12.167ms` vs `11.793ms`) and was reverted. Do not retry warp32.
  - `2026-05-16-gated-readout-forward-combine-gb10.md`: retained 64-thread GatedReadout combine reduces generic binop/reduce surface and reaches `2.05x-2.14x` clean-cache total, but standard compares can still show the known unrelated `cell_0000/channel_mixer` edge.
  - `2026-05-16-wire-time-mixer-gates-gb10.md`: TimeMixer learning-rate/value-residual gate wiring is a kept positive change that removed the large generic f32 elementwise surface.
  - `2026-05-16-layernorm-ordered256-accuracy-rootcause.md`: ordered-256 math itself matches the `1024` order on trace-backed CPU diagnostics; the failed post-gates ordered-256 candidate is a device implementation/boundary problem. Do not reintroduce it without a device-side or trace-backed guard.
  - Duplicate guards: residual-add/Burn-add A/B, channel-mixer forced Cube/Burn-reference/fusion, LocalTuner bypass, WKV7 forced `row_tile=16`, lm-head target-logit/atomic/online-softmax/prune-256, and GatedReadout warp32 are already closed.
- Machine/GPU: remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`. The local GPU must not be used because the user is using it for other work.
- Remote run directory: `/home/caizus/Projects/Packages/rwkv-rs-stable`, synchronized non-git copy. Branch provenance remains local.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, regenerated remote baseline under `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Hypothesis: after the kept gate wiring and 64-thread GatedReadout combine, the next correct step is to identify the current remote live copy's largest remaining non-duplicate surface from `nsys`/compare evidence. `ncu` counters are known blocked by `ERR_NVGPUCTRPERM`, so use `nsys` launch geometry/time plus source inspection unless permissions change.
- Candidate parameters: none yet. This branch first validates remote source/profiler provenance and ranks current kernel groups.
- Expected keep/revert boundary: keep this note as attribution evidence. If the top surfaces are Cubek TMA matmul internals or already rejected project kernels, do not edit them. If a project-owned unfused generic surface remains, open a fresh implementation branch before editing.
- Next command: remote preflight on `10.100.1.253` to confirm source snippets, available `nsys` artifacts, autotune cache state, and whether a current standard compare/profile must be rerun.

## Remote Preflight

- Host/GPU result: `spark-35ac`, `NVIDIA GB10`, compute capability `12.1`, driver `580.95.05`, GPU utilization `0%` at preflight.
- Remote directory result: `/home/caizus/Projects/Packages/rwkv-rs-stable` exists and is still a non-git synchronized run copy.
- Available profiler artifacts:
  - `target/rwkv-test/nsys-gated-readout-forward-combine-clean.sqlite`
  - `target/rwkv-test/nsys-gated-readout-forward-combine.sqlite`
  - `target/rwkv-test/nsys-gated-readout-warp32.sqlite`
  - `target/rwkv-test/nsys-lm-head-online-gb10.sqlite`
  - `target/rwkv-test/nsys-remote-time-mixer.sqlite`
  - `target/rwkv-test/nsys-time-mixer-gates-gb10.sqlite`
  - `target/rwkv-test/nsys-value-residual-gate-vector-axis.sqlite`
- Remote live source result:
  - `gated_readout_combine` is back on the retained 64-thread/two-warp implementation: `BLOCK_SIZE=64`, `NUM_WARPS=2`, shared-memory inter-warp reduction.
  - `WeightPrepare::forward` uses the kept `learning_rate_gate` and `value_residual_gate` fused paths.
  - `GatedReadout::forward` uses the retained `gated_readout_combine(...)` before output projection.
- Autotune cache preflight: `target/autotune` contains 10 files. Do not clear it in this attribution branch unless a note first records a stale/polluted-cache hypothesis.
- Next command: inspect remote profiler artifact mtimes and query the newest live-state SQLite for current top CUDA kernel groups, launch configs, and remaining generic elementwise surfaces.

## Profiler Artifact Query Attempt 1

- Command: remote `ls -lt target/rwkv-test/*.sqlite` plus Python `sqlite3` top-kernel query.
- Artifact mtime ordering:
  - newest: `nsys-value-residual-gate-vector-axis.sqlite` (`2026-05-15 23:53`)
  - then `nsys-gated-readout-warp32.sqlite` (`23:26`, rejected code)
  - then `nsys-gated-readout-forward-combine-clean.sqlite` (`23:18`, retained 64-thread combine)
- Result validity: invalid query. The Python schema query lost SQL string quoting across the remote zsh/heredoc boundary and failed with `sqlite3.OperationalError: near "table": syntax error` before printing kernel rows.
- Next command: rerun the SQLite query without the fragile schema query, using direct `try/except` around the kernel summary query.

## Profiler Artifact Query Attempt 2

- Command: direct Python `sqlite3` summary over all remote `target/rwkv-test/*.sqlite`.
- Result validity: invalid query. All artifacts failed with `OperationalError('no such column: k.name')`; this Nsight schema uses string-id columns such as `shortName`/`mangledName`/`demangledName`, not a `name` column.
- Next command: inspect `PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)` and `PRAGMA table_info(StringIds)` for one artifact, then rerun with the actual schema.

## Nsight SQLite Schema

- Artifact inspected: `target/rwkv-test/nsys-gated-readout-forward-combine-clean.sqlite`.
- Kernel table name columns: `demangledName`, `shortName`, and `mangledName` are integer IDs into `StringIds`; there is no `name` text column.
- Useful launch fields available: `registersPerThread`, `gridX/Y/Z`, `blockX/Y/Z`, `staticSharedMemory`, `dynamicSharedMemory`, and local memory fields.
- Next command: rerun kernel ranking by joining `StringIds` against `demangledName` and grouping by launch geometry.

## Remote Kernel Ranking

- Command: remote Python `sqlite3` query over all available `target/rwkv-test/*.sqlite`, grouped by demangled kernel name plus launch geometry.
- Newest live-state artifact: `nsys-value-residual-gate-vector-axis.sqlite`. It reflects the retained 64-thread GatedReadout combine plus the later value-residual-gate vector-axis dispatch correction.
- Top relevant groups in that newest artifact:
  - Cubek BF16 TMA lm-head projection matmul: `46.706ms / 3`, `grid=(4096,16,1)`, `block=(32,12,1)`, `regs=73`, `dynamic_smem=27648`.
  - Cubek BF16 TMA recurring matmuls: `25.175ms / 144`, `21.250ms / 36`, `19.298ms / 36`; these are Burn/Cubek matmul internals and not a scoped project-kernel target.
  - `wkv7_pretrain_forward_output_kernel_f_bf16`: `21.895ms / 36`, `regs=108`, `dynamic_smem=1536`; WKV7 row-tile-only tuning is duplicate-guarded negative and larger algorithm changes need a separate design.
  - `channel_mixer_relu_square_forward_kernel_f__n_2`: `14.378ms / 36`; channel-mixer direct candidates are duplicate-guarded.
  - `mix6_forward_kernel_f__n_1`: `13.871ms / 35`; previous local ncu said this is memory-heavy, and existing key-design work covers line-size dispatch.
  - `lm_head_l2wrap_ce_forward_row_kernel_f__i_i32`: `13.156ms / 3`; lm-head row variants are duplicate-guarded negative.
  - `gated_readout_combine_forward_kernel_f_`: `11.940ms / 36`, retained 64-thread implementation; warp32 was tested and rejected.
  - `key_prepare_forward_64_kernel_f_`: `9.444ms / 36`, `grid=(24576,1,1)`, `block=(128,1,1)`, `regs=24`, no shared memory.
  - `layer_norm_forward_kernel_f_`: `6.140ms / 76`, `block=(256,1,1)`, remote GB10 policy is already the fast/safe `256` path.
- Source inspection for key prepare:
  - `crates/rwkv-nn/src/kernels/train/time_mixer/key_prepare/forward.rs` hard-codes `HEAD64_WARPS_PER_CUBE = 4`.
  - Launch uses `CubeDim::new_1d(HEAD64_WARPS_PER_CUBE * 32)`; the current GB10 launch is therefore `block=(128,1,1)`.
  - `key_prepare_forward_64_kernel` maps one warp to one `[batch,time,head]`, each lane computes two BF16 values, does one warp reduction, and writes three outputs.
- Decision: close this attribution branch as evidence and open a fresh implementation branch for key-prepare `warps_per_cube` runtime/autotune candidates. This is not the previously rejected key-prepare line-size microtuning; it changes launch packing (`num_warps`/block size) for a real remaining GB10 surface and directly matches the required hardware/shape dispatch dimension.
