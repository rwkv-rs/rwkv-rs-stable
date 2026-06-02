# Remote Current After GB10 Results

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-remote-current-after-gb10-results-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits and accumulated kernel tuning changes. This attempt owns only this attribution note unless a later entry explicitly opens a code prototype.
- User constraint: debug/tune first on remote `10.100.1.253`; do not use local GPU timing while the local GPU may be busy.
- Prior-note and memory search commands already run in this continuation:
  - `rg -n "kernel|WKV7|LayerNorm|ordered|10\\.100\\.1\\.253|GB10|residual|残差|autotune" /root/.codex/memories/MEMORY.md`
  - `rg -n "Decision:|Final decision|Next command|Next action|Next edit|Next implementation|open a fresh|open a separate|pending|unresolved|Remaining|未" .agents/notes/kernel-tuning/2026-05-16-*.md`
  - `rg -n "target-logit|target logit|key-prepare|warps_per_cube|warps-per-cube|row-pack|gated_readout|weight_decay_transform" .agents/skills/kernel-tuning/SKILL.md .agents/notes/kernel-tuning/2026-05-16-*.md`
- Matched prior evidence:
  - LayerNorm ordered-256/split-tail are closed by accuracy failures; GB10 `block_256` is allowed, local deterministic path remains `1024`.
  - WKV7 row-tile restrictions, shared-lanes, segment recompute, low-rank design, and output factorization are closed for this shape unless the algorithm boundary changes materially.
  - `lm_head_l2wrap_ce` direct target-logit is kept on GB10; online softmax, atomic loss, and prune-256 variants are closed.
  - `key_prepare` warps-per-cube autotune is rejected and reverted to fixed `HEAD64_WARPS_PER_CUBE = 4`.
  - GatedReadout row-pack is rejected and reverted; retained path is the 64-thread/two-warp combine.
  - `weight_decay_transform` is kept and should remain wired through `TrainBackend`.
  - Channel-mixer forced Cube/Burn-reference/fusion and LocalTuner bypass are recorded negatives or design-blocked. Projection-loss fusion needs a Cubek/TMA epilogue or GEMM-quality fused operator, which is broader than a small kernel tweak.
  - `ncu` on `10.100.1.253` is blocked by `RmProfilingAdminOnly: 1`; use standard compare, autotune logs, and `nsys` launch/register/shared-memory metadata.
- Machine/GPU: remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`; local GPU must not be used.
- Remote run directory: `/home/caizus/Projects/Packages/rwkv-rs-stable`, synchronized non-git mirror.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, heads `12`, head size `64`, vocab `65536`.
- Baseline path: `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Hypothesis: after the latest kept/reverted GB10 results, the correct next step is to refresh the remote current-source profile before selecting another implementation branch. Several old profiler artifacts predate kept target-logit and rejected/reverted launch changes.
- Candidate parameters: none. This is a read-only attribution attempt.
- Expected keep/revert boundary: keep this note as current-state evidence. If the top remaining project-owned surfaces are duplicate-closed or require broad Cubek/TMA operator work, do not edit kernels. If a non-duplicate, scoped boundary appears, open a fresh branch and note before editing.
- Next command: remote preflight on `10.100.1.253` to check GPU idleness, toolchain, current source snippets for kept/reverted decisions, and baseline availability.

## Remote Preflight

- Host: `spark-35ac`.
- Toolchain: `rustc 1.95.0`, `cargo 1.95.0`.
- GPU: `NVIDIA GB10`, compute capability `12.1`, utilization `0%`.
- Compute processes: only `/usr/libexec/gnome-remote-desktop-daemon` at `176 MiB`.
- Baseline: `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000` exists.
- Source boundary checks:
  - `lm_head_l2wrap_ce/kernel.rs` uses direct `target_logit` load and no `local_target` match in the inspected snippet.
  - `key_prepare/forward.rs` is restored to `HEAD64_WARPS_PER_CUBE = 4`.
  - `gated_readout_combine` has no `ROW_PACK` / `row_pack` matches.
  - `weight_prepare.rs` calls `weight_decay_transform(...)`, and `TrainBackend` imports `WeightDecayTransformBackend`.
- Decision: remote source matches the expected kept/reverted GB10 state. Proceed to compile/standard compare before profiling.
- Next command: remote `cargo check -p rwkv-test --features cuda`, then standard compare with `repeat=3,warmup=1` if compile passes.

## Remote Compile Check

- Command: `cargo check -p rwkv-test --features cuda` on `10.100.1.253`.
- Result: passed in dev check profile.
- Next command: standard remote compare with regenerated GB10 baseline, `repeat=3,warmup=1`, captured in `target/rwkv-test/remote-current-after-gb10-results-compare.log`.

## Remote Standard Compare

- Command: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1`, captured in `target/rwkv-test/remote-current-after-gb10-results-compare.log`.
- Binary/build provenance: remote release binary was already up to date after the kept/reverted GB10 source checks.
- Activation result: passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: positive with one known edge failure, `timing_summary compared=76 passed=75 failed=1 missing=0 extra=0 ignored=1 actual_total_ms=72.722 baseline_total_ms=175.887 speedup=2.42x`.
- Failed row: known short `timing/cells/cell_0000/channel_mixer.time.json`, `1.728ms` vs `1.668ms`, `0.97x`.
- Module timing signal:
  - `cells/*/time_mixer`: `42.360ms` vs `120.381ms`, `2.84x`.
  - `cells/*/channel_mixer`: `20.071ms` vs `25.759ms`, `1.28x`.
  - `loss/l2wrap_cross_entropy`: `4.272ms` vs `5.957ms`, `1.39x`.
- Decision: current remote state is activation-clean and total-speedup positive. Use this release binary for a short `nsys` ranking; profile-mode timing is attribution-only.
- Next command: short remote `nsys` profile with `repeat=1,warmup=1`, export sqlite, then rank CUDA kernel groups by demangled name and launch geometry.

## Remote Nsys Profile

- Command: `nsys profile --trace=cuda,nvtx --sample=none --force-overwrite=true --output=target/rwkv-test/nsys-current-after-gb10-results target/release/rwkv-test compare-rwkv-nn --color never --baseline ... --repeat 1 --warmup 1`.
- Result: exited nonzero because profile mode uses `repeat=1,warmup=1` against a `repeat=3,warmup=1` baseline. This is expected for attribution mode.
- Activation sanity inside profile run: passed, `activation_summary compared=54 passed=54 failed=0`.
- Profile-mode timing is invalid for acceptance due timing-profile mismatch, but rough total stayed positive: `actual_total_ms=76.755 baseline_total_ms=175.887 speedup=2.29x`.
- Generated artifact: `target/rwkv-test/nsys-current-after-gb10-results.nsys-rep`.
- Next command: export the nsys report to sqlite and query top CUDA kernel groups by demangled name plus launch geometry.

## Nsys Export Attempt 1

- Command: remote `nsys export ... && python3 <<'PY' ...`.
- Result validity: invalid shell quoting. The SQL string contained `'<unknown>'`, which broke the nested local/remote quoting and failed before export/query with `/bin/bash: line 1: unknown: No such file or directory`.
- Next command: rerun export/query using double-quoted SQL literals and a less fragile remote heredoc.

## Nsys Export And Query Attempts 2-3

- Export command: `nsys export --type sqlite --force-overwrite=true --output target/rwkv-test/nsys-current-after-gb10-results.sqlite target/rwkv-test/nsys-current-after-gb10-results.nsys-rep`.
- Export result: passed and produced `target/rwkv-test/nsys-current-after-gb10-results.sqlite`.
- Query attempt 2 result validity: invalid Python `-c` quoting; literal `\n` escapes caused `SyntaxError`.
- Next command: run the sqlite query via `ssh ... 'bash -s' <<'REMOTE'` heredoc to avoid nested quoting issues.

## Nsys Kernel Ranking

- Command: remote Python sqlite query grouped by demangled kernel name plus launch geometry.
- Top current groups:
  - lm-head projection Cubek/TMA matmul: `46.245ms / 3`, `grid=(4096,16,1)`, `block=(32,12,1)`, regs `73`, dynamic smem `27648`.
  - recurring Cubek/TMA matmul groups: `25.502ms / 144`, `20.874ms / 36`, `18.717ms / 36`, plus smaller matmul groups.
  - `wkv7_pretrain_forward_output_kernel_f_bf16`: `22.551ms / 36`, `grid=(12,16,1)`, `block=(64,1,1)`, regs `127`, dynamic smem `1536`.
  - `mix6_forward_kernel_f__n_1`: `14.713ms / 36`, `grid=(24576,1,1)`, `block=(32,8,1)`, regs `32`.
  - `channel_mixer_relu_square_forward_kernel_f__n_2`: `14.107ms / 36`, `grid=(49152,1,1)`, `block=(32,8,1)`, regs `16`.
  - `lm_head_l2wrap_ce_forward_row_kernel_f__i_i32`: `13.617ms / 3`, `grid=(8192,1,1)`, `block=(1024,1,1)`, regs `40`, dynamic smem `128`.
  - `gated_readout_combine_forward_kernel_f_`: `11.690ms / 36`, `grid=(8192,12,1)`, `block=(64,1,1)`, regs `25`, dynamic smem `8`.
  - `key_prepare_forward_64_kernel_f_`: `9.280ms / 36`, `grid=(24576,1,1)`, `block=(128,1,1)`, regs `24`.
  - `kernel_binop_c_bf16_n_8`: `8.579ms / 72`.
  - `value_residual_gate_forward_kernel_f__n_2`: `6.490ms / 33`.
  - `layer_norm_forward_kernel_f_`: `6.067ms / 78`, `block=(256,1,1)`.
  - `weight_decay_transform_forward_kernel_f__n_2`: `2.486ms / 36`.
- Interpretation:
  - The largest overall surface is still Cubek/TMA matmul, especially lm-head projection, but existing notes classify project-local projection/loss fusion as a broad Cubek/TMA epilogue or GEMM-quality operator problem.
  - The largest project-owned custom kernel is still WKV7, but current row-tile/shared-lanes/segment/factored output directions are closed for this boundary.
  - `mix6_forward` is now the largest remaining project-owned custom kernel without an obvious current-note closure beyond line-size/autotune-key work; inspect prior `mix6` notes/source before opening an implementation branch.
- Next command: read mix6-specific notes and source to determine whether a non-duplicate implementation boundary exists.

## Mix6 And Next Boundary Check

- Commands:
  - `rg -n "mix6.*line|line-size|axis|stacked|Mix6|mix6_forward" .agents/notes/kernel-tuning/2026-05-16-*.md .agents/skills/kernel-tuning/SKILL.md`.
  - Read `2026-05-16-mix6-forward-hardware-key.md`, `mix6/{forward.rs,kernel.rs,mod.rs}`, and TimeMixer call sites.
  - `rg -n "shared_value|value direct|direct value|WKV7.*value|wkv7.*shared.*value" .agents/notes/kernel-tuning crates/rwkv-nn/src/kernels/train/time_mixer/wkv7`.
- Mix6 result:
  - `2026-05-16-remote-current-profile-gb10.md` and related notes already show `mix6_forward` has `max_line_size=8` and GB10 selected `line_size_1` as the fastest candidate.
  - The value-residual-style vector-axis bug does not apply to Mix6; scale tensors are `[1,1,D]`, so embedded-axis vector eligibility was already valid.
  - Do not open another Mix6 line-size/axis branch from this evidence. A Mix6 improvement would need a different algorithm/output-layout boundary.
- New non-duplicate candidate:
  - Current WKV7 output kernel stores `value[lane]` into `shared_value[lane]`, then each row reads only `shared_value[row_index]`.
  - Unlike `decay`, `replacement_key`, `removal_key_normalized`, `replacement`, and `receptance`, `value` is not consumed across all columns by all rows. Each row can load its own scalar `value` directly from global memory once per timestep.
  - This changes only implementation traffic/shared-memory footprint; state update order and output accumulation order remain unchanged.
- Decision: close this attribution branch with a concrete next implementation attempt: open a fresh WKV7 branch that removes `shared_value` from `wkv7_pretrain_forward_output_kernel`, validates activation on `10.100.1.253`, and keeps only if standard compare and `nsys` WKV7 time improve versus the current `22.551ms / 36`, regs `127`, dynamic smem `1536` profile.
