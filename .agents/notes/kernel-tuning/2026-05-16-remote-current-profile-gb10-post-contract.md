# Remote Current Profile GB10 Post Contract

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-remote-current-profile-gb10-post-contract-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits plus retained/reverted kernel tuning changes. This attempt owns only this attribution note and remote profiler evidence unless a later entry explicitly opens a separate implementation branch.
- Prior-note search commands:
  - `rg -n "key_prepare|warps_per_cube|row_tile|WKV7|gated_readout|row-pack|rowpack|LayerNorm|ordered-256|channel_mixer|lm_head|LocalTuner|remote|10\\.100\\.1\\.253" .agents/notes/kernel-tuning crates/rwkv-nn/src/kernels/train -S`
  - `rg -n "completion|current-goal|autotune key|LocalTuner|ncu|ERR_NVGPUCTRPERM|speedup=|activation_summary|timing_summary|LayerNorm|ordered-256|10\\.100\\.1\\.253|local" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md /root/.codex/memories/MEMORY.md -S`
- Matched prior evidence:
  - Current remote `10.100.1.253` / GB10 is the primary timing source while the local GPU is reserved by the user.
  - `key_prepare` warps-per-cube tuning is a recorded GB10 negative and was reverted to fixed `HEAD64_WARPS_PER_CUBE=4`.
  - WKV7 row-tile-only tuning is closed; forced `row_tile=16` regressed, and a real time-split/state-handoff/scan design is a separate algorithmic branch.
  - Closed duplicate directions include residual-add/Burn-add A/B, channel-mixer forced Cube/Burn-reference/fusion/line-size, LocalTuner bypass, lm-head target-logit/atomic/online-softmax/prune-256, GatedReadout warp32/row-pack, LayerNorm ordered-256/split-tail, and key-prepare warps.
  - `lm_head/projection` timing contract was just made optional-comparable, but it does not change GPU kernels. A current nsys profile is needed before choosing another implementation boundary.
- Machine/GPU: remote `caizus@10.100.1.253`, expected `NVIDIA GB10`, compute capability `12.1`.
- Remote run directory: `/home/caizus/Projects/Packages/rwkv-rs-stable`, synchronized non-git mirror.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, heads `12`, head size `64`.
- Baseline path: `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Hypothesis: after the `lm_head/projection` timing-contract change and prior reverted attempts, the correct next action is to refresh remote current-source attribution. Do not choose another kernel implementation from stale profiles.
- Candidate parameters: none. This is a profiling/attribution attempt only.
- Expected keep/revert boundary: keep this note as evidence. If the top current surfaces are already duplicate-closed or Cubek matmul internals, do not edit. If the profile reveals a new project-owned non-duplicate fusion or dispatch boundary, open a fresh branch/note before editing.
- Next command: remote preflight and scoped source sync, then run a standard compare and short nsys profile on `10.100.1.253`. Do not run local GPU commands.

## Remote Preflight

- Command: remote preflight on `10.100.1.253` checking host, Rust/Cargo, GPU, baseline path, and Nsight Systems.
- Result: host `spark-35ac`; Rust/Cargo `1.95.0`; GPU `NVIDIA GB10`, compute capability `12.1`, utilization `0 %`; baseline directory present; Nsight Systems `2025.3.2`.
- Next command: sync the current local mirror to `/home/caizus/Projects/Packages/rwkv-rs-stable`, excluding `.git`, `target`, `weights`, and `results`, so the remote profile includes the latest timing-contract source and notes.

## Remote Sync

- Command: `rsync -az --delete --exclude '.git/' --exclude 'target/' --exclude 'weights/' --exclude 'results/'` to `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable/`.
- Result: completed successfully.
- Remote source provenance: mirror of local branch `kernel-tuning-remote-current-profile-gb10-post-contract-20260516` with the dirty-tree constraint recorded above.
- Next command: remote compile check `cargo check -p rwkv-test --features cuda` before profiling.

## Remote Compile Check

- Command: `cargo check -p rwkv-test --features cuda` on `10.100.1.253`.
- Result: passed in dev check profile.
- Next command: remote standard compare with regenerated baseline, `repeat=3`, `warmup=1`, to build the release binary and confirm current source is still activation-clean before `nsys`.

## Remote Standard Compare

- Command: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1`, output captured in `target/rwkv-test/remote-current-profile-post-contract-compare.log`.
- Correctness result: activation passed, `compared=54 passed=54 failed=0`.
- Timing result: `compared=76 passed=75 failed=1 missing=0 extra=0 ignored=1 actual_total_ms=74.519 baseline_total_ms=175.887 speedup=2.36x`.
- Single timing fail: `cells/cell_0000/channel_mixer.time.json`, actual `1.669 ms` vs baseline `1.668 ms`, effectively a boundary/noise edge outside this attribution branch.
- Decision: current remote source remains accuracy-valid and total speedup-positive. Use the built release binary for attribution, not as a clean all-row acceptance proof.
- Next command: remote short `nsys` profile of `target/release/rwkv-test compare-rwkv-nn --repeat 1 --warmup 1`, output `target/rwkv-test/nsys-current-post-contract-gb10`.

## Remote Nsys Profile

- Command: `nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --output target/rwkv-test/nsys-current-post-contract-gb10 target/release/rwkv-test compare-rwkv-nn --color never --baseline ... --repeat 1 --warmup 1`.
- Result: generated `target/rwkv-test/nsys-current-post-contract-gb10.nsys-rep`.
- Correctness inside profile run: activation passed `54/54`.
- Timing inside profile run: invalid for acceptance because actual profile uses `repeat=1,warmup=1` while baseline timing uses `repeat=3,warmup=1`; all row timing comparisons are expected profile-mismatch failures.
- Next command: export the nsys report to sqlite and summarize CUDA kernel groups by demangled name plus launch geometry.

## Nsys Export And Summary Attempt 1

- Command: remote `nsys export --type sqlite` followed by an inline Python sqlite summary.
- Export result: sqlite export succeeded at `target/rwkv-test/nsys-current-post-contract-gb10.sqlite`.
- Summary result: invalid. The inline Python heredoc was parsed incorrectly by the nested local/remote shell quoting, causing a Python `SyntaxError` before any kernel rows were printed.
- Decision: keep the export. Rerun only the sqlite summary by piping a locally quoted Python body into `ssh ... 'cd ... && python3 -'`.

## Nsys Summary Attempt 2

- Command: piped Python sqlite query over `target/rwkv-test/nsys-current-post-contract-gb10.sqlite`.
- Result: query succeeded.
- Current top CUDA groups:
  - Cubek BF16 matmul, `47.585 ms / 3`, `grid=(4096,16,1)`, `block=(32,12,1)`, regs `73`, dynamic smem `27648`.
  - Cubek BF16 matmul, `25.319 ms / 144`.
  - `wkv7_pretrain_forward_output_kernel_f_bf16`, `22.034 ms / 36`, `grid=(12,16,1)`, `block=(64,1,1)`, regs `108`, dynamic smem `1536`.
  - Cubek BF16 matmul groups, `20.833 ms / 36` and `18.856 ms / 36`.
  - `channel_mixer_relu_square_forward_kernel_f__n_2`, `14.163 ms / 36`.
  - `mix6_forward_kernel_f__n_1`, `14.090 ms / 36`.
  - `lm_head_l2wrap_ce_forward_row_kernel_f__i_i32`, `13.184 ms / 3`.
  - `gated_readout_combine_forward_kernel_f_`, `11.682 ms / 36`.
  - `key_prepare_forward_64_kernel_f_`, `9.489 ms / 36`.
  - `kernel_binop_c_bf16_n_8`, `8.686 ms / 72`.
  - `layer_norm_forward_kernel_f_`, `6.573 ms / 78`.
  - `value_residual_gate_forward_kernel_f__n_2`, `6.432 ms / 33`.
- Interpretation: the largest project-owned surfaces are still mostly duplicate-guarded or algorithmic: WKV7 needs state-handoff/scan rather than row-tile retry; channel mixer/mix6/lm-head/gated-readout/key-prepare all have current negative or kept boundaries; LayerNorm is already hardware/shape guarded. The remaining actionable evidence gap is `kernel_binop_c_bf16_n_8` / `value_residual_gate_forward_kernel_f__n_2` attribution and why value-residual still selects `n_2`.
- Next command: inspect current value-residual notes/source and remote autotune cache entries before deciding whether this is a duplicate or a materially different branch.

## Value Residual Duplicate Check

- Commands: read `2026-05-16-value-residual-gate-vector-axis-gb10.md`, `2026-05-16-value-residual-vector-width-gb10.md`, current `value_residual_gate/forward.rs`, and remote autotune cache files for value/learning-rate gate forward.
- Result:
  - The current source already contains `max_line_size_value_residual(...)`, separating `[value, value_from_first_cell, gate_input]` embedded-axis vector width from 1D `gate_base` axis `0`.
  - The prior duplicate note already closed the same vector-width idea without rerun.
  - Remote value-residual autotune cache has `max_line_size=8` and selects `line_size_2`; candidates are nearly tied: `line_size_2` median `220.868us`, `line_size_4` `221.572us`, `line_size_1` `221.604us`, `line_size_8` `222.836us`.
  - Remote learning-rate gate cache selects `line_size_8`, showing the candidate system can choose a wider vector when the benchmark supports it.
- Decision: do not open another value-residual vector-width branch. The current `n_2` result is a measured tuner choice, not an eligibility bug.
- Next command: query resolved ordered kernel windows around `kernel_binop_c_bf16_n_8` in the current post-contract sqlite to map remaining generic launches before deciding the next branch.

## Generic Binop Window Query

- Command: resolved ordered-window sqlite query around `kernel_binop_c_bf16_n_8` in `nsys-current-post-contract-gb10.sqlite`.
- Result: `72` centers collapsed into three repeated window signatures.
- Main repeated signature, `36` times: `gated_readout_combine_forward_kernel_f_ -> matmul -> kernel_binop_c_bf16_n_8 -> layer_norm_forward_kernel_f_ -> channel_mixer_mix_forward_kernel_f__n_8`.
- Second repeated signature, `33` times: `channel_mixer_relu_square_forward_kernel_f__n_2 -> matmul -> kernel_binop_c_bf16_n_8 -> layer_norm_forward_kernel_f_ -> mix6_forward_kernel_f__n_1`.
- Tail signature, `3` times: final channel-mixer/projection tail into lm-head.
- Per-launch details: each `kernel_binop_c_bf16_n_8` is about `109-117us`, `grid=(3072,1,1)`, `block=(32,8,1)`, `regs=16`, no shared memory.
- Interpretation: these generic BF16 binops map to residual/tensor-add boundaries between projection outputs and the next LayerNorm/module, not to a new isolated transform. The plain residual-add direction is duplicate-guarded by the recorded residual notes, and a broader residual fusion would have to cross matmul/projection boundaries rather than replace only this binop.
- Decision: close this profile branch as attribution evidence. The remaining large small-scope candidates are duplicate-closed; the next materially different implementation track is WKV7 algorithm design for time/chunk state handoff or scan-style composition, because row-tile-only tuning is already closed and the current WKV7 kernel is the largest project-owned non-matmul surface.
