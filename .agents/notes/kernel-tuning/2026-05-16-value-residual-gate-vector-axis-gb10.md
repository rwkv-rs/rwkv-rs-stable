# Value Residual Gate Vector Axis GB10

- Date: 2026-05-16
- Branch/worktree: `kernel-tuning-value-residual-gate-vector-axis-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this branch inherits broad unrelated workspace changes and prior kernel tuning edits. This attempt owns only `crates/rwkv-nn/src/kernels/train/time_mixer/value_residual_gate/forward.rs`, this note, and remote validation artifacts for this branch.
- Prior-note search command:
  - `rg -n "time_mixer|post-gates|nsys|mix6|key_prepare|value_residual|gated_readout|wkv7|learning_rate" .agents/notes/kernel-tuning | head -200`
  - `rg -n "value_residual_gate|ValueResidualGate|line_size|max_line_size|axis|n_1|vector width" .agents/notes/kernel-tuning crates/rwkv-nn/src/kernels/train/time_mixer/value_residual_gate -S`
- Matched prior evidence:
  - `2026-05-16-remote-time-mixer-post-gates-nsys.md` found `value_residual_gate_forward_kernel_f__n_1` at `6.620ms / 33` in the clean remote `nsys` profile and identified that its remote autotune key had `max_line_size=1`.
  - `2026-05-16-forward-elementwise-full-hardware-key.md` and `2026-05-16-autotune-key-audit.md` already expanded the forward elementwise autotune keys, so this branch must not duplicate key-field work.
  - No prior note records fixing the `gate_base` axis in `value_residual_gate` forward vector-width eligibility.
- Changed boundary: this is not a forced vector width or a LocalTuner bypass. It fixes candidate eligibility so the existing `LocalTuner` can consider `line_size_2/4/8` when large tensors and the 1D `gate_base` are all contiguous along their own embedded axes.
- Machine/GPU: primary validation on remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`; local GPU must not be used.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, regenerated remote baseline under `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Kernel/stage: `time_mixer/value_residual_gate` forward, nonzero cells only (`33` launches in current `nsys` profile).
- Implementation plan: change only `max_line_size` calculation in `value_residual_gate/forward.rs` so `value`, `value_from_first_cell`, and `gate_input` use the 3D embedded axis while `gate_base` uses axis `0`; keep the existing autotune key/candidates and kernel math unchanged.
- Expected keep/revert boundary: keep only if local non-GPU compile/format checks pass, remote CUDA check passes, remote activation remains `54/54 PASS`, and standard remote compare stays speedup `>1.0` without introducing broad timing failures. If LocalTuner still selects `line_size_1`, that is acceptable evidence; if trace correctness fails or total timing regresses materially, revert the code and keep the branch/note as negative evidence.
- Next command: edit `crates/rwkv-nn/src/kernels/train/time_mixer/value_residual_gate/forward.rs`, then run rustfmt and non-GPU compile checks.

## 2026-05-16 Code Change

- Changed `value_residual_gate/forward.rs` only.
- Replaced the old `max_line_size_many(&[value, value_from_first_cell, gate_base, gate_input], shape.num_dims() - 1)` calculation with `max_line_size_value_residual(...)`.
- New eligibility logic:
  - `value`, `value_from_first_cell`, and `gate_input` use the 3D embedded axis from the value tensor.
  - `gate_base` uses axis `0`, its actual `[embedded_dim]` axis.
  - The final `max_line_size` is the min of those two compatible vector widths.
- Kernel math and candidate set are unchanged.
- Next command: `rustup run nightly rustfmt crates/rwkv-nn/src/kernels/train/time_mixer/value_residual_gate/forward.rs`, then `cargo check -p rwkv-nn --features cuda,fusion`.

## 2026-05-16 Local Non-GPU Checks

- Formatting command: `rustup run nightly rustfmt crates/rwkv-nn/src/kernels/train/time_mixer/value_residual_gate/forward.rs`.
- Formatting result: passed.
- Compile command: `cargo check -p rwkv-nn --features cuda,fusion`.
- Compile result: passed.
- Next command: sync only `crates/rwkv-nn/src/kernels/train/time_mixer/value_residual_gate/forward.rs` and this note to `/home/caizus/Projects/Packages/rwkv-rs-stable` on `10.100.1.253`, clear only the remote value-residual-gate forward autotune log so the fixed candidate eligibility retunes, then run remote `cargo check -p rwkv-test --features cuda`.

## 2026-05-16 Remote Sync And Compile

- Sync command: `rsync -avR -e 'ssh -i ~/.ssh/id_ed25519_dgx_spark_windows ...' crates/rwkv-nn/src/kernels/train/time_mixer/value_residual_gate/forward.rs .agents/notes/kernel-tuning/2026-05-16-value-residual-gate-vector-axis-gb10.md caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable/`.
- Sync result: completed; only the owned Rust file and this note were transferred.
- Remote cache cleanup: removed `target/autotune/0.10.0/device-0-0-cuda/rwkv_nn-kernels-train-time_mixer-value_residual_gate-forward-value-residual-gate-forward.json.log`.
- Remote compile command: `cargo check -p rwkv-test --features cuda`.
- Remote compile result: passed.
- Next command: standard remote compare with regenerated remote baseline: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1`.

## 2026-05-16 Invalid Compare Wrapper

- Invalid command wrapper: `...; status=$?; tail -120 "$LOG"; exit $status` under the remote user's `zsh`.
- Failure: `zsh:1: read-only variable: status`.
- Validity: the shell wrapper is invalid. The compare may have completed and written `target/rwkv-test/remote-value-residual-gate-vector-axis-compare.log`, but no conclusion is allowed until that log is inspected and, if needed, rerun with a non-reserved variable name.
- Next command: inspect the remote log mtime/tail and the value-residual-gate autotune log; rerun compare only if the log is incomplete.

## 2026-05-16 Remote Compare Result

- Log inspection result: `target/rwkv-test/remote-value-residual-gate-vector-axis-compare.log` is complete despite the invalid wrapper exit.
- Activation result: `activation_summary compared=54 passed=54 failed=0 missing=0 extra=0 worst_abs=3.612041e-2 worst_rel=2.050613e4 worst_cosine=0.99999923`.
- Timing result: `timing_summary compared=76 passed=75 failed=1 missing=0 extra=0 ignored=0 actual_total_ms=87.142 baseline_total_ms=175.887 speedup=2.02x`.
- Only failed row: `timing/cells/cell_0000/channel_mixer.time.json`, `1.812ms` actual vs `1.668ms` baseline, `0.92x`; this is the already characterized narrow channel-mixer edge row, not a value-residual-gate row.
- New value-residual-gate autotune log:
  - key now has `max_line_size=8`, proving the axis bug in candidate eligibility is fixed.
  - selected `fastest_index=1`, `line_size_2`.
  - candidates: `line_size_2` median `220.868us`, `line_size_4` median `221.572us`, `line_size_1` median `221.604us`, `line_size_8` median `222.836us`.
- Interpretation so far: correctness is preserved and runtime dispatch now has the proper candidate set. The direct autotune win is tiny, and total compare is still strongly above `1.0x` but a bit slower than the prior `84.876ms` JSON reconstruction. Need profiler confirmation of the actual selected kernel suffix and total value-residual-gate time before keep/revert.
- Next command: short remote `nsys` profile with `repeat=1,warmup=1`, output `target/rwkv-test/nsys-value-residual-gate-vector-axis`, then query `value_residual_gate_forward` kernel group.

## 2026-05-16 Nsys Wrapper Check

- Nsys command returned nonzero before printing the sqlite query result. This may be the usual `compare-rwkv-nn` nonzero timing status under profiler or an `nsys` export issue.
- Next command: inspect `target/rwkv-test/nsys-value-residual-gate-vector-axis.log` and whether `.sqlite` exists, then query the sqlite directly if present.

## 2026-05-16 Nsys Result

- Artifact check: generated
  - `/home/caizus/Projects/Packages/rwkv-rs-stable/target/rwkv-test/nsys-value-residual-gate-vector-axis.nsys-rep`
  - `/home/caizus/Projects/Packages/rwkv-rs-stable/target/rwkv-test/nsys-value-residual-gate-vector-axis.sqlite`
- The `nsys` compare timing status is invalid for acceptance because it used `repeat=1,warmup=1` against a `repeat=3,warmup=1` baseline, but the profiler artifact is valid for kernel attribution.
- `nsys` kernel result: `value_residual_gate_forward_kernel_f__n_2`, `33` launches, `6.741ms` total, `204.282us` average, `16` registers/thread, grid `12288x1x1`, block `32x8x1`.
- Previous clean-profile baseline for this same kernel family was `value_residual_gate_forward_kernel_f__n_1`, `33` launches, `6.620ms` total, `200.608us` average, grid `24576x1x1`, block `32x8x1`.
- Interpretation: the code fix is real and changes dispatch from `n_1` to `n_2`, but the targeted profiler delta is effectively neutral to slightly negative (`+0.121ms`, about `1.8%` slower in this one `nsys` sample). The autotune log measured `line_size_2` slightly faster than `line_size_1` in the retune run, so this difference is within the current profiler/tuner noise floor rather than a clear performance win.
- Decision: keep the source change for now as a dispatch-correctness fix, not as a claimed speedup. It fixes the `max_line_size` eligibility bug, preserves activation, keeps remote total speedup above `1.0`, and lets the runtime choose among real candidates instead of accidentally forcing `line_size_1`. Do not claim this branch improved total time. If later clean runs show a repeatable value-residual regression, revert this branch while preserving the note.
- Keep/revert state: kept.
