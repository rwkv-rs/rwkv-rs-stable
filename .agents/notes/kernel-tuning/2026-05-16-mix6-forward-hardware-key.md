# Mix6 Forward Hardware Key

- Date: 2026-05-16
- Branch/worktree: `kernel-tuning-mix6-forward-hw-key-20260516` in the existing dirty workspace.
- Dirty-tree constraint: `crates/rwkv-nn/src/kernels/train/time_mixer/mix6/forward.rs` was already modified before this branch; the pre-existing diff had removed the older `Mix6ForwardAutotuneKey` and replaced it with direct `best_line_size(...)` selection. This attempt works on top of that current state and does not revert unrelated workspace changes.
- Prior-note search command: `rg -n "mix6|Mix6|forward|line_size|best_line_size|AutotuneKey|LocalTuner|vector width|residual|残差" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md /root/.codex/memories/MEMORY.md`.
- Matched prior evidence:
  - `2026-05-16-autotune-key-audit.md` records `time_mixer/mix6/forward.rs` as a real gap because vector width is selected without an `AutotuneKey` / `LocalTuner`.
  - `2026-05-16-backward-autotune-hardware-key.md` already expanded `Mix6BackwardAutotuneKey` with runtime, shape, hardware capability, vector width limit, alias, and deterministic fields.
  - `MEMORY.md` records that train-kernel validation should run real forward/backward paths, and that `mix6` has trace-specific BF16 tolerance constraints.
- Scope: `crates/rwkv-nn/src/kernels/train/time_mixer/mix6/forward.rs`.
- Hypothesis: restoring mix6 forward line-size tuning with the full hardware/shape key prevents vector-width choices from crossing runtimes/devices and aligns the forward path with the current backward and elementwise key design.
- Candidate parameters: `line_size` from `[1, 2, 4, 8, 16, 32, 64]`.
- Expected keep/revert boundary: keep if the crate compiles and trace-backed CUDA validation remains activation-safe; treat timing separately because key design alone may not improve `.time.json`.

## Result

- Code change: restored `LocalTuner` for mix6 forward line-size selection and expanded `Mix6ForwardAutotuneKey` with runtime, dtype, `num_elements`, `batch_size`, `context_len`, `embedded_dim`, `rows`, CubeCL hardware fingerprint fields, `max_line_size`, `is_in_place`, and `deterministic`.
- Formatting: `rustup run nightly rustfmt crates/rwkv-nn/src/kernels/train/time_mixer/mix6/forward.rs`.
- Compile check: `cargo check -p rwkv-nn --features cuda` passed.
- Trace compare command: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --repeat 3 --warmup 1`.
- Trace compare activation result: `activation_summary compared=54 passed=54 failed=0 missing=0 extra=0`.
- Trace compare timing result: command exited with timing failure: `timing_summary compared=76 passed=10 failed=66 missing=0 extra=0 ignored=0 actual_total_ms=43.380 baseline_total_ms=35.957 speedup=0.83x`.
- Module timing signal: `cells/*/time_mixer` was `actual_total_ms=25.351`, `baseline_total_ms=27.836`, `speedup=1.10x`; remaining global failures are dominated by `channel_mixer`, post-stage context timings, pre-layer-norm timings, `lm_head`, and `loss/l2wrap_cross_entropy`.
- Decision: keep this branch as a key-design correction because compile and activation are clean and the directly relevant time-mixer group remains above baseline. Do not claim overall acceptance; next tuning branch should target a remaining forward slow group with ncu evidence.
