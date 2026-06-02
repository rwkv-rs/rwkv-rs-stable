# Residual Add Hardware Key

- Date: 2026-05-16
- Branch/worktree: `kernel-tuning-residual-add-hw-key-20260516` in the existing dirty workspace.
- Dirty-tree constraint: `crates/rwkv-nn/src/kernels/train/residual_add/forward.rs` is already untracked in this checkout from prior custom residual-add work. This attempt edits that current custom path and does not compare against or revert to Burn tensor addition.
- Prior-note search command: `rg -n "residual_add|Burn add|Burn addition|trace_residual_add|plain Burn|custom residual|vector width|line_size|AutotuneKey|LocalTuner" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md /root/.codex/memories/extensions/ad_hoc/notes/2026-05-15T21-10-06-rwkv-nn-timing-optimization-iterations.md /root/.codex/memories/extensions/ad_hoc/notes/2026-05-15T21-26-58-rwkv-nn-sync-ablation.md /root/.codex/memories/MEMORY.md`.
- Matched prior evidence:
  - The timing optimization note says the custom residual-add kernel was better than switching the trace writer back to plain Burn tensor add; this attempt does not repeat that A/B.
  - The sync-ablation note says alias/in-place output reuse for `residual_add` and `channel_mixer_relu_square` improved the stable run from about `0.69x` to about `0.77x` but still needed validation.
  - `2026-05-16-autotune-key-audit.md` records `residual_add/forward.rs` as the remaining vector-width key gap.
- Scope: `crates/rwkv-nn/src/kernels/train/residual_add/forward.rs`.
- Hypothesis: keep the custom residual-add implementation, but move line-size selection behind `LocalTuner` with runtime, hardware, shape, vector-width, alias, and deterministic key fields so this path follows the same hardware/shape dispatch model as the other train kernels.
- Candidate parameters: `line_size` from `[1, 2, 4, 8, 16, 32, 64]`.
- Expected keep/revert boundary: keep if CUDA compile and activation comparison remain clean; if timing regresses, investigate whether host tuner lookup, candidate choice, or alias handling caused it before deciding revert.

## Result

- Code change: added `ResidualAddForwardAutotuneKey`, `LocalTuner`, and line-size candidates for the existing custom residual-add path. The key includes runtime, dtype, `num_elements`, `rows`, `innermost_dim`, CubeCL hardware fingerprint fields, `max_line_size`, lhs/rhs in-place alias state, and deterministic flag.
- Formatting: `rustup run nightly rustfmt crates/rwkv-nn/src/kernels/train/residual_add/forward.rs`.
- Compile check: `cargo check -p rwkv-nn --features cuda` passed.
- Trace compare command: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --repeat 3 --warmup 1`.
- Trace compare activation result: `activation_summary compared=54 passed=54 failed=0 missing=0 extra=0`.
- Trace compare timing result: command exited with timing failure: `timing_summary compared=76 passed=8 failed=68 missing=0 extra=0 ignored=0 actual_total_ms=42.363 baseline_total_ms=35.957 speedup=0.85x`.
- Module timing signal: `cells/*/time_mixer` stayed above baseline at `actual_total_ms=26.322`, `baseline_total_ms=27.836`, `speedup=1.06x`. Main remaining failed groups are still `channel_mixer`, post-stage context timings, pre-layer-norm timings, `lm_head`, and `loss/l2wrap_cross_entropy`.
- Decision: keep as a hardware/shape dispatch correction for the custom residual path. Do not claim overall acceptance; the next performance branch should target a current failing forward group with ncu evidence.
