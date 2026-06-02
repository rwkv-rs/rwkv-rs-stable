# Forward Elementwise Full Hardware Key

- Date: 2026-05-16
- Branch/worktree: `kernel-tuning-forward-elementwise-full-hw-key-20260516` in the existing dirty workspace.
- Dirty-tree constraint: this checkout already carries uncommitted kernel and skill changes from earlier tuning branches. This attempt only edits forward elementwise autotune keys and leaves the backward key changes intact.
- Prior-note search command: `rg -n "forward elementwise|ChannelMixerElementwiseAutotuneKey|max_cube_dim|num_tensor_cores|min_tensor_cores_dim|hardware key|residual|残差" .agents/notes/kernel-tuning crates/rwkv-nn/src/kernels/train/{channel_mixer,time_mixer/learning_rate_gate,time_mixer/value_residual_gate}/forward.rs`.
- Matched prior evidence:
  - `2026-05-16-forward-elementwise-autotune-key.md` says the forward elementwise keys gained hardware capability fields, but source inspection shows they still lack `max_cube_dim`, `num_tensor_cores`, and `min_tensor_cores_dim`.
  - `2026-05-16-lm-head-forward-hardware-key.md` and LayerNorm forward use those CubeCL hardware fields already.
  - `2026-05-16-skill-guardrails.md` records the residual-add negative result; this attempt does not touch `residual_add`.
- Machine/GPU: local CUDA machine, prior ncu reports CC 12.0.
- Scope: forward elementwise autotune keys for `channel_mixer`, `learning_rate_gate`, and `value_residual_gate`.
- Hypothesis: forward elementwise dispatch keys should include the full CubeCL-exposed hardware fingerprint used by LayerNorm and `lm_head_l2wrap_ce` forward, so cached line-size choices do not cross runtimes/devices with different cube/tensor-core capabilities.
- Candidate parameters: no line-size candidate change; only key fields and Display strings should change.
- Planned commands: `cargo +nightly fmt --all`; `rtk cargo check -p rwkv-nn --features cuda`; reuse the existing compare result only if no kernel behavior changed beyond the key.
- Code change: added `max_cube_dim`, `num_tensor_cores`, and `min_tensor_cores_dim` to `ChannelMixerElementwiseAutotuneKey`, `LearningRateGateForwardAutotuneKey`, and `ValueResidualGateForwardAutotuneKey`, including Display output and key construction from CubeCL hardware properties.
- Format result: `cargo +nightly fmt --all` passed.
- Compile result: `rtk cargo check -p rwkv-nn --features cuda` passed.
- Correctness/timing result: a later full compare on the accumulated key-design tree passed activation (`activation_summary compared=54 passed=54 failed=0`) but still failed timing (`actual_total_ms=44.873 baseline_total_ms=35.957 speedup=0.80x`). Treat this key-only change as cache/dispatch design work, not a speedup claim.
- Decision: keep the key expansion as hardware/shape dispatch correction. Next branch should target a forward slow group with a kernel-level hypothesis and ncu evidence, rather than only expanding cache keys.
