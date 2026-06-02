# LM Head Forward Hardware Key

- Date: 2026-05-16
- Branch/worktree: `kernel-tuning-lm-head-forward-hw-key-20260516` in the existing dirty workspace.
- Prior-note search command: `rg -n "LmHeadL2WrapCeForwardAutotuneKey|num_tensor_cores|min_tensor_cores_dim|max_cube_dim|compute capability|max_units_per_cube|is_in_place|deterministic|block_size|num_warps|AutotuneKey|hardware" crates/rwkv-nn/src .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md /root/.codex/memories/MEMORY.md`.
- Matched prior evidence:
  - `kernel-tuning` skill says implementation choice should be keyed by backend/runtime, GPU architecture or compute capability when available, dtype, `d_model`, rows, block size, warp count, vector width, in-place/alias behavior, and deterministic numeric boundary requirements.
  - LayerNorm hardware policy already added CubeCL-exposed hardware fields `max_cube_dim`, `num_tensor_cores`, and `min_tensor_cores_dim`.
  - `2026-05-16-lm-head-backward-autotune.md` documents that lm-head backward already records runtime, dtype, rows, vocab, vector/hardware fields, and candidate block size via tunable names.
  - `2026-05-16-lm-head-forward-target-logit.md` showed the forward row kernel is dominated by full-vocab passes; this attempt is not another row-kernel math tweak.
- Scope: `LmHeadL2WrapCeForwardAutotuneKey` in `crates/rwkv-nn/src/kernels/train/lm_head_l2wrap_ce/forward.rs`.
- Hypothesis: forward fused loss dispatch should include the same hardware/shape discriminators as LayerNorm where CubeCL exposes them, so cache entries do not cross devices with different cube limits/tensor-core shape. This is correctness-safe key design, not a claimed speedup by itself.
- Candidate parameters: block size and `num_warps = block_size / 32` remain represented by tunable names; this change expands the key with additional runtime/hardware/alias fields.
- Code change: added `max_cube_dim`, `num_tensor_cores`, `min_tensor_cores_dim`, and `is_in_place` to `LmHeadL2WrapCeForwardAutotuneKey` and its display string. The actual forward candidates and kernel launches are unchanged.
- Planned commands: `rtk cargo check -p rwkv-nn --features cuda`; if that passes, run the standard compare once to ensure no activation regression.
- Compile result: `rtk cargo check -p rwkv-nn --features cuda` passed.
- Compare command: `rtk cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --repeat 3 --warmup 1`.
- Compare result: activation stayed valid (`activation_summary compared=54 passed=54 failed=0`). Timing remains below target: `actual_total_ms=43.039`, baseline `35.957`, speedup `0.84x`; `layer_norm0` passed at `1.48x`, but channel mixer, pre-layer-norm, `lm_head`, and loss rows remain slower than baseline.
- Decision: keep this key expansion as correctness-safe design work toward hardware/shape dispatch. Do not claim it as a performance win; the total speedup target is still unmet.
