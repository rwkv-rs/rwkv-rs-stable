# LayerNorm D768 Split-512 Reduction

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-layernorm-d768-split512-20260516` in the existing dirty workspace.
- Dirty-tree constraint: this checkout already carries uncommitted kernel, skill, fixture, and crate-structure changes from earlier work. This attempt is scoped to `crates/rwkv-nn/src/kernels/train/layer_norm/**`.
- Prior-note search command: `rg -n "layer_norm|LayerNorm|BLOCK_SIZE|block512|block 512|block768|256|512|768|drift|value_from_first_cell|embedded_context|residual|残差" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md /root/.codex/memories/MEMORY.md`.
- Matched prior evidence:
  - `2026-05-16-layernorm-d768-bf16-block512.md`: regular BF16 `D=768` block `512` failed activation with `value_from_first_cell` and `lm_head/embedded_context` drift.
  - `2026-05-16-layernorm-d768-bf16-block768.md`: regular block `768` also failed the same activation boundary.
  - `2026-05-16-layernorm-hardware-policy.md`: local deterministic boundary stays at `1024`; `256`, `512`, and `768` must not be admitted as ordinary candidates on this machine.
  - `2026-05-16-skill-guardrails.md`: residual-add is a recorded negative result and is out of scope here.
- Machine/GPU: local CUDA BF16 run; remote `10.100.1.252` refused SSH and `10.100.1.253` rejected current SSH credentials before this attempt.
- Kernel/stage: LayerNorm forward, CUDA BF16 `rwkv_lm`, shape `B=16,T=512,D=768`, rows `8192`.
- Hypothesis: a specialized `512`-unit split-tail candidate can keep the smaller launch shape while separating the first `512` columns from the `256` tail during reduction. This is not the old regular `512` retry; the reduction tree and candidate name are different.
- Candidate parameters: `block_size=512`, `num_warps=16`, deterministic `true`, active only for BF16 `D=768`.
- Command: `cargo +nightly fmt --all`; `rtk cargo check -p rwkv-nn --features cuda`; `rtk cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --repeat 3 --warmup 1`.
- Correctness result: failed local trace compare. Activation comparison was `52/54 PASS`; `cells/cell_0000/time_mixer/value_from_first_cell.safetensors` failed with `max_abs=5.117188e0` and cosine `0.99518716`, and `lm_head/embedded_context.safetensors` failed with `max_abs=8.507812e0` and cosine `0.99520072`.
- Timing/profiler result: not accepted because activation failed. The same run reported `timing_summary compared=76 passed=10 failed=66 ... actual_total_ms=41.674 baseline_total_ms=35.957 speedup=0.86x`; `layer_norm0` was `1.81x`, but per-cell pre-layer-norm groups, `lm_head`, `channel_mixer`, and loss remained below target.
- Decision: negative result. The split-tail reduction changed the numerical boundary enough to reproduce the same downstream drift family as the earlier `512` and `768` attempts, with larger absolute drift. Do not retry split-tail `512` for local BF16 `D=768` unless a new branch changes the numerical algorithm or adds a trace-backed accuracy guard before selection.
- Keep/revert state: implementation reverted in this branch; keep this branch and note as negative evidence.
