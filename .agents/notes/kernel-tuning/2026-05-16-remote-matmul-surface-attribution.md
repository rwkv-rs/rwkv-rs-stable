# Remote Matmul Surface Attribution

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-remote-matmul-surface-attribution-20260516` in the existing dirty workspace.
- Dirty-tree constraint: this checkout carries broad unrelated uncommitted workspace changes and prior kernel-tuning notes. This attempt is read-only attribution of existing remote `nsys` artifacts; it must not edit kernel code.
- Prior-note search command:
  - `rg -n "10\\.100\\.1\\.253|remote|GB10|channel_mixer|channel mixer|matmul|TMA|LocalTuner|ncu|speedup|0\\.87|0\\.86|timing/cells/cell_0000/channel_mixer" .agents/notes/kernel-tuning /root/.codex/memories/MEMORY.md -S`
  - `rg -n "forward_logits|unembed|lm_head_l2wrap_ce|channel_mixer|TimeMixer::forward|struct RwkvLM|fn forward" crates/rwkv-nn/src crates/rwkv-test/src -S`
- Matched prior evidence:
  - `2026-05-16-remote-channel-mixer-edge-ncu.md` shows remote ncu is blocked by `ERR_NVGPUCTRPERM`; standard compare after revert is `75/76 PASS`, total `1.48x`, with only `cell_0000/channel_mixer` failing.
  - `2026-05-16-remote-time-mixer-evidence.md` shows the dominant remote GPU kernel family is Burn/Cubek BF16 TMA matmul, with the largest single shape adjacent to `lm_head_l2wrap_ce`.
  - `2026-05-16-lm-head-forward-online-softmax-gb10.md` retried online softmax on GB10 and reverted it after nsys showed the row kernel itself regressed.
  - Channel-mixer forced Cube, Burn reference/fusion, LocalTuner bypass, residual-add A/B, forced WKV7 `row_tile=16`, and lm-head online softmax are duplicate-guarded negative attempts.
- Machine/GPU: remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`, using the already fetched `target/rwkv-test/nsys-remote-time-mixer.sqlite`.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, regenerated remote baseline under `~/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Hypothesis: the next useful implementation branch should target a large remote surface that is not already duplicate-guarded. Since ncu counters are blocked, split existing nsys timeline matmul launches by shape and adjacency to identify whether any project-owned custom kernel or timing boundary can be changed without forking Cubek TMA matmul.
- Command to run: parse `target/rwkv-test/nsys-remote-time-mixer.sqlite` with Python `sqlite3`, resolving `StringIds`, and print top kernels plus neighboring kernel context for dominant BF16 matmul launches.
- Expected keep/revert boundary: keep attribution evidence only. If attribution points only to Cubek TMA matmul internals or already rejected candidates, do not open an implementation branch from this note; choose the next surface with a fresh branch and hypothesis.

## Evidence

- SQLite schema inspection confirmed `CUPTI_ACTIVITY_KIND_KERNEL` includes launch geometry, register count, and shared-memory fields, while `NVTX_EVENTS` is empty. Attribution must rely on timeline order and source-call structure.
- Top remote kernel+shape groups from the existing nsys artifact:
  - `48.276ms / 3` for BF16 TMA matmul `grid=(4096,16,1)`, `block=(32,12,1)`, `regs=73`, `dynamic_smem=27648`.
  - `37.640ms / 207` for `kernel_scalar_binop_c_f32_n_4`, `grid=(6144,1,1)`.
  - `29.330ms / 318` for `kernel_binop_c_bf16_n_8`, `grid=(3072,1,1)`.
  - `25.512ms / 144` for BF16 TMA matmul `grid=(16,48,1)`, `block=(32,12,1)`.
  - `24.875ms / 138` for `unary_float_f_f32_n_4`, `grid=(6144,1,1)`.
  - `23.508ms / 36` for `wkv7_pretrain_forward_output_kernel_f_bf16`.
  - `21.307ms / 36` and `18.590ms / 36` for the two large channel-mixer BF16 TMA matmuls, both with `regs=122`, `dynamic_smem=26624`.
  - `14.247ms / 36` for `mix6_forward_kernel_f__n_1`, `14.050ms / 36` for `channel_mixer_relu_square_forward_kernel_f__n_4`, and `12.787ms / 3` for `lm_head_l2wrap_ce_forward_row_kernel_f__i_i32`.
- Timeline adjacency confirms the `grid=(4096,16,1)` matmul is lm-head projection: each launch is immediately preceded by the final unembed LayerNorm and immediately followed by `lm_head_l2wrap_ce_forward_row_kernel` and finalize.
- Interpretation: do not open another lm-head row algorithm branch from this evidence. The largest single surface is Cubek TMA matmul, and the project-owned row kernel was already tested by the GB10 online-softmax branch and reverted. The next useful source inspection is the repeated generic Burn elementwise/cast groups around TimeMixer/LoRA, because their aggregate time is larger than any single custom kernel and may expose a project-owned fusion boundary.
- Next command: inspect `TimeMixer::forward`, LoRA forward, and related custom train kernels to map the large generic `kernel_scalar_binop_c_f32_n_4`, `kernel_binop_c_bf16_n_8`, `unary_float_f_f32_n_4`, and cast groups to source-level operations before opening an implementation branch.
- Source inspection result: `WeightPrepare::forward` still computes `learning_rate = sigmoid(self.param_learning_rate_lora.forward(learning_rate_input))` and value residual as `sigmoid(value_residual_lora.forward(value_input))` followed by Burn `lerp`. The project already has custom `learning_rate_gate` and `value_residual_gate` forward/backward kernels with hardware/shape autotune keys, but the remote time-mixer evidence found no tuner logs for them because this path is not wired into `WeightPrepare`.
- Implementation boundary found: keep the LoRA matmul path in Burn/Cubek so tensor-core matmul selection is preserved, but split LoRA into `(matmul output without bias, bias vector)` for these two NoOP+bias gates and feed those into the existing fused gate kernels. This is not a repeat of the prior key-only work; it wires already-keyed kernels into the real model path.
- Decision: open a separate implementation branch for wiring `learning_rate_gate` and `value_residual_gate` into `WeightPrepare::forward`. Keep this attribution branch read-only.
