# LayerNorm Runtime Dispatch

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-layernorm-runtime-dispatch-20260516` in the existing dirty workspace.
- Dirty-tree constraint: the checkout already carries broad uncommitted workspace and previous tuning changes. This attempt is scoped to `crates/rwkv-nn/src/kernels/train/layer_norm/forward.rs` plus this note.
- Prior-note search command: `rg -n "layer_norm|LayerNorm|BLOCK_SIZE|block_size|256|512|768|1024|drift|GB10|10\\.100\\.1\\.253|LocalTuner|AutotuneKey|accuracy guard|deterministic" .agents/notes/kernel-tuning /root/.codex/memories/MEMORY.md -S`.
- Matched prior evidence:
  - `2026-05-16-layernorm-hardware-policy.md` added hardware fields to the LayerNorm key and kept local BF16 `D=768` deterministic minimum at `1024` after `256`, `512`, and `768` drifted locally.
  - `2026-05-16-layernorm-safe-path-analysis.md` says a combined-reduction implementation passed activation but regressed timing; this branch must not repeat that implementation.
  - `MEMORY.md` records remote `10.100.1.253` regenerated-baseline evidence where GB10 accepted LayerNorm `BLOCK_SIZE=256` and reached `speedup=1.49x`, while local override with `256` failed activation and timing.
  - `2026-05-16-remote-current-key-compare.md` records the working remote SSH key and `~/Projects/Packages/...` layout.
- Machine/GPU: local RTX 5090 compute capability `12.0`; remote target `caizus@10.100.1.253` is expected GB10 but must be rechecked in-command before accepting a remote-only branch result.
- Kernel/stage: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, LayerNorm forward.
- Changed boundary: this is not a block-size-only retry. It changes the deterministic block-size policy into a hardware/shape runtime dispatch rule that can keep local `1024` while allowing the remote GB10 `256` case only under a specific hardware/shape/dtype key.
- Hypothesis: using CubeCL-exposed hardware fields as a conservative remote GB10 fingerprint can admit `256` for the known remote `BF16 D=768 rows=8192` case while preserving the local deterministic `1024` guard. This should avoid local activation drift and reduce remote LayerNorm timing.
- Candidate parameters: keep `BLOCK_SIZE_CANDIDATES = [64,128,256,512,768,1024]`; change `deterministic_min_block_size(...)` to receive the full key/hardware context and return `256` only for the verified remote-like hardware/shape, otherwise keep `d_model.next_power_of_two().min(max_units_per_cube)`.
- Planned validation:
  - `rustup run nightly rustfmt crates/rwkv-nn/src/kernels/train/layer_norm/forward.rs`
  - `cargo check -p rwkv-nn --features cuda`
  - local `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --repeat 3 --warmup 1`
  - remote sync/check/compare using regenerated remote baseline if local activation stays valid.
- Expected keep/revert boundary: keep only if local activation stays valid and remote regenerated-baseline compare remains valid. If the hardware fingerprint admits `256` locally or activation drifts, revert this branch and keep the note as negative evidence.

## 2026-05-16 Implementation

- Remote preflight command: `ssh -i ~/.ssh/id_ed25519_dgx_spark_windows -o BatchMode=yes -o ConnectTimeout=8 caizus@10.100.1.253 'nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader; cd ~/Projects/Packages/rwkv-rs-stable && sed -n "1,120p" target/autotune/0.10.0/device-0-0-cuda/rwkv_nn-kernels-train-layer_norm-forward-layer-norm-forward.json.log 2>/dev/null || true'`.
- Remote hardware evidence: `NVIDIA GB10, 12.1`; CubeCL LayerNorm autotune cache records `runtime=cuda`, `dtype=BF16`, `d_model=768`, `rows=8192`, `load_width=128`, `plane_size=32`, `max_units_per_cube=1024`, `max_cube_dim=[1024,1024,64]`, `max_shared_memory_size=101376`, `num_streaming_multiprocessors=48`, `num_tensor_cores=null`, `min_tensor_cores_dim=8`, and previous `deterministic_min_block_size=1024`.
- Local hardware evidence: `NVIDIA GeForce RTX 5090, 12.0`; local CubeCL LayerNorm cache has the same basic cube limits but `num_streaming_multiprocessors=170`.
- Code change: `deterministic_min_block_size(...)` now receives the runtime, dtype, shape, and CubeCL hardware signature. It returns `256` only for the narrow GB10 BF16 `D=768, rows=8192` signature above; all other cases keep `d_model.next_power_of_two().min(max_units_per_cube)`.
- Formatting result: `rustup run nightly rustfmt crates/rwkv-nn/src/kernels/train/layer_norm/forward.rs` passed.
- Compile result: `cargo check -p rwkv-nn --features cuda` passed.
- Local compare command: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --repeat 3 --warmup 1`.
- Local compare result: activation stayed valid, `activation_summary compared=54 passed=54 failed=0`; timing still failed overall, `timing_summary compared=76 passed=10 failed=66 actual_total_ms=43.254 baseline_total_ms=35.957 speedup=0.83x`.
- Local LayerNorm result: `layer_norm0` was `0.130ms` vs baseline `0.194ms`, `1.48x`; per-cell pre-layer-norm rows remain slow around `0.28x`.
- Local autotune evidence: the local LayerNorm cache still records `num_streaming_multiprocessors=170` and `deterministic_min_block_size=1024`, so the GB10 `256` branch did not open on this machine.
- Remote sync result: scoped `rsync --delete` to `caizus@10.100.1.253:~/Projects/Packages/rwkv-rs-stable/` completed, excluding `.git/`, `target/`, generated results, weights, datasets, and profile artifacts.
- Remote compile result: `cargo check -p rwkv-nn --features cuda` passed.
- Remote baseline preflight: `~/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000` exists with `76` `.time.json` files and `68` `.safetensors` files.
- Remote compare command: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline ~/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1`.
- Remote compare result: activation stayed valid, `activation_summary compared=54 passed=54 failed=0`; timing mostly passed, `timing_summary compared=76 passed=75 failed=1 actual_total_ms=116.483 baseline_total_ms=175.887 speedup=1.51x`.
- Remote LayerNorm result: `layer_norm0` was `0.200ms` vs baseline `0.528ms`, `2.63x`; pre-layer-norm groups were `4.74x` for time mix and `4.23x` for channel mix.
- Remote autotune evidence: the GB10 LayerNorm cache appended a key with `num_streaming_multiprocessors=48` and `deterministic_min_block_size=256`; autotune selected `block_256` as fastest (`~106us`) over `512`, `768`, and `1024`.
- Decision: keep this branch's LayerNorm runtime dispatch as a hardware/shape policy improvement. It preserves local correctness by keeping `1024`, and it enables the verified remote GB10 `256` path. The remaining remote failure is `cells/cell_0000/channel_mixer.time.json`, so the next tuning branch should target channel mixer or rerun that row to separate jitter from a real kernel issue.
