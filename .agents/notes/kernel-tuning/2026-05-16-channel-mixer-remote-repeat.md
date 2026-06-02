# Channel Mixer Remote Repeat

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-channel-mixer-remote-repeat-20260516` in the existing dirty workspace.
- Dirty-tree constraint: this branch inherits the live LayerNorm runtime-dispatch code from `kernel-tuning-layernorm-runtime-dispatch-20260516` and broad unrelated dirty workspace changes. This attempt is read-only measurement; it does not edit kernels.
- Prior-note search command: `rg -n "channel_mixer|channel mixer|matmul|TMA|LocalTuner|relu_square|Burn reference|forced Cube|fusion|lhs_size|rhs_size|ncu|speedup" .agents/notes/kernel-tuning /root/.codex/memories/MEMORY.md -S`.
- Matched prior evidence:
  - `2026-05-16-channel-mixer-cube-matmul.md`: forcing `MatmulStrategy::Cube` made channel mixer matmuls slower; do not repeat forced Cube.
  - `2026-05-16-channel-mixer-matmul-fusion-analysis.md`: Burn/Cubek does not expose a small public TMA matmul epilogue hook for `relu_square`; do not repeat Burn-reference/fusion without a new implementation boundary.
  - `2026-05-16-channel-mixer-tma-matmul-ncu.md`: current TMA matmul profiling was started but its first ncu run used a stale binary; ncu rerun remained pending.
  - `2026-05-16-layernorm-runtime-dispatch.md`: remote regenerated-baseline compare after LayerNorm runtime dispatch passed activation and total speedup `1.51x`, with only `cells/cell_0000/channel_mixer.time.json` failing at `0.89x`.
- Machine/GPU: remote `caizus@10.100.1.253`, `NVIDIA GB10`, regenerated baseline under `~/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Kernel/stage: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, channel mixer forward timing rows.
- Changed boundary: this is not a new channel mixer implementation. It is a repeat measurement on the already-synced LayerNorm dispatch tree to decide whether the single remote channel mixer fail is noise before opening an implementation branch.
- Command to run: `ssh -i ~/.ssh/id_ed25519_dgx_spark_windows -o BatchMode=yes -o ConnectTimeout=8 caizus@10.100.1.253 'cd ~/Projects/Packages/rwkv-rs-stable && cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline ~/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1'`.
- Expected keep/revert boundary: if timing becomes `76/76 PASS`, record the previous channel mixer fail as measurement jitter. If the same row or group fails again, open a separate channel mixer implementation/profiling branch instead of changing code here.

## 2026-05-16 Repeat Result

- Remote repeat result: activation stayed valid, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: `timing_summary compared=76 passed=74 failed=2 actual_total_ms=120.909 baseline_total_ms=175.887 speedup=1.45x`.
- Failed rows:
  - `cells/cell_0000/channel_mixer.time.json`: `1.698ms` vs `1.668ms`, `0.98x`.
  - `cells/cell_0010/channel_mixer.time.json`: `1.878ms` vs `1.870ms`, shown as `1.00x` but still slower by `0.008ms`.
- Interpretation: the remote total remains above `1.0x`, and the remaining row failures are marginal channel-mixer rows rather than LayerNorm. Because the same group failed again, do not treat it as fully resolved by jitter. Open a dedicated channel-mixer profiling branch before any implementation change.
