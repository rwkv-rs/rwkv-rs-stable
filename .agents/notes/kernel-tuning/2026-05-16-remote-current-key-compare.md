# Remote Current Key Compare

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-remote-current-key-compare-20260516` in the existing dirty workspace.
- Dirty-tree constraint: this branch measures the accumulated current key-design tree after the local negative attempts were reverted. The workspace still carries unrelated and broader uncommitted changes; sync must exclude build outputs, weights, datasets, results, and target directories.
- Prior-note/memory search command: `rg -n "10\\.100\\.1\\.253|remote|GB10|LayerNorm|BLOCK_SIZE|compare-rwkv-nn|rsync|sync" .agents/notes/kernel-tuning /root/.codex/memories/MEMORY.md /root/.codex/memories/extensions/ad_hoc/notes -S`.
- Matched prior evidence:
  - `MEMORY.md` says future timing/kernel tuning should prefer remote host `10.100.1.253`, use `~/Projects/...` if `/workspace` is absent, and keep scoped `rsync --delete` to code repo roots while excluding target/results/weights/datasets.
  - `MEMORY.md` records the durable compare command: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --repeat 3 --warmup 1`.
  - `MEMORY.md` records that remote regenerated-baseline LayerNorm `BLOCK_SIZE=256` passed and reached `speedup=1.49x`, while local smaller-block LayerNorm drifted; this run must not generalize either result without current evidence.
  - Local current evidence after key-design work: `mix6` and `residual_add` key changes kept activation passing but local total speedup remained below `1.0`; later `lm_head` candidate pruning and custom residual wiring were reverted after negative local timing.
- Machine/GPU: planned remote host `10.100.1.253`, expected GB10-class GPU but must verify with `nvidia-smi`/ncu output.
- Kernel/stage: full CUDA BF16 `rwkv-test compare-rwkv-nn` train-forward steady-state, shape `B=16,T=512,D=768`, plus focused ncu only if compare still fails.
- Hypothesis: the current hardware/shape keyed tree may pass on the remote regenerated-baseline environment differently from local, especially for LayerNorm policy. First establish current remote activation/timing before adding more local-only kernel changes.
- Candidate parameters: no live code candidate in this branch; measurement candidate is the current key-design tree on remote hardware/baseline.
- Planned preflight commands:
  - `ssh -o BatchMode=yes -o ConnectTimeout=5 10.100.1.253 'hostname; pwd; command -v nvidia-smi || true; command -v ncu || true; rustc --version || true; cargo --version || true; test -d /workspace && echo HAS_WORKSPACE || echo NO_WORKSPACE; ls -lah ~/Projects/Packages 2>/dev/null || true'`
  - If SSH works, scoped `rsync --delete` of this repo to the remote code root, excluding `.git/`, `target/`, results, weights, datasets, checkpoints, and other generated output.
  - Remote `cargo check -p rwkv-test --features cuda`, then standard compare.
- Expected keep/revert boundary: this branch is measurement-only unless remote evidence identifies a new hardware-specific implementation change. Do not edit kernels here unless a new attempt note/branch is created.

## 2026-05-16 Remote Preflight

- Direct root SSH failed with `Permission denied (publickey,password)`.
- Direct `caizus@10.100.1.253` also failed without an explicit key.
- Working SSH command: `ssh -i ~/.ssh/id_ed25519_dgx_spark_windows -o BatchMode=yes -o ConnectTimeout=5 caizus@10.100.1.253 '...'`.
- Remote preflight result:
  - hostname `spark-35ac`
  - home `/home/caizus`
  - `nvidia-smi` at `/usr/bin/nvidia-smi`
  - `rustc 1.95.0`, `cargo 1.95.0`
  - `/workspace` absent, so use `~/Projects/Packages/rwkv-rs-stable`
  - `~/Projects/Packages` contains `rwkv-rs-stable` and `rwkv-rs-test`
- Updated sync command before running: scoped `rsync --delete` to `caizus@10.100.1.253:~/Projects/Packages/rwkv-rs-stable/` using the working key, excluding `.git/`, `target/`, results, outputs, checkpoints, weights, data, datasets, and generated benchmark/profile artifacts.
- Sync result: scoped rsync completed successfully to `~/Projects/Packages/rwkv-rs-stable/`.
- Remote check command planned: `ssh -i ~/.ssh/id_ed25519_dgx_spark_windows caizus@10.100.1.253 'cd ~/Projects/Packages/rwkv-rs-stable && cargo check -p rwkv-test --features cuda'`.
- Remote check result: `cargo check -p rwkv-test --features cuda` passed on `spark-35ac`.
- Remote standard compare command planned: `ssh -i ~/.ssh/id_ed25519_dgx_spark_windows caizus@10.100.1.253 'cd ~/Projects/Packages/rwkv-rs-stable && cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --repeat 3 --warmup 1'`.
- Remote standard compare result with synced default baseline: activation passed (`54/54 PASS`) but timing failed with `actual_total_ms=119.866`, `baseline_total_ms=35.957`, `speedup=0.30x`. This result uses the synced local baseline timings and is invalid for remote acceptance because the baseline provenance is local, not remote regenerated.
- Remote regenerated-baseline command planned: first verify `~/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`, then run `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline ~/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1`.
