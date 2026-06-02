# Local GPU Availability Preflight

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-local-gpu-availability-preflight-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout contains broad unrelated workspace edits and previous kernel-tuning notes. This attempt owns only this note unless a later entry explicitly expands scope.
- User constraint: do not use local GPU timing while the user may be running other work locally. This branch only checks availability with `nvidia-smi`; it does not run `compare-rwkv-nn`, `nsys`, `ncu`, or any GPU workload.
- Objective gap: current completion audits mark fresh local speedup `>1.0` as incomplete. This preflight checks whether the local machine is plausibly available for a later acceptance run, without consuming timing resources now.
- Command to run:
  - `nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total,driver_version --format=csv`
  - `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv`
- Decision boundary:
  - If compute apps or nontrivial utilization are present, keep local acceptance blocked.
  - If the GPU appears idle, still do not run timing in this branch; a later branch must record explicit local acceptance scope before running `compare-rwkv-nn`.

## Preflight Result

- `nvidia-smi --query-gpu`:
  - GPU `0`: `NVIDIA GeForce RTX 5090`, driver `596.36`.
  - Utilization: `4%`.
  - Memory: `3091 MiB / 32607 MiB`.
- `nvidia-smi --query-compute-apps`: no compute apps listed.
- Interpretation: no active compute workload is visible. The remaining memory/utilization looks like display or resident context noise rather than a training run.
- Scope expansion: run one standard local acceptance compare in this branch because the explicit blocker is not currently visible. Treat it as the fresh local acceptance candidate; if timing fails, keep the result and do not open duplicate kernel branches without profiler evidence.
- Next command:
  - `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --repeat 3 --warmup 1`

## Local Acceptance Result

- Command: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --repeat 3 --warmup 1`, captured in `target/rwkv-test/local-gpu-availability-acceptance-compare.log`.
- Build provenance: release build completed, then `target/release/rwkv-test` ran the compare.
- Activation: `activation_summary compared=54 passed=54 failed=0 missing=0 extra=0`.
- Timing: `timing_summary compared=76 passed=13 failed=63 missing=0 extra=0 ignored=1 actual_total_ms=33.446 baseline_total_ms=35.957 speedup=1.08x`.
- Module timing signal:
  - `cells/*/time_mixer`: `17.065ms` actual vs `27.836ms` baseline, `1.63x`.
  - `layer_norm0`: `0.161ms` actual vs `0.194ms` baseline, `1.20x`.
  - `cells/*/channel_mixer`: `9.286ms` actual vs `5.527ms` baseline, `0.60x`.
  - pre/post context and pre-layernorm rows remain slower than baseline.
  - `lm_head`: `0.119ms` actual vs `0.034ms` baseline, `0.29x`.
  - `loss/l2wrap_cross_entropy`: `0.880ms` actual vs `0.717ms` baseline, `0.81x`.
- Interpretation:
  - This is the first fresh local compare in the current branch set with total speedup above `1.0`.
  - It is not row-clean acceptance: `63/76` canonical timing rows still fail.
  - The remaining local row failures match known local blockers from earlier post-gates attribution: channel-mixer/Cubek matmul, pre-layernorm/context timing rows, lm-head, and loss. Do not open duplicate channel-mixer, residual, lm-head row, or ordinary LayerNorm block-size branches from this result alone.
- Completion impact:
  - The explicit local total speedup `>1.0` gap is now covered by current evidence.
  - The remaining completion blocker is still remote `ncu` achieved-counter evidence, because 253 performance counters are permission-blocked.
