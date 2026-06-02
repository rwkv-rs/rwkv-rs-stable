# NCU Admin Run Script GB10

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-ncu-admin-run-script-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout contains broad unrelated workspace edits and previous kernel-tuning notes. This attempt owns only this note and the admin script unless a later entry explicitly expands scope.
- User constraint: use `10.100.1.253` first; do not rely on local GPU for remote counter evidence.
- Scope: create an admin-run script for the remaining `ncu` achieved-counter blocker. Do not run the script as normal `caizus` because `RmProfilingAdminOnly: 1` and lack of passwordless sudo are already confirmed.
- Prior evidence:
  - `2026-05-16-ncu-blocker-verification-253.md`: `ncu 2025.3.1.0` is installed, runbook section names are valid, but normal-user counter collection remains permission-blocked.
  - `2026-05-16-privileged-ncu-command-plan-gb10.md`: records the top-owned and projection-matmul command plan.
  - `2026-05-16-completion-audit-local-speedup-ncu-blocked.md`: local and 253 total speedup requirements are covered; the remaining hard blocker is `ncu` achieved occupancy / warp efficiency / memory throughput on 253.
- Machine/GPU target: `10.100.1.253` / `spark-35ac` / `NVIDIA GB10`.
- Baseline:
  - current acceptance baseline: `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`
  - projection R9 baseline: `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_r9_20260516/rwkv_lm/bf16/case_000000`
- Next edit: add `.agents/notes/kernel-tuning/ncu-gb10-admin-run.sh` with exact commands and validity checks.

## Script

- Added `.agents/notes/kernel-tuning/ncu-gb10-admin-run.sh`.
- Script behavior:
  - uses `/home/caizus/Projects/Packages/rwkv-rs-stable` by default, overrideable with `RWKV_RS_STABLE_ROOT`;
  - checks `ncu --version`, `RmProfilingAdminOnly`, `nvidia-smi`, and baseline directories;
  - builds `target/release/rwkv-test`;
  - runs top-owned custom-kernel `ncu` capture into `target/rwkv-test/ncu-top-owned-gb10.csv`;
  - runs projection-like BF16 matmul `ncu` capture into `target/rwkv-test/ncu-lm-head-projection-matmul-gb10.csv`;
  - uses sections `SpeedOfLight`, `Occupancy`, `SchedulerStats`, `WarpStateStats`, and `MemoryWorkloadAnalysis`;
  - keeps compare timing under `ncu` non-acceptance by using `repeat=1,warmup=1`; CSV counters are the artifact.
- Local validation:
  - `bash -n .agents/notes/kernel-tuning/ncu-gb10-admin-run.sh` passed.
- Do not run this script as normal `caizus`; it is intended for an admin-enabled shell or a host where `RmProfilingAdminOnly` is `0`.

## Privileged Continuation

- User provided interactive sudo access for `10.100.1.253`.
- Next command: run `.agents/notes/kernel-tuning/ncu-gb10-admin-run.sh` on `10.100.1.253` with sudo in the remote repo.
- Secret handling boundary: do not write the sudo password into any file, shell command argument, environment file, note, or committed artifact.
- Expected artifact paths:
  - `/home/caizus/Projects/Packages/rwkv-rs-stable/target/rwkv-test/ncu-top-owned-gb10.csv`
  - `/home/caizus/Projects/Packages/rwkv-rs-stable/target/rwkv-test/ncu-lm-head-projection-matmul-gb10.csv`
- Validity boundary: if `ncu` still reports `ERR_NVGPUCTRPERM`, the run is invalid and the next action is to enable non-admin performance counters at the driver level or run from a root-owned shell on the host.

## Invalid Run: Sudo PATH

- Command attempted on `10.100.1.253`: `sudo -E bash .agents/notes/kernel-tuning/ncu-gb10-admin-run.sh`.
- Result: invalid before profiling because sudo environment could not find `cargo`.
- Evidence:
  - `ncu --version` printed `2025.3.1.0`.
  - `RmProfilingAdminOnly: 1` and GB10 GPU info printed.
  - Script failed at build step with `cargo: command not found`.
- Follow-up edit: prepend `/home/caizus/.cargo/bin`, `/usr/local/cuda-13.0/bin`, and `/usr/local/cuda/bin` to `PATH` inside the run script.
- Next command: resync the script to `10.100.1.253` and rerun the same sudo script.

## Top-Owned NCU CSV

- Command rerun on `10.100.1.253` with updated PATH reached `ncu` capture and generated `/home/caizus/Projects/Packages/rwkv-rs-stable/target/rwkv-test/ncu-top-owned-gb10.csv`.
- Artifact size/evidence: `11220` CSV lines, about `2.6M`.
- Activation under the captured run: `54/54 PASS`.
- Timing output under `ncu` is invalid for speedup decisions because profiler replay makes the measured `.time.json` values enormous and the compare exits with timing profile mismatch (`repeat=1,warmup=1` vs baseline `repeat=3,warmup=1`).
- The script stopped after the first capture because `compare-rwkv-nn` returned status `1`; the projection-matmul capture did not run in this script invocation.
- Next command: run only the projection BF16 matmul `ncu` capture with the same sections and projection R9 baseline.

## Projection NCU CSV And Counter Summary

- Projection command run separately on `10.100.1.253`:
  - kernel regex: `matmul_entry_lhs_bf16_lhs_size_1_rhs_bf16_rhs_size_1_acc_bf16_acc_size_8`
  - sections: `SpeedOfLight`, `Occupancy`, `SchedulerStats`, `WarpStateStats`, `MemoryWorkloadAnalysis`
  - artifact: `/home/caizus/Projects/Packages/rwkv-rs-stable/target/rwkv-test/ncu-lm-head-projection-matmul-gb10.csv`
- The two artifacts were copied back locally:
  - `target/rwkv-test/ncu-top-owned-gb10.csv`
  - `target/rwkv-test/ncu-lm-head-projection-matmul-gb10.csv`
- Projection run activation: `54/54 PASS`.
- Projection timing under `repeat=1,warmup=1` and projection R9 baseline printed total speedup `1.42x`, with `lm_head/projection` `15.017ms` vs `15.158ms` (`1.01x`), but row status is still marked mismatch because actual profile settings differ from baseline `repeat=9,warmup=3`.

Median counters from `ncu-top-owned-gb10.csv`:

| kernel | duration ns | achieved occupancy | theoretical occupancy | SM throughput | memory throughput | no eligible | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `layer_norm_forward_kernel_f_` | 144864 | 95.26% | 100.00% | 29.89% | 29.89% | 67.10% | Occupancy is high; issue is latency/eligible-warps, not launch underfill. |
| `wkv7_pretrain_forward_output_kernel_f_bf16` | 736512 | 16.49% | 33.33% | 48.98% | 48.98% | 71.46% | Confirms structural underfill; achieved occupancy is about half of theoretical and grid is only `(12,16,1)`. |
| `lm_head_l2wrap_ce_forward_row_kernel_f__i_i32` | 4264192 | 66.20% | 66.67% | 22.12% | 20.64% | 78.77% | Occupancy is at theoretical; remaining issue is low eligible warp / latency, not occupancy alone. |
| `channel_mixer_relu_square_forward_kernel_f__n_2` | 456640 | 70.06% | 100.00% | 13.46% | 13.97% | 93.36% | Very low eligible warp rate; likely memory/dependency latency dominated. |
| `mix6_forward_kernel_f__n_1` | 466736 | 74.78% | 100.00% | 16.46% | 19.20% | 86.81% | Similar eligible-warp bottleneck; do not force vector width without changing dependency/load structure. |
| `key_prepare_forward_64_kernel_f_` | 293840 | 88.02% | 100.00% | 14.37% | 15.96% | 86.04% | High achieved occupancy but low issue utilization; supports keeping fixed warps unless kernel body changes. |
| `value_residual_gate_forward_kernel_f__n_2` | 216544 | 74.97% | 100.00% | 10.65% | 11.80% | 87.53% | Low throughput and low eligible warp rate; current line-size result is dispatch correctness, not a timing win. |
| `gated_readout_combine_forward_kernel_f_` | 415424 | 92.75% | 100.00% | 40.20% | 40.20% | 71.12% | High occupancy; prior row-pack total-regression remains the guardrail. |

Median projection-matmul counters from `ncu-lm-head-projection-matmul-gb10.csv`:

| block/grid | n | duration ns | achieved occupancy | theoretical occupancy | SM throughput | memory throughput | L2 hit | no eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `(32,12,1)` / `(16,48,1)` | 7 | 277056 | 36.30% | 50.00% | 33.71% | 49.31% | 93.26% | 71.58% |
| `(32,8,1)` / `(64,48,1)` | 1 | 714976 | 20.20% | 33.33% | 51.91% | 77.39% | 96.88% | 81.62% |
| `(32,8,1)` / `(16,48,1)` | 1 | 762496 | 20.61% | 33.33% | 48.45% | 62.07% | 93.65% | 79.78% |

Decision:

- The remaining ncu blocker is cleared for GB10: privileged `ncu` produced achieved occupancy, scheduler/warp, and memory-workload counters.
- The data supports keeping the current LayerNorm `256` on GB10: high achieved occupancy and activation passes on 253; the earlier local drift remains a hardware/numeric guardrail, so this still belongs behind keyed dispatch/autotune rather than a hard-coded global constant.
- The clearest future tuning target remains WKV7 output because it is underfilled in both theoretical and achieved occupancy, but previous row-tile/direct-value/output-factor/segment attempts are still duplicate-guarded negative results.
