# Remote Channel Mixer Edge Samples

- Date: 2026-05-16
- Branch/worktree: `kernel-tuning-remote-channel-mixer-edge-samples-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace and train-kernel changes. This attempt owns only this note and read-only analysis of remote timing artifacts; it must not edit kernels or run the local GPU.
- Prior-note search command:
  - `rg -n "cell_0000/channel_mixer|channel_mixer\\.time\\.json|edge sample|edge-row|gated_readout|warp32|ordered-256|sample" .agents/notes/kernel-tuning`
  - `rg -n "10\\.100\\.1\\.253|Projects/Packages|compare-rwkv-nn|trace-train|RWKV_TRACE_ROOT|RWKV_RS_STABLE_ROOT" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md`
- Matched prior evidence:
  - `2026-05-16-remote-channel-mixer-edge-ncu.md` records that remote `ncu` counter collection is blocked by `ERR_NVGPUCTRPERM`, then a standard remote compare found only `timing/cells/cell_0000/channel_mixer.time.json` failing. Its sample arrays were actual `[1846263, 1571780, 2117450]ns` vs baseline `[1553827, 1494197, 1956200]ns`.
  - `2026-05-16-gated-readout-forward-combine-gb10.md` records the kept 64-thread/two-warp GatedReadout combine branch with remote total speedup around `2.05x-2.14x`, but still sometimes `75/76` because of `cell_0000/channel_mixer`.
  - `2026-05-16-gated-readout-warp32-gb10.md` records warp32 rejection, revert to the 64-thread combine code, and post-revert remote compare `54/54` activation with total `2.07x`; the only failed row was `cell_0000/channel_mixer`, `1.756ms` actual vs `1.668ms` baseline.
- Changed boundary: this is not a new channel-mixer implementation or a repeat of the blocked remote `ncu` attempt. It inspects sample-level timing under the current reverted 64-thread GatedReadout remote state to determine whether the remaining `75/76` row is a statistical/outlier artifact, a baseline-median issue, or evidence for a real channel-mixer implementation branch.
- Machine/GPU: remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`; remote run directory `/home/caizus/Projects/Packages/rwkv-rs-stable` is a synchronized non-git copy.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, regenerated remote baseline under `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Kernel/stage: `cells/*/channel_mixer.time.json`, especially `cells/cell_0000/channel_mixer`.
- Command to run: read remote actual timing artifacts and regenerated baseline timing artifacts without launching local GPU; summarize `samples_ns`, medians/min/max, per-cell deltas, and whether `cell_0000` is uniquely slow after the GatedReadout revert.
- Expected decision boundary: if current samples show only one or two outliers and the minimum/median overlaps baseline, close this as variance evidence and target a larger remaining timing surface. If cell0 is consistently slower across samples while other cells are not, open a fresh channel-mixer implementation branch before editing code.

## 2026-05-16 Remote Artifact Inspection

- Host/GPU check command: `ssh -i ~/.ssh/id_ed25519_dgx_spark_windows -o BatchMode=yes -o ConnectTimeout=10 caizus@10.100.1.253 'cd /home/caizus/Projects/Packages/rwkv-rs-stable && hostname && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader && find target/rwkv-test ...'`.
- Host/GPU result: `spark-35ac`, `NVIDIA GB10`, compute capability `12.1`.
- Remote actual artifact mtime: `target/rwkv-test/rwkv_nn_actual/.../timing/cells/cell_0000/channel_mixer.time.json` at `2026-05-15 23:36:40 -0500`; `target/release/rwkv-test` at `2026-05-15 23:36:35 -0500`.
- Log caveat: visible compare logs under `target/rwkv-test/*.log` are older than the current `rwkv_nn_actual` directory. `remote-gated-readout-forward-combine-compare-rerun.log` belongs to an earlier run where `cell_0000/channel_mixer` passed and `cell_0003/time_mixer` was the only outlier. Do not mix that log with the current `rwkv_nn_actual` sample arrays.
- Current actual-vs-baseline timing reconstruction from JSON: `compared=76`, `failed=1`, `actual_total_ms=84.876`, `baseline_total_ms=175.887`, `speedup=2.072`. The only failed row is `cells/cell_0000/channel_mixer.time.json`, `1.756ms` actual vs `1.668ms` baseline, `0.95x`.
- Per-cell channel-mixer samples from current `rwkv_nn_actual`:

| cell | actual elapsed ms | baseline elapsed ms | speedup | min speedup | median speedup | actual samples ms | baseline samples ms |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `cell_0000` | 1.756 | 1.668 | 0.950 | 0.953 | 0.970 | `[1.602, 1.568, 2.099]` | `[1.554, 1.494, 1.956]` |
| `cell_0001` | 1.594 | 2.459 | 1.543 | 1.193 | 1.378 | `[1.596, 1.589, 1.596]` | `[3.282, 2.198, 1.895]` |
| `cell_0002` | 1.594 | 2.200 | 1.380 | 1.368 | 1.369 | `[1.600, 1.584, 1.599]` | `[2.166, 2.190, 2.245]` |
| `cell_0003` | 1.735 | 2.121 | 1.222 | 1.024 | 1.302 | `[1.744, 1.576, 1.886]` | `[1.614, 2.478, 2.272]` |
| `cell_0004` | 1.589 | 2.651 | 1.669 | 1.377 | 1.740 | `[1.594, 1.575, 1.597]` | `[2.169, 3.012, 2.773]` |
| `cell_0005` | 1.600 | 2.154 | 1.347 | 1.305 | 1.341 | `[1.607, 1.592, 1.599]` | `[2.240, 2.079, 2.145]` |
| `cell_0006` | 1.763 | 2.247 | 1.274 | 1.081 | 1.091 | `[1.651, 1.566, 2.072]` | `[1.694, 3.244, 1.803]` |
| `cell_0007` | 1.768 | 2.034 | 1.150 | 1.140 | 1.078 | `[1.616, 1.758, 1.931]` | `[2.364, 1.895, 1.843]` |
| `cell_0008` | 1.657 | 2.316 | 1.397 | 1.366 | 1.390 | `[1.803, 1.570, 1.599]` | `[2.145, 2.580, 2.222]` |
| `cell_0009` | 1.597 | 1.809 | 1.133 | 1.010 | 1.078 | `[1.616, 1.582, 1.593]` | `[1.717, 2.112, 1.598]` |
| `cell_0010` | 1.597 | 1.870 | 1.171 | 1.049 | 1.186 | `[1.611, 1.583, 1.596]` | `[1.892, 1.661, 2.056]` |
| `cell_0011` | 1.595 | 2.229 | 1.398 | 1.018 | 1.498 | `[1.610, 1.569, 1.604]` | `[1.597, 2.403, 2.689]` |

- Channel-mixer aggregate over all 12 cells: elapsed-sum speedup `1.298x`, min-sum speedup `1.157x`, median-sum speedup `1.283x`.
- Interpretation: current channel-mixer performance is broadly faster than the regenerated baseline. The single `cell_0000` fail is not a large surface and is dominated by a narrow first-cell/baseline comparison: actual cell0 steady samples are `1.568-1.602ms` with one `2.099ms` outlier; baseline cell0 is `1.494-1.554ms` with one `1.956ms` outlier. Later cells mostly have actual medians around `1.59-1.65ms` while baseline medians are much higher. This does not justify a channel-mixer implementation branch as the next highest-value target.
- Decision: close this branch as read-only variance/evidence. Keep the current code. The next implementation branch should target a larger remaining surface from the current timing reconstruction, with `time_mixer` and `loss/l2wrap_cross_entropy` still dominating absolute actual time. If a clean `76/76` remote gate is required, increase repeat count or use a more robust timing statistic in the harness under a separate benchmark-method branch rather than editing channel mixer for this edge row.
