# Projection Matmul Autotune Audit GB10

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-projection-matmul-autotune-audit-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits plus retained/reverted kernel tuning changes. This attempt owns only this audit note unless a later entry explicitly records a source or cache-policy edit.
- User constraint: debug first on remote `10.100.1.253`; do not use local GPU timing.
- Prior-note and memory search commands already run before this note:
  - `rg -n "lm_head/projection|projection matmul|matmul_entry|Cubek|TMA|projection baseline|R9|0\\.98x|0\\.99x|autotune|cache|selected" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md /root/.codex/memories/MEMORY.md -S`
  - `rg -n "matmul|projection\\+loss|TMA|Cubek|FusedMatmul|epilogue|GlobalWriter" .agents/notes/kernel-tuning crates/rwkv-nn/src crates/rwkv-test/src -S`
- Matched prior evidence:
  - `2026-05-16-projection-baseline-statistics-gb10.md` shows the R9 projection baseline makes `lm_head/projection` canonical and stable: activation passes, total speedup stays about `1.48x`, but projection itself is near parity/slightly slow (`15.24ms` actual vs `15.16ms` baseline, about `0.99x`).
  - `2026-05-16-lm-head-projection-loss-design.md` rejects a naive project-local projection+loss fusion. Burn `Linear` lowers to Cubek/TMA matmul and the public path does not expose a TMA-quality epilogue hook.
  - `2026-05-16-current-goal-audit-after-r9.md` and `2026-05-16-completion-audit-current.md` keep the goal open because local row-level acceptance and remote counter-level `ncu` evidence remain incomplete.
  - Remote `ncu` counters are blocked by `RmProfilingAdminOnly: 1`, so this branch can inspect autotune logs, compare logs, and existing `nsys` metadata, but should not rerun `ncu` as normal `caizus`.
- Machine/GPU: remote `caizus@10.100.1.253`, host `spark-35ac`, `NVIDIA GB10`, compute capability `12.1`.
- Remote run directory: `/home/caizus/Projects/Packages/rwkv-rs-stable`, synchronized non-git mirror.
- Baseline path: `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_r9_20260516/rwkv_lm/bf16/case_000000`.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, vocab `65536`.
- Kernel/stage: `lm_head/projection`, the unembed `Linear` projection backed by Cubek/TMA BF16 matmul.
- Hypothesis: the projection row's `0.98-0.99x` result may be a stable Cubek-vs-PyTorch boundary, or it may be affected by stale/poor matmul autotune cache selection. Before opening a broad matmul implementation branch, inspect remote matmul autotune logs and profiler metadata to confirm the selected candidate and cache provenance.
- Candidate parameters: none in this audit. Do not clear autotune cache or force matmul candidates unless a later note records a stale-cache or wrong-candidate hypothesis.
- Expected keep/revert boundary: keep this note as measurement evidence. If the logs show stale or mismatched matmul selection, open a fresh cache/candidate-validation branch before changing cache or source. If the selection is current TMA and the row is still near parity, keep the projection-matmul boundary classified as broad upstream/Cubek work rather than a small `rwkv-nn` kernel fix.
- Next command: inspect remote `target/autotune` log inventory and find entries whose tunable/key names correspond to Cubek/Burn matmul or the `lm_head/projection` shape, without modifying cache files.

## Remote Autotune Inventory

- Command: remote `find target/autotune -type f -maxdepth 3 ...` on `10.100.1.253`.
- Result: remote host is `spark-35ac`. The only generic Burn/CubeCL matmul autotune file is `target/autotune/0.10.0/device-0-0-cuda/burn_cubecl-kernel-matmul-tune-base.json.log`, mtime `2026-05-15 23:15`. Project custom-kernel logs are newer and include the current forward custom tuners.
- Interpretation: the projection matmul is likely governed by the generic Burn/CubeCL matmul cache file, not a project `rwkv-nn` tuner.
- Next command: inspect the generic matmul autotune log structure and entries related to BF16 `m=8192,n=65536,k=768` or the launch geometry seen in `nsys` (`grid=(4096,16,1)`, `block=(32,12,1)`), without editing cache files.

## Matmul Log Structure

- Command: remote Python read of `burn_cubecl-kernel-matmul-tune-base.json.log`.
- Result:
  - File exists, size `62758` bytes, `10` JSON-lines entries.
  - The last entry is the only vocab-scale one: key definition `m=512,n=65536,k=1024`, BF16 lhs/rhs/out, `analysis.scale_global=Large`, `kind=General`.
  - That entry selected `fastest_index=28`, candidate `matmul_specialized_tma_mma`, with recorded mean around `15.566ms` and median around `15.701ms`.
- Interpretation:
  - The generic matmul cache does contain a large-vocab BF16 entry matching the projection timing scale and selected a TMA candidate.
  - The key shape is bucketed/rounded (`m=512,k=1024`) rather than literal `[8192,65536,768]`, so the next step is to print all candidate timings for that entry and inspect whether the selected TMA candidate is clearly fastest or near-tied.
- Next command: parse the vocab-scale matmul log entry and print candidate names, indices, and mean/median/min/max timings in sorted order.

## Vocab-Scale Matmul Candidate Ranking

- Command: remote Python JSON-lines parser for the `n=65536` BF16 matmul entry.
- Result:
  - Entry key: `m=512,n=65536,k=1024`, BF16 lhs/rhs/out, contiguous layouts, `scale_global=Large`, checksum `e69471b2de574e1c09587cf0d6ba33d1`.
  - Selected `fastest_index=28`.
  - Ranked candidate timings:
    - `idx=28 matmul_specialized_tma_mma`: mean `15.566ms`, median `15.701ms`, min `15.046ms`, max `16.089ms`.
    - `idx=24 matmul_simple_tma_mma`: mean `16.288ms`, median `16.326ms`, min `15.924ms`, max `16.918ms`.
    - `idx=25 matmul_simple_tma_cmma_multi_rows`: mean `19.371ms`.
    - cyclic/ordered candidates are `23ms+`; other TMA/CMMA candidates are much slower.
  - Several non-applicable candidates were skipped (`matmul_naive`, vecmat/unit variants).
- Interpretation:
  - The cache did not select an obviously bad candidate. `matmul_specialized_tma_mma` is the clear winner in the existing autotune sample and lines up with the R9 projection timing scale.
  - The R9 projection row being `0.98-0.99x` is therefore unlikely to be fixed by just clearing the cache or forcing another existing generic matmul candidate.
- Next command: inspect existing `nsys` sqlite metadata around the projection matmul to confirm current launches use the same TMA-scale geometry and to compare profiler timing with the autotune log and R9 `.time.json` row.

## Existing Nsys Projection Metadata

- Command: remote Python sqlite query over `target/rwkv-test/nsys-current-after-gb10-results.sqlite`; no new profile run.
- Result:
  - Each `lm_head_l2wrap_ce_forward_row_kernel` is immediately preceded by final LayerNorm and then a large BF16 matmul.
  - Projection-like matmul launches:
    - `15.343ms`, `grid=(4096,16,1)`, `block=(32,12,1)`, `regs=73`, dynamic shared memory `27648`.
    - `14.983ms`, same launch geometry.
    - `15.919ms`, same launch geometry.
  - Projection-like total: `46.245ms / 3`, average `15.415ms`.
  - The following row-loss launches are about `4.31-4.73ms`, using `block=(1024,1,1)`, `regs=40`, dynamic shared memory `128`.
- Interpretation:
  - The nsys projection times match the generic matmul autotune winner and the R9 `.time.json` actual row (`~15.2-15.4ms`).
  - The launch geometry is the known optimized TMA matmul path, not a fallback or skipped non-TMA candidate.
  - This confirms the projection row is not currently slow because of a stale selected generic matmul candidate.
- Next command: inspect whether the current remote R9 compare binary/log is newer than the matmul autotune cache and whether a cache retune would be expected to change anything. Do not delete cache.

## Binary And Log Provenance Check

- Command: remote `stat` for generic matmul autotune log, `target/release/rwkv-test`, and R9 compare logs, plus grep of the last R9 compare result.
- Result:
  - Matmul autotune cache mtime: `2026-05-15 23:15:18 -0500`.
  - Current remote `target/release/rwkv-test` mtime: `2026-05-16 08:09:24 -0500`.
  - Existing R9 projection rerun log mtime: `2026-05-16 05:34:56 -0500`.
  - Existing backward-key R9 compare log mtime: `2026-05-16 06:46:30 -0500`.
  - Existing backward-key R9 result: activation passed `54/54`; timing `77` compared, `76` passed, one failed `lm_head/projection` row at `15.433ms` actual vs `15.158ms` baseline, total speedup `1.47x`.
- Interpretation:
  - The existing R9 timing logs are useful historical evidence, but they are older than the current remote release binary. A fresh remote R9 compare is warranted before treating this projection matmul audit as current-binary evidence.
  - Do not clear the matmul autotune cache for this run; the hypothesis under test is current-binary behavior with the existing selected TMA candidate.
- Next command: run one fresh remote R9 compare against `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_r9_20260516/rwkv_lm/bf16/case_000000`, capture it in `target/rwkv-test/remote-projection-matmul-autotune-audit-r9.log`, then extract activation/timing/projection rows.

## Fresh Current-Binary R9 Compare

- Command:
  - Intended compare: remote `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --repeat 9 --warmup 3 --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_r9_20260516/rwkv_lm/bf16/case_000000`, captured in `target/rwkv-test/remote-projection-matmul-autotune-audit-r9.log`.
  - Shell wrapper caveat: the command attempted to assign `status=$?` under remote zsh, where `status` is read-only. The shell wrapper exited with a zsh error after the compare log was written, so the wrapper exit code is invalid. The log itself is complete and was extracted separately.
- Log artifact:
  - Path: `target/rwkv-test/remote-projection-matmul-autotune-audit-r9.log`.
  - Mtime/size: `2026-05-16 08:33:25 -0500`, `21900` bytes.
- Result:
  - Activation passed: `activation_summary compared=54 passed=54 failed=0`.
  - Timing passed cleanly: `timing_summary compared=77 passed=77 failed=0 missing=0 extra=0 ignored=0 actual_total_ms=86.666 baseline_total_ms=130.484 speedup=1.51x`.
  - `lm_head/projection` now passes: `15.035ms` actual vs `15.158ms` baseline, `1.01x`.
  - Module totals: `cells/*/time_mixer` `1.70x`, `cells/*/channel_mixer` `1.19x`, `loss/l2wrap_cross_entropy` `1.35x`.
- Interpretation:
  - Current remote binary plus the existing matmul cache is clean against the R9 projection baseline. The earlier `0.98-0.99x` projection result was a near-threshold flip, not proof of a stale or bad matmul candidate.
  - The generic matmul autotune selection is credible: `matmul_specialized_tma_mma` is fastest in the cache, nsys shows the expected TMA launch geometry, and the current R9 projection row passes without cache changes.
  - Do not open a cache-clearing or candidate-forcing branch from this evidence.
- Next edit: update the kernel-tuning skill's projection measurement guard from "near parity/slightly slower" to "near parity; R9 has seen both 0.98x and 1.01x, so do not overclaim either direction and do not edit channel_mixer/projection from R3 noise."

## Skill Guard Update

- Edited `.agents/skills/kernel-tuning/SKILL.md`.
- Updated the known GB10 `lm_head/projection` measurement guard:
  - R3 projection baseline remains too noisy for per-row claims.
  - R9 projection should be treated as near parity, with historical `0.99x` and current-binary `1.01x` evidence.
  - Do not open cache-clearing or candidate-forcing branches unless fresh evidence shows the selected Cubek/TMA matmul candidate is stale or wrong.
- Next command: run `git diff --check` for the touched note and skill, then sync both to the remote mirror.

## Validation And Sync

- Command: `git diff --check -- .agents/notes/kernel-tuning/2026-05-16-projection-matmul-autotune-audit-gb10.md .agents/skills/kernel-tuning/SKILL.md`.
- Result: passed.
- Remote sync command: scoped `rsync -azR` of this note and `.agents/skills/kernel-tuning/SKILL.md` to `/home/caizus/Projects/Packages/rwkv-rs-stable` on `10.100.1.253`.
- Result: passed.
- Decision:
  - Keep this audit note and skill update.
  - Do not clear the generic matmul cache or force a different existing Cubek matmul candidate. The cache selects `matmul_specialized_tma_mma`, existing nsys geometry matches the TMA path, and the fresh current-binary R9 compare passes all `77/77` timing rows with total speedup `1.51x`.
  - `lm_head/projection` should remain classified as near parity and visible in timing, not as a current small-kernel regression.
