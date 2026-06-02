# Channel Mixer Edge Rerun After Projection Timing

- Date: 2026-05-16
- Branch/worktree: `kernel-tuning-channel-mixer-edge-rerun-after-projection-timing-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout inherits broad unrelated workspace changes and the kept `lm_head/projection` timing-contract change. This branch is read-only except for this note.
- Prior-note/source search command:
  - `rg -n "channel_mixer|cell_0000/channel_mixer|remote-channel-mixer|0\\.99x|GB10|edge" .agents/notes/kernel-tuning /root/.codex/memories/MEMORY.md -S`
- Matched prior evidence:
  - `2026-05-16-remote-channel-mixer-edge-ncu.md` records that remote ncu counters are blocked by `ERR_NVGPUCTRPERM`, so remote counter evidence is unavailable without permission changes.
  - `2026-05-16-remote-channel-mixer-edge-samples.md` reconstructs a previous current run as `75/76`, total `2.072x`, only `cell_0000/channel_mixer` failed, and concludes this is a narrow edge row rather than broad channel-mixer regression.
  - `2026-05-16-lm-head-projection-timing-contract.md` changed the current binary/timing boundary and the first remote compare after that change had `ignored=1`, total `2.06x`, and only `cell_0000/channel_mixer` failed by `0.017ms`.
- Changed boundary: this rerun uses the post-`lm_head/projection` timing-contract binary and existing remote warmed build/cache. It is not a channel-mixer implementation change.
- Machine/GPU: remote `caizus@10.100.1.253`, NVIDIA GB10, compute capability `12.1`; local GPU must not be used.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, regenerated remote baseline under `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Kernel/stage: `cells/cell_0000/channel_mixer` timing row, plus aggregate `cells/*/channel_mixer`.
- Candidate parameters: none.
- Hypothesis: the single `0.99x` row after the projection timing-contract change is warm-run timing jitter, because aggregate channel mixer remains faster and earlier warm runs sometimes produced clean `76/76`.
- Expected keep/revert boundary: keep no code changes. If the rerun passes `76/76`, record remote clean gate for the current binary. If it fails again only on the same row with total speedup above `1.0`, do not edit channel_mixer; open a separate benchmark-method/baseline-statistics branch if a clean gate is required.
- Next command: run standard remote compare on `10.100.1.253` with the existing regenerated baseline and current post-projection-timing binary.

## 2026-05-16 Remote Rerun

- Command: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline ~/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1`.
- Binary provenance: existing remote release binary was already up to date for the post-`lm_head/projection` timing-contract source; Cargo finished in `0.19s`.
- Correctness result: `activation_summary compared=54 passed=54 failed=0`.
- Timing result: clean remote gate, `timing_summary compared=76 passed=76 failed=0 missing=0 extra=0 ignored=1 actual_total_ms=83.677 baseline_total_ms=175.887 speedup=2.10x`.
- Channel-mixer evidence:
  - `timing/cells/cell_0000/channel_mixer.time.json` passed this run: `1.599ms` actual vs `1.668ms` baseline, `1.04x`.
  - Aggregate `cells/*/channel_mixer` stayed clearly faster: `19.769ms` actual vs `25.759ms` baseline, `1.30x`.
- Interpretation: the previous `cell_0000/channel_mixer` failure after projection timing was warm-run timing jitter, not an implementation regression. Do not edit channel mixer for this edge row.
- Decision: keep no code changes in this branch. The current remote GB10 post-projection-timing binary has a clean `76/76` timing gate and total speedup above `1.0`.
