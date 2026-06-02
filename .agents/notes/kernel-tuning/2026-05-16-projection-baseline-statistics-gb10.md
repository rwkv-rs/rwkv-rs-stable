# Projection Baseline Statistics GB10

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-projection-baseline-statistics-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits plus retained/reverted kernel tuning changes. This attempt owns only this note and remote measurement-contract evidence unless a later entry explicitly opens a code or baseline-generator edit.
- User constraint: debug first on remote `10.100.1.253`; do not use the local GPU because the user is running other work there.
- Prior-note and memory search commands:
  - `rg -n "lm_head/projection|projection baseline|channel_mixer|edge|timing_summary|repeat|warmup|baseline statistics|10\\.100\\.1\\.253" .agents/notes/kernel-tuning /root/.codex/memories/MEMORY.md -S`
  - `rg -n "WKV7|shared-lanes|segment|low-rank|row_tile|projection\\+loss|ncu|ERR_NVGPUCTRPERM" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md -S`
- Matched prior evidence:
  - The old regenerated remote baseline has a clean current gate after projection timing is emitted one-sided: activation `54/54`, timing `76/76`, `ignored=1`, speedup `2.10x`.
  - The new companion projection baseline emits `timing/lm_head/projection.time.json`, making compare cover `77` rows and `ignored=0`, but one run failed four `cells/*/channel_mixer` rows while total speedup stayed `2.24x`.
  - Channel-mixer implementation directions are duplicate-closed: forced Cube, Burn reference/fusion, LocalTuner bypass, and line-size retry are recorded negatives or non-actions.
  - WKV7 row-tile, shared-lanes, dense segment recompute, and low-rank segment designs are closed. Projection+loss fusion is a broad new operator family and not a small timing-statistics fix.
  - Remote `ncu` counter collection is blocked by `ERR_NVGPUCTRPERM`; use file timing samples, standard compare, nsys metadata, and source evidence on this host.
- Machine/GPU: remote `caizus@10.100.1.253` via `ssh -i ~/.ssh/id_ed25519_dgx_spark_windows`, host expected `spark-35ac`, `NVIDIA GB10`, compute capability `12.1`.
- Stable remote path: `/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Companion baseline path under test: `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_20260516/rwkv_lm/bf16/case_000000`.
- Old baseline path: `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000`.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, heads `12`, head size `64`, vocab `65536`.
- Kernel/stage: measurement contract around `lm_head/projection` and per-cell `channel_mixer` timing rows.
- Hypothesis: the four failures against the new projection baseline are baseline/statistical variance in short channel-mixer module timings, not a kernel regression. Before changing kernels, inspect actual/baseline timing sample arrays and rerun remote compare against the same projection baseline with the warmed binary/cache.
- Candidate parameters: none. This is a benchmark-method evidence branch.
- Commands to run:
  1. Remote preflight and read actual/baseline timing samples for `timing/cells/cell_*/channel_mixer.time.json` plus `timing/lm_head/projection.time.json`.
  2. Remote standard compare against the projection baseline with `--repeat 3 --warmup 1`, captured in `target/rwkv-test/remote-projection-baseline-rerun-compare.log`.
  3. If failures persist only on short channel-mixer rows, consider a separate baseline-statistics policy branch or regenerate a higher-repeat projection baseline; do not edit channel-mixer kernels in this branch.
- Expected keep/revert boundary: keep as measurement evidence. If rerun passes, classify the earlier four failures as timing noise. If rerun fails on different short rows but aggregate modules stay faster and total speedup stays `>1.0`, avoid kernel edits and move to method/policy. If activation fails or a broad module regresses, stop and inspect source/binary provenance before any tuning.

## Remote Sample Inspection Attempt 1

- Command: nested `ssh ... python3 <<'PY'` sample extraction for projection baseline and current actual timing rows.
- Result: invalid command. Local/remote shell quoting corrupted Python f-strings and string literals, producing a remote Python `SyntaxError`. No timing data was collected and no conclusion is drawn.
- Next command: pipe a locally quoted Python script into `ssh ... 'cd ... && python3 -'` so the remote shell does not reinterpret the script body.

## Remote Sample Inspection

- Command: piped local Python source into remote `python3 -` from `/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Result: source and baseline paths exist.
- Channel-mixer sample evidence:
  - Failing rows from the previous projection-baseline compare line up with short timing variance, not broad module drift. Examples:
    - `cell_0000`: actual `1.984ms`, samples `[2.050, 1.625, 2.277]ms`; baseline `1.712ms`, samples `[1.660, 1.621, 1.856]ms`.
    - `cell_0003`: actual `1.907ms`, samples `[2.042, 1.716, 1.963]ms`; baseline `1.687ms`, samples `[1.836, 1.518, 1.706]ms`.
    - `cell_0008`: actual `1.603ms`, samples `[1.626, 1.606, 1.577]ms`; baseline `1.594ms`, samples `[1.626, 1.509, 1.648]ms`, effectively a noise-edge fail.
  - Passing rows show the same variance family in the other direction; for example `cell_0007` actual `1.605ms` vs baseline `2.190ms`.
- Projection row evidence:
  - Actual `lm_head/projection`: `15.624ms`, samples `[15.152, 15.157, 16.564]ms`.
  - Baseline `lm_head/projection`: `32.149ms`, samples `[41.981, 42.635, 11.833]ms`.
  - Projection is compared and passes despite baseline sample skew; the failed rows are not projection-specific.
- Interpretation: the projection baseline is contract-correct, but `repeat=3` per-row channel-mixer timing is too noisy to make every short row a stable pass/fail oracle. Do not edit channel-mixer kernels from this evidence.
- Next command: rerun standard remote compare against the projection baseline with the same `repeat=3,warmup=1`, capturing `target/rwkv-test/remote-projection-baseline-rerun-compare.log`.

## Remote Projection Baseline Rerun

- Command: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_20260516/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1`, captured in `target/rwkv-test/remote-projection-baseline-rerun-compare.log`.
- Binary provenance: existing remote release binary was up to date; Cargo finished in `0.19s`.
- Correctness result: activation passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: `timing_summary compared=77 passed=74 failed=3 missing=0 extra=0 ignored=0 actual_total_ms=91.072 baseline_total_ms=203.247 speedup=2.23x`.
- Projection row: passed, `actual_ms=16.103`, `baseline_ms=32.149`, speedup `2.00x`.
- Failure pattern:
  - `cell_0003/channel_mixer`: `1.991ms` actual vs `1.687ms` baseline.
  - `cell_0008/channel_mixer`: `1.616ms` actual vs `1.594ms` baseline, a `0.022ms` edge.
  - `cell_0011/pre_layer_norm_for_channel_mix`: `0.207ms` actual vs `0.171ms` baseline, a very short timing row.
- Interpretation: failed row set changed and includes a sub-0.25ms row while module totals remain positive (`cells/*/channel_mixer` `1.09x`, `cells/*/time_mixer` `2.76x`, total `2.23x`). This confirms method/statistical noise more than kernel regression.
- Next command: generate a higher-repeat projection baseline under `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_r9_20260516` with `RWKV_TRACE_WARMUP=3` and `RWKV_TRACE_REPEAT=9`, then compare actual with `--warmup 3 --repeat 9`. This is a measurement-method check, not a kernel implementation change.

## Remote R9 Baseline Generation Attempt 1

- Command: from `/home/caizus/Projects/Packages/rwkv-rs-test`, ran `PATH=/home/caizus/.venvs/uv/bin:$PATH RWKV_RS_STABLE_ROOT=... RWKV_TRACE_ROOT=... RWKV_TRACE_WARMUP=3 RWKV_TRACE_REPEAT=9 bash trace-train.sh`.
- Result: invalid command, `bash: trace-train.sh: No such file or directory`.
- Interpretation: the script is not at the companion repo root. No baseline data was generated.
- Next command: locate the trace script under the remote companion repo, then rerun from the correct directory.

## Remote Trace Script Location

- Command: `find . -maxdepth 4 -name trace-train.sh -o -name "*trace*train*"` from `/home/caizus/Projects/Packages/rwkv-rs-test`.
- Result: target script is `train-repo/rwkv-lm/trace-train.sh`.
- Next command: rerun R9 baseline generation from `/home/caizus/Projects/Packages/rwkv-rs-test/train-repo/rwkv-lm` with the same `RWKV_TRACE_ROOT`, `RWKV_TRACE_WARMUP=3`, and `RWKV_TRACE_REPEAT=9`.

## Remote R9 Baseline Generation

- Command: from `/home/caizus/Projects/Packages/rwkv-rs-test/train-repo/rwkv-lm`, ran `PATH=/home/caizus/.venvs/uv/bin:$PATH RWKV_RS_STABLE_ROOT=/home/caizus/Projects/Packages/rwkv-rs-stable RWKV_TRACE_ROOT=/home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_r9_20260516 RWKV_TRACE_WARMUP=3 RWKV_TRACE_REPEAT=9 bash trace-train.sh`, captured in `target/rwkv-test/remote-projection-r9-baseline-gen.log`.
- Result: completed successfully. The Python trace run loaded the L12 D768 BF16 model and executed one training step.
- Next command: confirm `timing/lm_head/projection.time.json` exists in the R9 baseline, then run Rust compare against it with `--warmup 3 --repeat 9`.

## Remote R9 Artifact Check Attempt 1

- Command: nested remote Python heredoc to read R9 baseline timing JSON files.
- Result: invalid command. Shell quoting stripped string quotes from the Python script, causing `SyntaxError`. No artifact conclusion is drawn.
- Next command: rerun the artifact check by piping the Python script into remote `python3 -`.

## Remote R9 Artifact Check

- Command: piped Python into remote `python3 -` and read the R9 baseline timing JSON files.
- Result: R9 baseline artifacts exist.
- R9 baseline samples:
  - `lm_head/projection`: elapsed `15.158ms`, samples `[12.171, 41.904, 12.536, 11.666, 11.667, 11.789, 11.972, 11.152, 11.561]ms`.
  - `cell_0003/channel_mixer`: elapsed `2.020ms`, samples `[2.447, 2.239, 1.667, 1.486, 1.494, 2.170, 2.109, 2.268, 2.298]ms`.
  - `cell_0008/channel_mixer`: elapsed `1.837ms`, samples `[2.471, 1.886, 1.587, 1.624, 1.856, 1.836, 1.839, 1.715, 1.722]ms`.
  - `cell_0011/pre_layer_norm_for_channel_mix`: elapsed `0.223ms`, samples `[0.257, 0.195, 0.257, 0.258, 0.255, 0.155, 0.238, 0.150, 0.238]ms`.
- Interpretation: higher repeat changes the baseline materially. In particular, the projection baseline is now about `15.16ms` instead of the R3 baseline `32.15ms`, proving the original R3 projection row had slow outlier skew. The R9 compare may reveal the Rust projection is near parity or slower rather than `2.0x` faster.
- Next command: run remote Rust compare against the R9 projection baseline with `--warmup 3 --repeat 9`, captured in `target/rwkv-test/remote-projection-r9-compare.log`.

## Remote R9 Compare

- Command: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_r9_20260516/rwkv_lm/bf16/case_000000 --repeat 9 --warmup 3`, captured in `target/rwkv-test/remote-projection-r9-compare.log`.
- Binary provenance: existing remote release binary was up to date; Cargo finished in `0.16s`.
- Correctness result: activation passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: `timing_summary compared=77 passed=76 failed=1 missing=0 extra=0 ignored=0 actual_total_ms=87.891 baseline_total_ms=130.484 speedup=1.48x`.
- All short channel-mixer and pre-LayerNorm rows passed under R9:
  - `cells/*/channel_mixer`: `20.100ms` actual vs `23.444ms` baseline, speedup `1.17x`.
  - `cells/*/time_mixer`: `42.462ms` actual vs `71.108ms` baseline, speedup `1.67x`.
- Single failed row: `timing/lm_head/projection.time.json`, `15.242ms` actual vs `15.158ms` baseline, speedup `0.99x`, delta `0.084ms`.
- Interpretation: R9 resolves the short-row noise and shows the projection row is near parity, not `2.0x` faster as the noisy R3 baseline implied. This is not channel-mixer regression evidence. The remaining question is whether projection is stable slightly slower than the PyTorch baseline or a near-threshold noise flip.
- Next command: rerun the same R9 compare once more, captured in `target/rwkv-test/remote-projection-r9-rerun-compare.log`, before deciding whether to open a broad Cubek/TMA projection branch or keep this as measurement-method evidence.

## Remote R9 Compare Rerun

- Command: repeated `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_r9_20260516/rwkv_lm/bf16/case_000000 --repeat 9 --warmup 3`, captured in `target/rwkv-test/remote-projection-r9-rerun-compare.log`.
- Correctness result: activation passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing result: `timing_summary compared=77 passed=76 failed=1 missing=0 extra=0 ignored=0 actual_total_ms=88.202 baseline_total_ms=130.484 speedup=1.48x`.
- Single failed row reproduced: `timing/lm_head/projection.time.json`, `15.241ms` actual vs `15.158ms` baseline, speedup `0.99x`, delta `0.083ms`.
- Stable module results:
  - `cells/*/channel_mixer`: `19.958ms` actual vs `23.444ms` baseline, `1.17x`.
  - `cells/*/time_mixer`: `42.294ms` actual vs `71.108ms` baseline, `1.68x`.
  - `loss/l2wrap_cross_entropy`: `4.337ms` actual vs `5.979ms` baseline, `1.38x`.
- Decision:
  - Keep the projection timing contract; it correctly exposes the largest lm-head matmul boundary.
  - Do not claim projection `2.0x` speedup from the R3 baseline. The R9 baseline shows Rust/Burn projection is near parity and slightly slower than the Python/PyTorch baseline on this GB10 run.
  - Do not edit channel_mixer. The earlier R3 channel-mixer failures were measurement noise and disappear under R9.
  - Do not open a small kernel branch for projection; matching or beating PyTorch/Cubek projection would require broad matmul/operator work, not a scoped rwkv-nn row-loss tweak.
- Next edit: update `.agents/skills/kernel-tuning/SKILL.md` with this GB10 measurement guard so future runs do not repeat the R3 projection-baseline overclaim or channel-mixer edge-row detour.

## Skill Guard Update

- Edited `.agents/skills/kernel-tuning/SKILL.md`.
- Added known GB10 measurement guard:
  - keep `lm_head/projection` visible in timing;
  - do not claim projection `2x` from the noisy `repeat=3,warmup=1` baseline;
  - use the R9 evidence that projection is near parity/slightly slower (`15.24ms` actual vs `15.16ms` baseline) while total speedup stays `1.48x`;
  - do not edit channel_mixer for R3 projection-baseline edge-row failures.
- Keep/revert state: keep this skill update as duplicate-experiment and overclaim prevention.
- Next command: sync this note and the updated skill to `/home/caizus/Projects/Packages/rwkv-rs-stable` on `10.100.1.253`.

## Remote Note And Skill Sync

- Command: scoped `rsync -azR` of `.agents/skills/kernel-tuning/SKILL.md` and this note to `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable/`.
- Result: sync completed successfully.
- Final decision for this branch: keep the note and skill update. No kernel code changed in this branch.
