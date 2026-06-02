# LM Head Projection Baseline Contract

- Date: 2026-05-16 18:24 +0800.
- Stable branch/worktree: `kernel-tuning-remote-current-after-lowrank-gb10-20260516` in `/mnt/g/Projects/Packages/rwkv-rs-stable`; this note is stored here with the kernel-tuning ledger.
- Companion branch/worktree: `kernel-tuning-lm-head-projection-baseline-20260516` in `/mnt/g/Projects/Packages/rwkv-rs-test`.
- Dirty-tree constraint:
  - `rwkv-rs-stable` carries broad unrelated workspace edits and kernel notes; this attempt owns only this note unless explicitly expanded.
  - `rwkv-rs-test` has pre-existing untracked `test_gen_local_regen_20260516/`; this attempt must not edit or delete it.
- User constraint: debug first on remote `10.100.1.253`; do not use the local GPU.
- Prior evidence:
  - Rust actual trace now emits optional `timing/lm_head/projection.time.json` without writing logits.
  - Current remote compare is clean, but `ignored=1` because the regenerated Python baseline lacks `lm_head/projection`.
  - Current remote nsys ranks the lm-head projection Cubek/TMA matmul as the largest single surface: about `46.376ms / 3` launches.
  - Projection+loss fusion is too broad for a quick kernel branch; the safer next step is to make the baseline timing contract see the projection boundary.
- Companion source inspection:
  - `train-repo/rwkv-lm/src/model.py::_forward_features` traces final LayerNorm as `lm_head`, then returns hidden.
  - In the non-head-chunk training path, `forward` currently returns `self.head(x)` without trace timing.
  - Static tests already assert no `lm_head/logits.safetensors` trace output; this branch must keep that constraint.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, vocab `65536`.
- Changed boundary: add timing-only trace for the Python baseline `self.head(x)` as module `lm_head/projection`, without exporting `lm_head/logits.safetensors`.
- Expected keep/revert boundary:
  - Keep only if companion static tests pass, remote baseline regeneration emits `timing/lm_head/projection.time.json`, and stable remote compare compares rather than ignores that row when using the regenerated baseline.
  - Revert if logits are exported, old activation contract changes, or compare treats projection as a missing/extra failure.
- Next edit: update `/mnt/g/Projects/Packages/rwkv-rs-test/train-repo/rwkv-lm/src/model.py` and its static test to wrap only `self.head(x)` in `trace("lm_head/projection", ..., outputs={})` or equivalent timing-only form.

## Implementation Edit

- Edited `/mnt/g/Projects/Packages/rwkv-rs-test/train-repo/rwkv-lm/src/model.py`.
  - Non-head-chunk `forward` now returns `trace("lm_head/projection", lambda: self.head(x))`.
  - `outputs` is omitted intentionally, so the helper writes only `timing/lm_head/projection.time.json` and does not write `lm_head/logits.safetensors`.
- Edited `/mnt/g/Projects/Packages/rwkv-rs-test/train-repo/rwkv-lm/tests/test_static_contract.py`.
  - Added a static assertion for the projection timing trace call.
- Next command: run companion static tests for `rwkv-lm`; no local GPU timing.

## Local Static Test Attempt 1

- Command: `python -m pytest train-repo/rwkv-lm/tests/test_static_contract.py` from `/mnt/g/Projects/Packages/rwkv-rs-test`.
- Result: invalid environment command, `python` is not on PATH in this shell.
- Retry command: `python3 -m pytest train-repo/rwkv-lm/tests/test_static_contract.py`.
- Retry result: invalid environment command, `pytest` is not installed for `/usr/bin/python3`.
- Next command: run the test file through stdlib `unittest`, since it is a `unittest.TestCase` test.

## Local Static Test

- Command: `python3 -m unittest train-repo/rwkv-lm/tests/test_static_contract.py` from `/mnt/g/Projects/Packages/rwkv-rs-test`.
- Result: passed, `Ran 6 tests`.
- Diff check: only `train-repo/rwkv-lm/src/model.py` and `train-repo/rwkv-lm/tests/test_static_contract.py` changed in the companion repo. Git warned that `model.py` has CRLF normalization on future touch; the visible diff is the intended one-line trace wrapper.
- Next command: sync the two companion files to `/home/caizus/Projects/Packages/rwkv-rs-test` on `10.100.1.253`, then run the same static test remotely before regenerating the baseline.

## Remote Sync Attempt 1

- Command: scoped `rsync -avR` of companion `model.py` and `test_static_contract.py` to `/home/caizus/Projects/Packages/rwkv-rs-test`.
- Result: companion file sync succeeded.
- Note sync in the same shell command failed with rsync code `23` because the command was run from the companion repo while the note files live in `/mnt/g/Projects/Packages/rwkv-rs-stable/.agents/notes/...`.
- Interpretation: remote companion source is updated; only note sync failed.
- Next command: sync the notes from the stable repo path, then run remote static tests.

## Remote Static Test

- Note sync retry: synced this note and `2026-05-16-remote-current-after-lowrank-gb10.md` from the stable repo path to `/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Command: `python3 -m unittest train-repo/rwkv-lm/tests/test_static_contract.py` from `/home/caizus/Projects/Packages/rwkv-rs-test` on `10.100.1.253`.
- Result: passed, `Ran 6 tests`.
- Next command: regenerate a separate remote baseline under `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_20260516` using `trace-train.sh`, then confirm `timing/lm_head/projection.time.json` exists before using it for compare.

## Remote Baseline Regeneration Attempt 1

- Command: ran `trace-train.sh` with `RWKV_RS_STABLE_ROOT=/home/caizus/Projects/Packages/rwkv-rs-stable`, `RWKV_TRACE_ROOT=/home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_20260516`, `RWKV_TRACE_WARMUP=1`, and `RWKV_TRACE_REPEAT=3`.
- Result: invalid environment command, `uv: command not found`.
- Interpretation: remote login PATH does not include the `uv` binary used in earlier baseline regeneration.
- Next command: locate `uv` on `10.100.1.253` and rerun with a command-scoped PATH if available.

## Remote Baseline Regeneration

- `uv` lookup result: `/home/caizus/.venvs/uv/bin/uv`.
- Command: reran `trace-train.sh` with command-scoped `PATH=/home/caizus/.venvs/uv/bin:$PATH`, `RWKV_RS_STABLE_ROOT=/home/caizus/Projects/Packages/rwkv-rs-stable`, `RWKV_TRACE_ROOT=/home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_20260516`, `RWKV_TRACE_WARMUP=1`, and `RWKV_TRACE_REPEAT=3`.
- Result: completed successfully after compiling/loading the Python CUDA extensions and running one trace training step.
- Projection timing artifact exists:
  - Path: `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_20260516/rwkv_lm/bf16/case_000000/timing/lm_head/projection.time.json`.
  - Content summary: `elapsed_ns=32149468`, `repeat=3`, `warmup=1`, samples `[41981113, 42634622, 11832670]`.
- Interpretation: the companion baseline now emits the same `lm_head/projection` timing row as the Rust actual path, without exporting `lm_head/logits.safetensors`.
- Next command: run stable remote `compare-rwkv-nn` against the regenerated projection baseline and confirm the projection row is compared instead of ignored.

## Remote Compare With Projection Baseline

- Command: `target/release/rwkv-test compare-rwkv-nn --color never --baseline /home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_20260516/rwkv_lm/bf16/case_000000 --repeat 3 --warmup 1`, captured in `target/rwkv-test/remote-projection-baseline-compare.log`.
- Activation result: passed, `activation_summary compared=54 passed=54 failed=0`.
- Timing contract result:
  - `timing/lm_head/projection.time.json` is now compared, not ignored.
  - `timing_scope=canonical_compute ignored=0`.
  - `timing_summary compared=77 passed=73 failed=4 missing=0 extra=0 ignored=0 actual_total_ms=90.616 baseline_total_ms=203.247 speedup=2.24x`.
- Projection row result: `actual_ms=15.624`, `baseline_ms=32.149`, speedup `2.06x`.
- Timing caveat: the four failed rows are all `cells/*/channel_mixer.time.json` edge rows. This is not caused by the projection timing contract; it reflects the newly regenerated Python baseline being faster/noisier for those rows than the older remote baseline.
- Decision: keep the companion `lm_head/projection` timing change. It makes the largest lm-head projection matmul visible to compare without exporting logits and without missing/extra timing failures.
- Keep/revert state: keep companion files `train-repo/rwkv-lm/src/model.py` and `train-repo/rwkv-lm/tests/test_static_contract.py` on branch `kernel-tuning-lm-head-projection-baseline-20260516`.
