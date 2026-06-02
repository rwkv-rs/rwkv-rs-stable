# LayerNorm Hardware Fingerprint Helper

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-layernorm-hardware-fingerprint-helper-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits and prior kernel-tuning notes. This attempt owns only `crates/rwkv-nn/src/kernels/train/layout.rs`, `crates/rwkv-nn/src/kernels/train/layer_norm/forward.rs`, and this note.
- User constraint: do not use the local GPU for timing. This is a non-GPU code-quality/key-design step.
- Prior-note/source search:
  - `rg -n "HardwareFingerprint|hardware fingerprint|load_width|plane_size|max_units_per_cube|max_cube_dim|max_shared_memory_size|max_vector_size|num_streaming_multiprocessors|num_tensor_cores|min_tensor_cores_dim" crates/rwkv-nn/src/kernels/train crates/rwkv-nn/src -S`
  - `sed -n '1,220p' crates/rwkv-nn/src/kernels/train/layout.rs`
- Matched evidence:
  - Train-kernel autotune keys repeatedly hand-copy the same public CubeCL hardware fingerprint fields.
  - The active objective requires kernel implementation choice to be keyed by hardware and shape; repeated hand-copying makes future omissions likely.
  - The previous audit established literal CUDA compute capability is not exposed through public CubeCL properties, so this helper should represent only the available public hardware fingerprint.
- Scope: introduce a reusable `CubeHardwareFingerprint` helper and use it in the LayerNorm autotune key first. Do not refactor every key in this branch.
- Hypothesis: centralizing the public hardware fingerprint for LayerNorm preserves the same key information while making the hardware/shape dispatch policy clearer and less error-prone.
- Expected keep/revert boundary:
  - Keep if `cargo check -p rwkv-nn --features cuda` passes and the generated key still includes all prior hardware fields in its `Display`.
  - Revert if this changes LayerNorm candidate validity, removes any hardware discriminator, or creates feature/dependency boundary issues.
- Planned edits:
  - Add `CubeHardwareFingerprint` to `crates/rwkv-nn/src/kernels/train/layout.rs`.
  - Replace the repeated hardware fields in `LayerNormForwardAutotuneKey` with `hardware: CubeHardwareFingerprint`.
  - Keep the GB10 BF16 D768 policy semantically unchanged.

## Change Log

- Added `CubeHardwareFingerprint` in `crates/rwkv-nn/src/kernels/train/layout.rs`, copying the same public CubeCL hardware fields previously embedded directly in LayerNorm's autotune key.
- Updated `LayerNormForwardAutotuneKey` in `crates/rwkv-nn/src/kernels/train/layer_norm/forward.rs` to hold `hardware: CubeHardwareFingerprint`.
- Kept the LayerNorm `Display` key text with the same hardware discriminator labels: `load`, `p`, `u`, `cube`, `s`, `vec`, `sm`, `tc`, and `tcdim`.
- Kept the GB10 BF16 D768 deterministic policy boundary semantically unchanged: it still requires CUDA, BF16, `d_model=768`, `rows=8192`, load width `128`, plane size `32`, max cube units `1024`, cube dim `(1024, 1024, 64)`, shared memory `101376`, SM count `48`, no exposed tensor core count, and min tensor-core dim `8`.

## Validation Log

- `rustup run nightly rustfmt crates/rwkv-nn/src/kernels/train/layout.rs crates/rwkv-nn/src/kernels/train/layer_norm/forward.rs`: passed.
- `cargo check -p rwkv-nn --features cuda`: passed locally in `13.11s`. This was compile-only and did not use local GPU timing.

## Remote Plan

- Host: `caizus@10.100.1.253`.
- Repo path: `/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Baseline path for later compare, if needed: `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_r9_20260516`.
- Next command: inspect remote repo path and then sync only the three owned files in this attempt. Do not sync local `target/`, weights, datasets, results, or broad dirty-tree files.

## Remote Log

- `ssh caizus@10.100.1.253 '...'`: failed before running remote commands with `Permission denied (publickey,password)` and missing `/usr/bin/ssh-askpass`. No remote files were changed.
- Prior notes identify the correct identity file as `~/.ssh/id_ed25519_dgx_spark_windows`.
- Next command: retry remote path inspection with `ssh -i ~/.ssh/id_ed25519_dgx_spark_windows -o BatchMode=yes -o ConnectTimeout=10 caizus@10.100.1.253`.
- Remote path inspection with that identity succeeded. Host is `spark-35ac`; repo paths exist under `/home/caizus/Projects/Packages`. Existing remote mtimes before sync: `layer_norm/forward.rs` `2026-05-16 01:13:28 -0500`, `layout.rs` `2026-05-15 23:03:44 -0500`.
- Next command: scoped path-preserving `rsync -azR` of `layout.rs`, `layer_norm/forward.rs`, and this note to the remote repo.
- Remote sync result: scoped `rsync -azR` completed for only `layout.rs`, `layer_norm/forward.rs`, and this note.
- Next command: remote compile gate `cargo check -p rwkv-nn --features cuda` in `/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Remote compile result: `cargo check -p rwkv-nn --features cuda` passed on `10.100.1.253` in `2.65s`.
- Remote baseline-path check: `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_r9_20260516` exists, and the older `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm` tree also exists.
- Next command: run remote standard compare against the current R9 projection baseline, captured in `target/rwkv-test/remote-layernorm-hardware-fingerprint-helper-r9-compare.log`.
- Remote compare result: invalid before activation/timing comparison. Release build completed in `2m 28s`, then `rwkv-test compare-rwkv-nn` failed copying `embedding/token_ids.safetensors` because `target/rwkv-test/rwkv_nn_actual/rwkv_lm/bf16/case_000000/embedding` did not exist. No correctness or speedup conclusion is drawn from this run.
- Next command: precreate the actual-output directory tree from the R9 baseline directory on `10.100.1.253`, then rerun `target/release/rwkv-test compare-rwkv-nn` against the same baseline and capture a new log.
- Remote compare rerun result: invalid with the same missing `embedding` parent path. The directory precreation command did not hit the path used by the compare binary. No correctness or speedup conclusion is drawn from this rerun.
- Next command: inspect the remote actual-output path with `readlink -f`, `pwd -P`, and `ls -ld`, then create the exact missing parent path explicitly before rerunning.
- Remote path inspection correction: the actual-output directory existed; the failed copy was caused by an invalid baseline argument. `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_r9_20260516` is the parent trace root, while `compare-rwkv-nn --baseline` needs `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_r9_20260516/rwkv_lm/bf16/case_000000`.
- Next command: rerun `target/release/rwkv-test compare-rwkv-nn` against the corrected R9 case directory and capture `target/rwkv-test/remote-layernorm-hardware-fingerprint-helper-r9-case-compare.log`.
- Remote corrected-case compare result: activation passed (`compared=54 passed=54 failed=0`). Timing was not an acceptance result because actual used `repeat=3,warmup=1` while the R9 baseline files record `repeat=9,warmup=3`; all timing rows failed with `timing profile mismatch`. Aggregate output still showed `actual_total_ms=90.758 baseline_total_ms=130.484 speedup=1.44x`, but it is not the clean timing gate.
- Next command: rerun the corrected R9 case compare with `--repeat 9 --warmup 3` to match the baseline timing profile, captured in `target/rwkv-test/remote-layernorm-hardware-fingerprint-helper-r9-repeat9-compare.log`.
- Remote R9 repeat-9 compare result on `10.100.1.253`: activation passed (`compared=54 passed=54 failed=0`). Timing summary: `compared=77 passed=76 failed=1 missing=0 extra=0 ignored=0 actual_total_ms=87.373 baseline_total_ms=130.484 speedup=1.49x`.
- The single timing failure was `timing/lm_head/projection.time.json`, `actual=15.571ms`, `baseline=15.158ms`, `speedup=0.97x`. This matches the already recorded R9 projection caveat and is outside this helper refactor's changed files.
- LayerNorm-related timing rows passed: `layer_norm0` `0.102ms` vs `0.274ms` (`2.68x`), and pre-layer-norm rows in cells all passed.
- Decision: keep the helper refactor. It preserves correctness on remote GB10, keeps total speedup above `1.0`, and does not introduce a new LayerNorm drift.
- Next command: sync the updated note back to the remote mirror so the failed baseline-root attempts and the final repeat-9 result are preserved there too.
- Remote note sync result: synced this updated note to `10.100.1.253`.
