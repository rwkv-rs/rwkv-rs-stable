# Forward Autotune Key Hardware Fingerprint Sweep

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-forward-keys-hardware-fingerprint-sweep-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits and prior kernel-tuning work. This attempt owns only forward autotune key structure changes in `crates/rwkv-nn/src/kernels/train/**/forward.rs`, the already introduced `CubeHardwareFingerprint` helper, and this note.
- User constraint: use remote `10.100.1.253` for GPU validation; do not use local GPU timing.
- Prior-note/source search:
  - `rg -n "lm_head/projection|projection|WKV7|wkv7|time_mixer|ncu|nsys|online|target-logit|Cubek|TMA" .agents/notes/kernel-tuning /root/.codex/memories/MEMORY.md -S`
  - `rg -n "struct .*AutotuneKey|impl AutotuneKey|local_tuner!|TuneGroup|TunableSet::new|anchor\\(" crates/rwkv-nn/src/kernels/train -S`
  - `rg -n "CubeHardwareFingerprint|hardware:|load_width|plane_size|max_units_per_cube|max_vector_size|num_streaming_multiprocessors" crates/rwkv-nn/src/kernels/train -S`
- Matched evidence:
  - The previous LayerNorm helper branch introduced `CubeHardwareFingerprint` and validated it on remote GB10 with activation `54/54`, total speedup `1.49x`, and no LayerNorm drift.
  - Many forward autotune keys still hand-copy the same public CubeCL hardware fields.
  - Duplicate guards close residual-add A/B, LayerNorm ordered-256, WKV7 row-tile-only changes, key-prepare warps, gated-readout warp32, and lm-head row-kernel variants. This attempt is mechanical key-structure cleanup, not a new kernel algorithm or candidate.
- Changed boundary:
  - Convert forward autotune keys that already carry the repeated public hardware fields to `hardware: CubeHardwareFingerprint`.
  - Keep candidate sets, kernel launch shapes, math, deterministic policy, in-place/alias flags, and `Display` text semantically unchanged.
  - Do not touch backward keys or `residual_add` in this branch.
- Machine/GPU for runtime validation: remote `caizus@10.100.1.253`, host `spark-35ac`, NVIDIA GB10.
- Shape/dtype target: `rwkv_lm` BF16, `B=16`, `T=512`, `d_model=768`, `rows=8192`, R9 projection baseline case under `/home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_r9_20260516/rwkv_lm/bf16/case_000000`.
- Expected keep/revert boundary:
  - Keep if local and remote compile checks pass and remote compare preserves activation while total speedup stays above `1.0`.
  - Revert if any candidate validity changes, any forward key loses a hardware discriminator, activation drifts, or the compare result shows a new timing regression attributable to this key structure change.
- Next edit: update only forward autotune key structs and their key construction/use sites to use `CubeHardwareFingerprint`.

## Change Log

- Updated `crates/rwkv-nn/src/kernels/train/lm_head_l2wrap_ce/forward.rs` to use `hardware: CubeHardwareFingerprint` in `LmHeadL2WrapCeForwardAutotuneKey`; `block_size <= max_units_per_cube` now reads through `key.hardware`.
- Updated `crates/rwkv-nn/src/kernels/train/channel_mixer/forward.rs` to use `hardware: CubeHardwareFingerprint` in `ChannelMixerElementwiseAutotuneKey` for both mix and ReLU-square tuner paths.
- Updated `crates/rwkv-nn/src/kernels/train/time_mixer/mix6/forward.rs` to use `hardware: CubeHardwareFingerprint` in `Mix6ForwardAutotuneKey`.
- Updated `crates/rwkv-nn/src/kernels/train/time_mixer/learning_rate_gate/forward.rs`, `value_residual_gate/forward.rs`, and `weight_decay_transform/forward.rs` to use `hardware: CubeHardwareFingerprint` in their forward keys.
- Updated `crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/forward.rs` to use `hardware: CubeHardwareFingerprint` in `Wkv7PretrainOutputAutotuneKey`; `row_tile <= max_units_per_cube` now reads through `key.hardware`.
- Candidate names, candidate sets, kernel launch arguments, shape anchors, `is_in_place`, and deterministic flags were intentionally left unchanged.

## Validation Log

- Next command: search forward train kernels for remaining repeated public hardware fields to catch missed key sites before formatting.
- Remaining-field search result: only `residual_add/forward.rs` still hand-copies public hardware fields among forward key files, which is intentionally out of scope due the residual duplicate-experiment guard; `layer_norm/forward.rs` only uses fields through `CubeHardwareFingerprint` for the GB10 deterministic policy checks.
- Next command: run nightly rustfmt on the touched forward files.
- `rustup run nightly rustfmt` on the seven touched forward files: passed.
- Next command: local compile gate `cargo check -p rwkv-nn --features cuda` without local GPU timing.
- `cargo check -p rwkv-nn --features cuda`: passed locally in `10.36s`. This was compile-only and did not use local GPU timing.
- Next command: scoped path-preserving sync of the seven touched forward files, `layout.rs`, and this note to `10.100.1.253`, then run the remote compile gate.
- Remote sync result: scoped `rsync -azR` copied the seven touched forward files, `layout.rs`, and this note to `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable/`.
- Next command: remote compile gate `cargo check -p rwkv-nn --features cuda` in `/home/caizus/Projects/Packages/rwkv-rs-stable`.
- Remote compile result: `cargo check -p rwkv-nn --features cuda` passed on `10.100.1.253` in `1.50s`.
- Next command: remote R9 compare with the corrected case path and matching timing profile: `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --baseline ~/Projects/Packages/rwkv-rs-test/test_gen_projection_r9_20260516/rwkv_lm/bf16/case_000000 --repeat 9 --warmup 3`, captured in `target/rwkv-test/remote-forward-keys-hardware-fingerprint-sweep-r9-repeat9.log`.
- Remote R9 compare result on `10.100.1.253`: release rebuild completed in `2m 24s`; activation passed (`compared=54 passed=54 failed=0`).
- Timing summary: `compared=77 passed=76 failed=1 missing=0 extra=0 ignored=0 actual_total_ms=88.043 baseline_total_ms=130.484 speedup=1.48x`.
- The single failed timing row was the known R9 projection caveat: `timing/lm_head/projection.time.json`, `actual=15.223ms`, `baseline=15.158ms`, delta `0.065ms`, reported as `1.00x` but still slightly slower.
- Rows touched by this key-structure sweep remained correctness-clean and total speedup stayed above `1.0`; no new activation drift appeared.
- Decision: keep the forward key hardware fingerprint sweep. Residual forward and backward keys remain for later scoped branches.
- Next command: sync the updated note back to the remote mirror.
- Remote note sync result: synced this updated note to `10.100.1.253`.
