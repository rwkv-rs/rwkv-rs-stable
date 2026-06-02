# LayerNorm Ordered-256 Reconstruct

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-layernorm-ordered256-reconstruct-20260516` in the existing dirty workspace.
- Dirty-tree constraint: this checkout carries broad unrelated uncommitted workspace changes plus the current remote-clean tuning tree. This branch is scoped to `crates/rwkv-nn/src/kernels/train/layer_norm/{forward.rs,kernel.rs}` and this note unless a later note expands validation.
- Prior-note/source search command:
  - `rg -n "ordered-256|ordered256|block_256_d768_ordered|reconstruct|device-side guard|CPU diagnostic|ordinary 256|value_from_first_cell|lm_head/embedded_context" .agents/notes/kernel-tuning /root/.codex/memories/MEMORY.md -S`
- Matched prior evidence:
  - `2026-05-16-layernorm-d768-ordered256.md` recorded an earlier ordered-256 candidate that passed local activation and selected around `50.4us` vs `93.6us` for `block_1024`, but was reverted under remote-first acceptance because remote GB10 used ordinary `block_256`.
  - `2026-05-16-layernorm-ordered256-post-gates.md` later rejected a rewritten ordered-256 candidate because activation failed in `value_from_first_cell` and `lm_head/embedded_context`, then moved the stale ordered cache aside.
  - `2026-05-16-layernorm-ordered256-accuracy-rootcause.md` showed CPU-simulated ordered-256 exactly matches the `1024` reduction order for `layer_norm0`, `cell0_pre_time_mix_ln1`, and `cell0_pre_channel_mix_ln2`; ordinary `256` causes BF16 output deltas at cell pre-LayerNorm boundaries. The failure is implementation/boundary hygiene, not the intended math.
  - `2026-05-16-local-short-kernel-provenance.md` shows local acceptance is currently activation-clean but timing-fails at `0.92x`; a correct ordered-256 LayerNorm candidate is the only non-duplicate local implementation with enough likely upside.
- Machine/GPU: local `NVIDIA GeForce RTX 5090`, compute capability `12.0`; remote GB10 must remain on the ordinary `256` policy if this branch reaches remote validation.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`.
- Kernel/stage: LayerNorm forward.
- Changed boundary: reconstruct ordered-256 as a new implementation attempt. This is not ordinary `block_256`, and it is not the rejected post-gates source; it explicitly computes the 24 warp partials that the `1024` path would produce for columns `0..767`, then reduces those partials in one final warp.
- Candidate parameters: `block_256_d768_ordered`, block size `256`, `8` warps, BF16 `D=768`, rows `8192`, deterministic `true`, valid only when `deterministic_min_block_size > 256` so it does not compete with the verified GB10 ordinary-256 key.
- Expected keep/revert boundary: keep only if compile passes, local activation remains `54/54 PASS`, and local timing improves over the current `0.92x` without stale cache contamination. If activation drifts, revert immediately and keep this note as implementation-negative evidence.
- Next edit: add the ordered-256 kernel and candidate, then format and compile.

## 2026-05-16 Code Change

- Added `layer_norm_d768_ordered256_forward_kernel` in `layer_norm/kernel.rs`.
  - The kernel launches `256` units / `8` warps.
  - For each row it writes `3 * 8 = 24` warp partials for columns `0..255`, `256..511`, and `512..767`, then reduces those partials with one final warp. This is intended to match the `1024` path's 24 nonzero warp partials plus 8 zero partials.
  - It computes sum and squares with the same ordered helper and writes output with stride `256`.
- Added `block_256_d768_ordered` in `layer_norm/forward.rs`.
  - Valid only for CUDA BF16 `D=768`, rows `8192`, deterministic mode, `deterministic_min_block_size > 256`, and `max_units_per_cube >= 256`.
  - This prevents it from competing on the verified GB10 ordinary-256 policy.
- Next command:
  - `rustup run nightly rustfmt crates/rwkv-nn/src/kernels/train/layer_norm/forward.rs crates/rwkv-nn/src/kernels/train/layer_norm/kernel.rs`
  - `cargo check -p rwkv-nn --features cuda`

## 2026-05-16 First Compile Result

- Formatting passed.
- `cargo check -p rwkv-nn --features cuda` failed in the Cube macro with `usize: From<i32>` at the new ordered kernel.
- Likely cause: an untyped integer literal in the new Cube kernel, most likely `SharedMemory::<f32>::new(32)`, is being treated as `i32` by macro lowering.
- Next edit: make the shared-memory size literal explicitly `usize`, then rerun rustfmt and compile.

## 2026-05-16 Compile Fix Result

- Code fix: changed the ordered kernel shared-memory allocation to `SharedMemory::<f32>::new(32usize)`.
- Formatting passed.
- `cargo check -p rwkv-nn --features cuda` passed.
- Next command:
  - Check local GPU/cache state before running the standard local compare.

## 2026-05-16 Local Pre-Compare Check

- GPU state: `NVIDIA GeForce RTX 5090`, compute capability `12.0`, utilization `5%`, memory `3018/32607 MiB`; no compute apps reported.
- Active LayerNorm cache still has only the old checksum and selected `block_1024`.
- The moved-aside rejected cache contains older failed/stale entries, including the previous `block_256_d768_ordered`; it is not the active cache file.
- Because the candidate set changed, the active persistent cache checksum should not match and the next compare should retune the new candidate set.
- Next command:
  - `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --repeat 3 --warmup 1`

## 2026-05-16 Local Compare Result

- Command:
  - `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --repeat 3 --warmup 1`
- Build provenance:
  - Rebuilt and ran `target/release/rwkv-test` from this branch.
- Correctness:
  - `activation_summary compared=54 passed=52 failed=2 missing=0 extra=0`
  - Failed rows:
    - `cells/cell_0000/time_mixer/value_from_first_cell.safetensors`, `max_abs=1.152344e-1`.
    - `lm_head/embedded_context.safetensors`, `max_abs=9.375000e-2`.
  - `layer_norm0/embedded_context.safetensors` passed but showed nontrivial drift: `max_abs=2.075195e-3`, cosine `0.99998410`.
- Timing:
  - `timing_summary compared=76 passed=14 failed=62 missing=0 extra=0 ignored=1 actual_total_ms=38.717 baseline_total_ms=35.957 speedup=0.93x`
  - Pre-layernorm groups improved versus the `0.92x` acceptance run but activation failed, so this is rejected.
- Interpretation:
  - This reconstructed device kernel still does not preserve the trace-backed `1024` numerical boundary. It may still differ from the CPU simulation because the device helper does not exactly match the `1024` kernel's Cube execution/cast behavior, or because of another implementation detail in the new kernel.
  - Do not keep this code.
- Next command:
  - Inspect the active LayerNorm autotune cache to confirm whether `block_256_d768_ordered` was selected, then revert only this branch's LayerNorm code changes and clean the active generated cache if needed.

## 2026-05-16 Cache/Diff Inspection

- Active cache appended a new checksum entry and selected `block_256_d768_ordered` as fastest:
  - `fastest_index=6`
  - `block_256_d768_ordered` median about `49.597us`
  - `block_1024` median about `97.844us`
- This confirms the failed activation came from the reconstructed ordered candidate being selected.
- `layer_norm/{forward.rs,kernel.rs}` are untracked in the broader dirty workspace, so `git diff -- <files>` is empty even though the files contain this branch's source edit. Revert must be a manual source edit, not `git checkout`.
- Next edit:
  - Remove only the reconstructed ordered candidate/kernel from `layer_norm/forward.rs` and `layer_norm/kernel.rs`, then move the active generated LayerNorm cache aside before any further compare.

## 2026-05-16 Revert Result

- Reverted only this branch's reconstructed ordered candidate/kernel from:
  - `crates/rwkv-nn/src/kernels/train/layer_norm/forward.rs`
  - `crates/rwkv-nn/src/kernels/train/layer_norm/kernel.rs`
- Moved the active generated LayerNorm cache aside to:
  - `target/autotune/0.10.0/device-0-0-cuda/rwkv_nn-kernels-train-layer_norm-forward-layer-norm-forward.json.log.rejected-ordered256-reconstruct-20260516`
- Formatting passed after revert.
- `cargo check -p rwkv-nn --features cuda` passed after revert.
- Decision:
  - Reject this reconstructed ordered-256 implementation. It confirms the user's concern: the failing precision is not explained by the intended math; my device implementation still failed to reproduce the safe boundary.
  - Keep this branch and note as negative implementation evidence.
- Next command:
  - Confirm no ordered-256 source symbol remains in the live LayerNorm source.

## 2026-05-16 Source Cleanup Check

- `rg` found no ordered-256 symbols under `crates/rwkv-nn/src/kernels/train/layer_norm`; matches were only in this note.
- Next edit:
  - Update the project `kernel-tuning` skill with this negative result so the same reconstructed device-kernel attempt is not repeated.

## 2026-05-16 Skill Update

- Updated `.agents/skills/kernel-tuning/SKILL.md` with a known local negative result:
  - reconstructed BF16 `D=768` ordered-256 LayerNorm device kernel still failed activation;
  - CPU diagnostics showed the intended math matches `1024`;
  - future attempts must not reintroduce ordered-256 without a device-side or trace-backed guard proving equality to the `1024` output on cell pre-LayerNorm inputs.
- No further compile is required for this skill-only edit.
