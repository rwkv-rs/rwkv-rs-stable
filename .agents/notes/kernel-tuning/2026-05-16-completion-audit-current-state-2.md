# Completion Audit Current State 2

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-completion-audit-current-state-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout already carries broad unrelated workspace edits, generated traces, untracked notes, and untracked skill files. This attempt owns only this completion-audit note unless a later entry explicitly names a skill or source edit.
- User constraint: continue remote-first on `10.100.1.253`; do not use local GPU timing while the user may be using the local GPU.
- Scope: current-state completion audit against the active kernel-tuning objective. This branch does not run GPU benchmarks, clear caches, or edit kernel source.
- Prior-note search commands already run:
  - `find .agents/notes/kernel-tuning -maxdepth 1 -type f -name '*completion*audit*.md' -o -name '*current-goal*audit*.md' | sort`
  - `rg -n "completion audit|Current Goal|Final acceptance|speedup|ncu|local|10\\.100\\.1\\.253|LocalTuner|autotune key|LayerNorm|WKV7" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md -S`
- Matched prior evidence:
  - Existing completion-audit notes already mark remote GB10 as clean in several historical states, but they also leave local speedup and remote `ncu` counter evidence incomplete.
  - `2026-05-16-autotune-key-coverage-audit.md` records the current key coverage and fixed-dispatch classification.
  - `.agents/skills/kernel-tuning/SKILL.md` records the current duplicate guards and the known remote `ncu` permission blocker.
- Objective restated as concrete success criteria:
  1. Kernel-tuning workflow uses branch/worktree plus note per materially new attempt; failed attempts are reverted without deleting branches.
  2. Kernel implementation choice is keyed or dispatched by backend/runtime, hardware proxy or compute capability when available, dtype, model shape, rows, block/warp/vector candidate dimensions, in-place/alias behavior, and deterministic numeric policy.
  3. LayerNorm specifically preserves local correctness while allowing the verified remote GB10 `256` path only under a hardware/shape/dtype/deterministic key.
  4. Burn/CubeCL autotune mechanics are understood and documented: `LocalTuner::execute` key/cache path, warmup/profiling miss boundary, cache hit hot path, candidate parameters, and accuracy-guard limits.
  5. CUDA lower-level analysis exists for reduction launch configuration, occupancy/register/shared-memory/coalescing, BF16 numerical differences, and bad-result root causes rather than immediate unexamined reverts.
  6. `ncu` achieved occupancy, warp execution efficiency, and memory throughput are collected where permission allows; remote permission blockers are explicitly documented.
  7. Final acceptance requires trace-backed activation passing and timing speedup `> 1.0` on both local and `10.100.1.253`.
- Expected outcome: if any criterion remains missing or weakly verified, do not mark the goal complete. Record the missing items and next concrete action.
- Next command: inspect the most recent completion audits, LayerNorm runtime-dispatch note, autotune-key coverage audit, and current kernel-tuning skill.

## Evidence Inspection

- Commands:
  - `sed -n '1,220p' .agents/notes/kernel-tuning/2026-05-16-current-goal-audit-after-r9.md`
  - `sed -n '1,220p' .agents/notes/kernel-tuning/2026-05-16-completion-audit-local-fresh-remote-clean.md`
  - `sed -n '1,180p' .agents/notes/kernel-tuning/2026-05-16-layernorm-runtime-dispatch.md`
  - `sed -n '1,160p' .agents/notes/kernel-tuning/2026-05-16-layernorm-ordered256-accuracy-rootcause.md`
  - `sed -n '1,180p' .agents/notes/kernel-tuning/2026-05-16-autotune-key-coverage-audit.md`
  - `sed -n '1,120p' .agents/skills/kernel-tuning/SKILL.md`
  - live-source `rg` reads for `CubeHardwareFingerprint`, `LayerNormForwardAutotuneKey`, `Wkv7PretrainOutputAutotuneKey`, WKV7 fixed dispatch, and skill rules.
- Source evidence:
  - `crates/rwkv-nn/src/kernels/train/layout.rs` defines `CubeHardwareFingerprint` with load width, plane size, max cube units/dimensions, shared memory, max vector size, SM count, and tensor-core fields.
  - `crates/rwkv-nn/src/kernels/train/layer_norm/forward.rs` has `BLOCK_SIZE_CANDIDATES = [64, 128, 256, 512, 768, 1024]`, `LayerNormForwardAutotuneKey` fields for runtime, dtype, d_model, rows, hardware, in-place, deterministic, and deterministic minimum block size. The launch passes `block_size / WARP_SIZE` as `num_warps`.
  - `layer_norm/forward.rs` also has `supports_gb10_bf16_d768_layer_norm(...)`, so the remote GB10 `256` path is keyed by runtime, dtype, d_model, rows, and hardware fingerprint rather than a universal constant.
  - `wkv7/forward.rs` has `Wkv7PretrainOutputAutotuneKey` and `PRETRAIN_OUTPUT_ROW_TILE_CANDIDATES = [16, 32, 64]`; the live candidate dimension is row tile. WKV7 state forward and backward still use fixed `CubeDim::new_1d(shape[3] as u32)`.
  - `.agents/skills/kernel-tuning/SKILL.md` records `LocalTuner::execute` cache-hit/miss behavior, `autotune-checks` limitations, remote `RmProfilingAdminOnly: 1`, and the fixed-dispatch classification rule. One `rg` command against this file had an unescaped backtick in the search pattern, causing a shell `1.0` command-substitution error, but the output still returned the relevant skill lines.
- Current remote acceptance evidence from prior notes:
  - R9 projection-baseline audit: activation `54/54`, timing total speedup `1.48x`; older run had one `lm_head/projection` row at `0.99x`.
  - Later projection-matmul audit: activation `54/54`, timing `77/77`, total speedup `1.51x`, `lm_head/projection` `1.01x`.
  - Remote `ncu` counters are still blocked by `RmProfilingAdminOnly: 1`; standard compare and `nsys` metadata are the available remote evidence until privileged access changes.
- Current local acceptance evidence from prior notes:
  - Local regenerated-baseline compare exists with activation `54/54` and total speedup `2.44x`.
  - Local row-level timing remains incomplete (`22/76` pass in the cited audit), and older checked-in baseline evidence around the same period was still below `1.0`.
  - The user explicitly asked not to use local GPU timing right now, so this audit does not refresh local numbers.

## Prompt-To-Artifact Checklist

| Requirement | Evidence inspected | Status |
| --- | --- | --- |
| Branch/worktree plus note per new tuning attempt. | Current branch is `kernel-tuning-completion-audit-current-state-20260516`; this note was created before audit work. Skill requires branch+note discipline and duplicate preflight. | Covered for current continuation. |
| Failed attempts reverted, branches/notes retained. | Skill and many branch notes record keep/revert states for WKV7, GatedReadout, key_prepare, residual, LayerNorm, lm-head, and channel-mixer attempts. | Covered by ledger, not by a single verifier. |
| Runtime/backend in autotune key. | Live source keys use `runtime: R::name(...)`; key-coverage audit classifies fixed dispatch separately. | Covered for live tuned custom kernels. |
| GPU arch / compute capability. | Exact CUDA CC is not exposed in current public CubeCL hardware properties; project uses `CubeHardwareFingerprint` fields and records the limitation in skill. | Covered through available hardware proxy; literal CC remains unavailable without CubeCL/API extension. |
| dtype. | Live keys include `dtype: DType`; LayerNorm and WKV7 evidence read from source. | Covered. |
| d_model and rows. | LayerNorm key has `d_model` and rows; WKV7 key has rows, d_model, heads/head_size; other audited keys use embedded dim or token/vocab dimensions. | Covered for inspected kernels. |
| block size, num_warps, vector width. | Candidate names/groups encode `block_*`, `line_size_*`, `row_tile_*`; LayerNorm/lm-head derive `num_warps` from block size. Fixed-dispatch kernels are explicitly classified. | Covered where candidate sets exist; fixed-dispatch paths require a new keyed wrapper before changing constants. |
| in-place / alias. | Live tuned keys include `is_in_place` or lhs/rhs in-place flags; residual add covers both sides. | Covered for tuned kernels. |
| deterministic numeric boundary. | Live keys include `deterministic`; LayerNorm includes `deterministic_min_block_size`; skill blocks known-invalid numeric boundaries. | Covered for known sensitive kernels. |
| LayerNorm remote `256` vs local `1024` handled by dispatch/autotune. | `layer_norm/forward.rs` GB10 support function plus runtime dispatch note: local activation stayed valid with deterministic `1024`; remote GB10 selected/passed `256`. | Covered. |
| Burn/CubeCL LocalTuner mechanics understood. | Skill documents cache hit/miss, `LocalTuner::init`, persistent checksum, warmup/profiling boundary, and `autotune-checks` limitation. | Covered. |
| Host-side tuner lookup cost not counted as hot path without proof. | Skill says cache-hit overhead is key lookup plus dispatch and must be proven before bypassing; local bypass attempt is recorded negative. | Covered by rule and negative attempt. |
| CUDA lower-level design: warp/block, occupancy/register/shared-memory/coalescing. | Current R9 `nsys` note and skill record WKV7 underfill (`192` blocks, `regs/thread=127`, theoretical occupancy `33.3%`, wave occupancy about `16.7%`) and other kernels' launch/register metadata. | Partially covered; achieved counters still missing on remote. |
| Use `ncu` achieved occupancy, warp efficiency, memory throughput. | Remote `ncu` permission blocker recorded: `RmProfilingAdminOnly: 1`, no passwordless sudo. Local ncu exists in prior notes, but local GPU is currently excluded by user. | Incomplete / environment-blocked remotely. |
| Poor timing results investigated before revert. | WKV7 segment/output-factor/direct-value, GatedReadout rowpack, key_prepare warps, LayerNorm combined/ordered attempts all include cause analysis. | Covered for recorded attempts. |
| Precision problems checked for code-vs-math root cause. | Ordered-256 diagnostic proved intended ordered math matches `1024` on CPU for relevant LayerNorm inputs; failure classified as device implementation/boundary issue. | Covered for the known LayerNorm precision issue. |
| Final speedup `>1.0` on local and `10.100.1.253`. | Remote current R9 total speedup is `>1.0`; local regenerated-baseline total speedup is `2.44x`. Local row-level clean acceptance is still incomplete and local timing was not refreshed due user GPU constraint. | Partially covered; goal remains open. |

## Audit Decision

- Do not mark the active goal complete.
- Covered: workflow discipline, key coverage for live tuned kernels, LayerNorm hardware/shape dispatch, LocalTuner mechanism, most duplicate guards, root-cause analysis for the ordered-256 precision issue, and remote total speedup `>1.0`.
- Missing or weakly verified:
  - Remote `ncu` achieved occupancy / warp efficiency / memory throughput remains blocked by `RmProfilingAdminOnly: 1`.
  - Local clean row-level acceptance remains incomplete even though a locally regenerated baseline has total speedup `>1.0`.
  - WKV7 still has the clearest structural performance gap, but all small duplicate families are closed; the next valid WKV7 branch must be a materially different algorithm.
- Next concrete action after this audit: sync this completion-audit note to `10.100.1.253` with the explicit SSH key, then stop this audit branch. Do not run local GPU or repeat closed WKV7/key_prepare/channel-mixer experiments.

## Validation And Sync

- Command: `git diff --no-index --check -- /dev/null .agents/notes/kernel-tuning/2026-05-16-completion-audit-current-state-2.md`
- Result: passed; no whitespace errors reported.
- Command: `rsync -azR -e 'ssh -i ~/.ssh/id_ed25519_dgx_spark_windows -o BatchMode=yes -o ConnectTimeout=10' .agents/notes/kernel-tuning/2026-05-16-completion-audit-current-state-2.md caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable/`
- Result: passed; remote shell printed the expected zshenv trace and rsync returned exit `0`.
