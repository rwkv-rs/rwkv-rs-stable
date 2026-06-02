# WKV7 Low-Rank State-Scan Design GB10

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-wkv7-lowrank-state-scan-design-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits plus retained/reverted kernel tuning changes. This design attempt owns only this note unless a later entry explicitly opens a source implementation boundary.
- User constraint: debug first on remote `10.100.1.253`; do not use the local GPU.
- Prior-note/source search commands:
  - `rg -n "WKV7|wkv7|state-scan|state scan|segment|segment_recompute|shared-lanes|row_tile|low-rank|lowrank|P/Q|255|13\\.2x" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md crates/rwkv-nn/src/kernels/train/time_mixer/wkv7 -S`
  - `sed -n '1,520p' crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/{forward.rs,kernel.rs}`
- Matched prior evidence:
  - Dense segment transform/scan/recompute with f32 `P/Q` and `segment_len=16` passed activation on GB10, but transform+scan+recompute was about `13.2x` the current `row_tile_64` output kernel and the scan kernel used `255` registers/thread.
  - WKV7 shared-lanes and row-tile-only variants are closed; GB10 selects `row_tile_64`.
  - Current post-target-logit attribution has WKV7 output around `22.4ms / 36`, `grid=(12,16,1)`, `block=(64,1,1)`, regs `108`, dynamic smem `1536`.
  - Remote `ncu` counters are blocked by `ERR_NVGPUCTRPERM`; this branch can use algebra, source inspection, `nsys` metadata, and trace-backed compare in later implementation, but not remote achieved occupancy counters.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, heads `12`, head size `64`, chunk length `16`.
- Changed boundary: this is not another dense segment `P/Q` implementation. The hypothesis is to exploit the per-step transition structure `diag(decay) + outer(removal, replacement)` so segment transforms are represented as diagonal plus low-rank factors rather than dense `64 x 64` f32 matrices.
- Initial hypothesis:
  - A product of `L` matrices of the form `diag(d_t) + a_t p_t^T` can be represented as a diagonal transform plus rank-`<=L` factors.
  - The segment additive term `Q` can also be represented as a sum of outer products with rank bounded by the segment length if later right-transforms are kept in the same low-rank form.
  - This may reduce global transform storage and scan register pressure compared with dense `P/Q`, while still exposing time-segment parallelism.
- Expected keep/revert boundary for this design branch:
  - Keep only if the note derives a concrete representation, estimates storage/compute against the rejected dense segment path and current row-tile path, and identifies a trace-backed validation plan.
  - Do not edit WKV7 kernels in this branch unless the note first proves the implementation boundary, intermediate tensor shapes, and failure diagnostics.
- Next command: derive the low-rank segment transform algebra from the current WKV7 recurrence and classify whether it is a viable implementation candidate.

## Low-Rank Algebra Derivation

- Source basis: current WKV7 output recurrence for one `(batch, head)` state matrix is
  `S_{t+1} = S_t M_t + outer(value_t, replacement_key_t)`, where
  `M_t = diag(decay_t) + outer(removal_key_normalized_t, replacement_t)`.
- A segment linear transform can be represented as
  `P = diag(g) + U V^T`, with rank bounded by the segment length before exact rank
  saturation at `head_size=64`.
- A segment additive transform can also be represented in low-rank form:
  `Q = sum_t outer(value_t, replacement_key_t P_{t+1:r})`, so for a segment of length
  `L`, rank is bounded by `L`.
- For `segment_len=16`, this reduces transform storage versus dense f32 `P/Q`:
  - dense `P+Q`: `2 * 64 * 64 = 8192` f32 values per segment;
  - low-rank `P+Q`: diagonal `64` plus four `64 x 16` factor blocks, about `4160`
    f32 values per segment.
- This is only about a 2x storage reduction. It does not remove the need to obtain a
  correct segment start state or to generate outputs for every timestep.

## Viability Check Against Recorded GB10 Evidence

- The rejected dense segment-recompute branch already measured the three pieces on GB10:
  - transform: `131.157ms / 36`;
  - scan: `112.660ms / 36`, with `255` registers/thread;
  - recompute output: `46.976ms / 36`;
  - current row-tile output reference: about `22.083ms / 36`.
- Even if low-rank factors made transform and scan free, the already-measured recompute
  output kernel alone is about `2.1x` slower than the current row-tile WKV7 output path.
- Avoiding recompute would require output-prefix correction. For each time inside a
  segment, output has a start-state contribution `S_start z_t`, where `z_t` is a
  per-time transformed receptance vector. Computing this for all rows/times is still a
  dense state-vector product unless the prefix state remains very low rank.
- Across segments, exact prefix state rank grows with time and saturates at `64`; after
  saturation, low-rank correction has the same order as dense `64 x 64` state-vector
  work. Exact rank compression would require a new numeric algorithm and its own BF16
  accuracy boundary, not a small WKV7 tuning candidate.

## Decision

- Do not implement low-rank segment state-scan in this tuning pass.
- The low-rank representation is mathematically valid for segment summaries, but it does
  not overcome the measured GB10 lower bound from the rejected segment-recompute path.
- This closes the current WKV7 low-rank branch as design-negative. The next WKV7 attempt
  would need a different output-generation algorithm that avoids both dense global
  `P/Q` traffic and per-segment recompute/correction work; otherwise it is a duplicate
  of the rejected segment family.
- Keep/revert state: no source edits were made in this branch. Keep only this note as
  duplicate-prevention evidence.
