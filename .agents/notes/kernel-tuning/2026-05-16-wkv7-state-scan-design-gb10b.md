# WKV7 State-Scan Design GB10B

- Date: 2026-05-16 17:22 +0800.
- Branch/worktree: `kernel-tuning-wkv7-state-scan-design-gb10-20260516b` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits plus retained/reverted kernel tuning changes. This attempt owns only this design note unless a later entry explicitly opens an implementation branch.
- Prior-note, source, and memory search commands:
  - `rg -n "WKV7|wkv7|state-scan|state scan|time-split|time split|chunk|handoff|shared-lanes|row_tile|prefix|scan" .agents/notes/kernel-tuning crates/rwkv-nn/src/kernels/train/time_mixer/wkv7 /root/.codex/memories/MEMORY.md -S`
  - `sed -n '270,690p' crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/forward.rs`
  - `sed -n '1,260p' crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/kernel.rs`
  - `sed -n '488,580p' crates/rwkv-nn/src/kernels/train/time_mixer/wkv7/mod.rs`
- Matched prior evidence:
  - Current remote post-target-logit attribution on `10.100.1.253` records `wkv7_pretrain_forward_output_kernel_f_bf16` at `22.412ms / 36`, `grid=(12,16,1)`, `block=(64,1,1)`, regs `108`, dynamic smem `1536`.
  - `2026-05-16-wkv7-shared-state-lanes-gb10.md` rejected shared-lanes candidates: activation-safe but much slower; `shared_lanes_8` around `2.894ms` median and `shared_lanes_4` around `3.633ms`, while `row_tile_64` stayed around `0.598ms`.
  - `2026-05-16-wkv7-row-tile16-forced.md` and later notes reject row-tile-only narrowing. Current GB10 autotune selects `row_tile_64`.
  - `2026-05-16-wkv7-remote-launch-design.md` says a real WKV7 improvement must change the recurrence split/state composition instead of repeating row-tile or lane-packing tweaks.
  - Remote `ncu` counters are blocked by `ERR_NVGPUCTRPERM`; this design can use source, algebra, nsys launch metadata, and trace-backed compare, but not remote achieved occupancy/warp/memory counters until permissions change.
- Machine/GPU: design target is remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`; local GPU must not be used.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, heads `12`, head size `64`, chunk len `16`.
- Kernel/stage: WKV7 pretrain forward output path only, currently dispatched through `wkv7_pretrain_output::<R,F>` and the `row_tile_{16,32,64}` `LocalTuner`.
- Changed boundary: this is not another `row_tile`, shared-lanes, or line-size attempt. It analyzes whether the WKV7 recurrence admits a state-scan/time-split implementation with mathematically correct state handoff.
- Hypothesis: the current one-block-per `(batch, head, row_tile)` implementation underfills GB10 because each row serially scans all `T=512` timesteps with a private `64`-column state vector. A time-split design could expose more blocks, but only if the recurrence state transition for each time segment can be composed exactly enough for BF16 trace correctness.
- Candidate parameters: none yet. A later implementation branch must name exact split parameters such as segment length, scan level, row/column ownership, saved handoff tensor shape, block size/warps, and deterministic numeric boundary.
- Expected keep/revert boundary: keep this note if it cleanly classifies feasible and infeasible state-scan designs. Do not edit WKV7 kernel code in this branch unless the note first proves a specific recurrence composition and trace validation path.
- Next command: derive the WKV7 state transition algebra from the current reference/kernel and decide which time-split forms are mathematically valid for a future implementation.

## Recurrence Algebra

- Current reference recurrence for one `(batch, head)` state matrix `S`:
  - `state_replacement = S * removal_key_normalized` reduced over columns.
  - `S' = S * diag(decay) + outer(state_replacement, replacement) + outer(value, replacement_key)`.
  - `output = S' * receptance` reduced over columns.
- For row-vector convention, define:
  - `a_t = removal_key_normalized_t`
  - `p_t = replacement_t`
  - `d_t = decay_t`
  - `k_t = replacement_key_t`
  - `v_t = value_t`
  - `r_t = receptance_t`
  - `M_t = diag(d_t) + a_t p_t^T`
- Then each step is:
  - `S_{t+1} = S_t M_t + outer(v_t, k_t)`
  - `output_t = S_{t+1} r_t`
- A segment `[l, r)` is therefore an affine state transform:
  - `S_r = S_l P_{l:r} + Q_{l:r}`
  - Segment composition is exact in f32 algebra:
    - `(P_a, Q_a) then (P_b, Q_b) = (P_a P_b, Q_a P_b + Q_b)`
- Consequence: a time-split implementation can be mathematically valid, but it cannot only split the existing loop. It must either:
  1. compute segment transforms `(P, Q)`, scan them to get each segment's start state, then recompute outputs inside each segment; or
  2. precompute enough per-time prefix transform data so outputs can be corrected from segment start state without full recomputation.

## Candidate Classification

- Invalid or duplicate:
  - More `row_tile` variants: duplicate of rejected/selected `row_tile_{16,32,64}` evidence.
  - More shared-lanes/row-column lane packing: duplicate of rejected shared-lanes branch; it changed resource mapping but not the time recurrence.
  - Splitting time into independent chunks without state handoff: mathematically wrong because `S_l` is required for every later chunk.
- Feasible but large:
  - Two-pass segment recompute:
    1. Kernel A computes per-segment `(P, Q)` for segment length `L` such as `16` or `32`.
    2. Kernel B scans segment transforms across `T/L` segments for each `(batch, head)` to get segment start states.
    3. Kernel C recomputes the original recurrence inside each segment from its correct start state and writes outputs.
  - This exposes about `B * H * num_segments = 16 * 12 * 32 = 6144` segment blocks instead of the current `B * H * row_tiles = 192` row-tile blocks, which addresses under-occupancy.
  - It adds substantial work and storage: `(P, Q)` are each `64 x 64` f32 per segment. For `B=16`, `H=12`, `T/L=32`, one full f32 tensor is about `6144 * 4096 * 4 = 96 MiB`; storing both is about `192 MiB` before any prefix/output helper tensors.
- More promising but still large:
  - Segment transform plus output-prefix correction:
    1. Precompute segment `(P, Q)` and zero-start segment outputs.
    2. Also precompute per-time vectors that map segment start state to output contribution.
    3. After segment-start scan, add the start-state contribution to each output.
  - This can avoid full output recomputation, but it still stores additional per-time/per-segment prefix vectors and requires careful f32 reduction-order validation.

## Numerical Boundary

- The current single-kernel path updates each row's `64`-element state in a fixed loop order and writes BF16 outputs.
- Segment-scan changes the order of f32 matrix products and affine accumulation. Even if algebraically equivalent, BF16 traces may drift because `P/Q` composition changes summation order and may round intermediate saved tensors.
- Any implementation candidate must keep segment transform intermediates in f32 and validate against the real trace output first. If it stores `P/Q` as BF16 to save memory, that is a separate numeric candidate and should be expected to drift until proven otherwise.

## Decision

- Do not implement a quick WKV7 state-scan kernel in this branch. A correct implementation is a multi-kernel operator with new intermediate tensors and a trace-backed numeric plan.
- The next implementation branch, if chosen, should prototype only the two-pass segment recompute path with f32 `P/Q`, segment length `16`, and forward-output only. Keep/revert boundary should be:
  - remote compile passes,
  - remote activation `54/54` passes,
  - WKV7 `nsys` targeted time improves over the current `22.412ms / 36` post-target-logit reference,
  - total remote standard compare stays `>1.0`,
  - if activation drifts, inspect whether the drift comes from transform composition order, f32/BF16 storage, or implementation indexing before reverting.
- This is materially different from row-tile/shared-lanes attempts because it changes time decomposition and state handoff, not row ownership.
