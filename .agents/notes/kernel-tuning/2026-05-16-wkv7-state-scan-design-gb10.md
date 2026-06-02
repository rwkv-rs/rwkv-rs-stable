# WKV7 State Scan Design GB10

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-wkv7-state-scan-design-gb10-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits plus retained/reverted kernel tuning changes. This attempt owns this design note first; no WKV7 code edit is allowed until the algorithm contract and validation plan are explicit.
- Prior-note search commands:
  - `rg -n "WKV7|wkv7|row_tile|time split|chunk|state handoff|scan|row-tile|pretrain_forward_output" .agents/notes/kernel-tuning crates/rwkv-nn/src/kernels/train/time_mixer/wkv7 -S`
  - `rg -n "key_prepare|warps_per_cube|row_tile|WKV7|gated_readout|LayerNorm|ordered-256|channel_mixer|lm_head|LocalTuner|remote|10\\.100\\.1\\.253" .agents/notes/kernel-tuning crates/rwkv-nn/src/kernels/train -S`
- Matched prior evidence:
  - Current post-contract GB10 profile has `wkv7_pretrain_forward_output_kernel_f_bf16` as the largest project-owned non-matmul surface: `22.034 ms / 36`, `grid=(12,16,1)`, `block=(64,1,1)`, regs `108`, dynamic smem `1536`.
  - `2026-05-16-wkv7-row-tile16-forced.md` closed row-tile-only tuning; forced `row_tile=16` passed activation but worsened timing.
  - `2026-05-16-wkv7-remote-launch-design.md` states a real WKV7 improvement needs time/chunk split, state handoff, or scan-style composition rather than another row-tile retry.
  - Other current large project-owned surfaces are duplicate-closed: channel-mixer forced Cube/Burn/reference/fusion, mix6 line-size/axis retry, lm-head target-logit/atomic/online-softmax/prune, GatedReadout warp32/row-pack, key-prepare warps.
- Machine/GPU: design target is remote `caizus@10.100.1.253`, `NVIDIA GB10`, compute capability `12.1`; do not use local GPU.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, heads `12`, head size `64`, chunks according to current WKV7 kernel contract.
- Kernel/stage: `wkv7_pretrain_forward_output_kernel`.
- Hypothesis: WKV7 performance cannot be materially improved by changing only row tile. The plausible next implementation is a time-chunked recurrence with explicit state handoff or an associative scan/composition over chunk summaries. This branch first derives whether the recurrence has an associative composition form and what tensors would be needed for trace-backed validation.
- Candidate parameters: none yet. Potential future parameters may include time chunk length, per-chunk state summary layout, row tile, and whether output generation uses one or two passes.
- Expected keep/revert boundary: keep this design note if it identifies a concrete, testable implementation plan or proves the algorithm is too large/risky for the current tuning pass. Open a separate implementation branch before any WKV7 code change.
- Next command: inspect current WKV7 forward source/kernel and prior WKV7 notes to derive the recurrence/state contract.

## Source And Prior Evidence

- Commands: inspected `wkv7/forward.rs`, `wkv7/kernel.rs`, `wkv7/io.rs`, module call sites, and prior WKV7 notes.
- Current output path:
  - `fused_wkv7_pretrain(...)` calls `wkv7_pretrain_output(...)`, not the saved snapshot path.
  - `wkv7_pretrain_output_with_row_tile(...)` launches `wkv7_pretrain_forward_output_kernel` with candidate `row_tile` values `[16,32,64]`.
  - Current GB10 profile shows the tuner-selected `row_tile=64`: `grid=(12,16,1)`, `block=(64,1,1)`.
- Current kernel mapping:
  - One cube owns one `(head, batch, row_tile)` group.
  - For `row_tile=64`, one cube owns all 64 rows for a `(batch, head)` pair.
  - Each active unit owns one state row and stores an `Array<f32>(64)` in registers.
  - For each of `context_len=512` steps, each active row loads shared per-head inputs, computes `state_replacement = dot(state_row, removal_key)`, updates every state column, and emits one output element.
- Existing non-output WKV7 paths:
  - `wkv7_pretrain_forward_kernel` and `wkv7_state_forward_kernel` keep the full state matrix in shared memory and optionally write snapshots / state replacement for backward/statepass.
  - Those paths still launch one cube per `(batch, head)` and one unit per row; they do not parallelize a row's 64-column dot/update work.
- Prior closed result:
  - Forced `row_tile=16` increased cube count but repeated shared input loads and regressed end-to-end timing.
  - Therefore row-tile-only tuning remains closed.

## Recurrence Contract

For one `(batch, head, row)` and time `t`, with column index `c`:

- `state_replacement_t[row] = sum_j state_t[row,j] * removal_t[j]`
- `state_{t+1}[row,c] = state_t[row,c] * decay_t[c] + state_replacement_t[row] * replacement_t[c] + value_t[row] * replacement_key_t[c]`
- `output_t[row] = sum_c state_{t+1}[row,c] * receptance_t[c]`

Equivalently, for each row vector `s`:

- `s' = s * (diag(decay) + outer(removal, replacement)) + value[row] * replacement_key`

This is affine in the incoming row state. A chunk summary can be composed, but the summary is large:

- linear transform `A_chunk`: `64 x 64`, shared by rows for a `(batch, head, chunk)`;
- additive state `B_chunk`: `64 x 64`, row-dependent because `value[row]` is row-dependent;
- plus output generation still needs the prefix state at each time step.

For `B=16`, heads `12`, `T=512`, `chunk_len=16`, there are `16*12*32=6144` chunk summaries. Storing `A_chunk` alone is `6144*4096` elements; storing `B_chunk` doubles that. Even BF16 summaries are tens of MiB, and f32 summaries are about twice that, before the second output pass. This makes a full scan/state-handoff implementation a separate algorithmic project rather than the next small tuning patch.

## Smaller Candidate

A materially different but still local implementation candidate is not a time scan; it is an intra-block shared-state kernel:

- One cube still owns one `(batch, head)` pair.
- Store the full `64 x 64` state matrix in shared memory.
- Use multiple lanes per row, for example `lanes_per_row=4` with `block=256` or `lanes_per_row=8` with `block=512`.
- Each row's lanes split the 64 columns, reduce partial `state_replacement` and partial `output` through shared memory, then cooperatively update the row state.
- Candidate parameters: `lanes_per_row` / block size, with `num_warps = block_size / 32`; key must include runtime, hardware fingerprint, dtype, `B`, `T`, rows, `d_model`, `head_size`, `chunk_len`, block size, alias/in-place, and deterministic boundary.
- Expected tradeoff: fewer serial column operations per lane and lower register pressure, at the cost of much more shared memory, more barriers, and per-time reductions. This is a real implementation candidate distinct from row-tile retry.

## Design Decision

- Do not implement chunk-scan/state-handoff in the current tuning pass; the memory footprint and two-pass output requirement make it too large without a trace-backed reference and a separate acceptance plan.
- Next implementation branch should prototype the intra-block shared-state WKV7 output kernel with `lanes_per_row=4` and possibly `8`, then compare against the current register-row kernel on remote GB10.
- Keep the existing row-tile tuner intact until the new shared-state candidate proves activation-safe and faster.
