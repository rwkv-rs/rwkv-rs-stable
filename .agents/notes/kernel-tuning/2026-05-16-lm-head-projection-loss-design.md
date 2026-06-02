# LM Head Projection-Loss Design

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-lm-head-projection-loss-design-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits and accumulated kernel-tuning changes. This attempt owns only feasibility/design notes unless a later entry explicitly records code changes.
- User constraint: use `10.100.1.253` for GPU validation; do not use local GPU timing.
- Prior search commands:
  - `rtk rg -n "lm_head/projection|projection|matmul|Cubek|lm_head.*0\\.98|0\\.98x|target-logit|online|atomic|prune" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md /root/.codex/memories/MEMORY.md -S`
  - `rtk rg -n "lm_head|projection|trace_.*lm|linear|matmul" crates/rwkv-nn/src crates/rwkv-test/src -S`
  - `rtk sed -n '1,260p' crates/rwkv-test/src/rwkv_nn_trace/writer.rs`
- Branch creation:
  - `rtk git switch -c kernel-tuning-lm-head-projection-loss-design-20260516`
  - Result: switched to new branch.
- Matched prior evidence:
  - R9 remote compare keeps activation passing and total speedup above `1.0`, but `lm_head/projection` remains near parity/slightly slow (`~0.98-0.99x`).
  - `lm_head/projection` is a Cubek/TMA BF16 matmul, not the project-owned row-loss kernel.
  - Prior small `lm_head_l2wrap_ce` row-kernel variants are duplicate-guarded: target-logit, atomic-loss, online-softmax, and prune/row variants either did not improve or were reverted.
  - Prior attribution says a true projection+loss fusion needs a new API/operator boundary that preserves tensor-core matmul quality; a naive row-dot custom CE kernel is likely slower.
  - Remote `ncu` counters are blocked, so this branch can use source/API inspection and existing `nsys` launch metadata first.
- Shape/dtype target: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, vocab `65536`, remote GB10.
- Question: can `lm_head/projection` and `lm_head_l2wrap_ce` be fused or retimed in a way that preserves Cubek/TMA tensor-core matmul quality and still avoids materializing/then rereading full logits for loss?
- Expected keep/revert boundary:
  - Keep this note if it identifies a concrete feasible operator boundary or proves the design is currently too broad/risky.
  - Do not implement a naive custom projection-loss kernel unless the design first explains how it preserves GEMM performance and trace correctness.
  - Any later code branch must pass local compile, remote compile, remote activation, and remote compare with total speedup above `1.0`; projection row must improve only if it remains part of the timing contract.
- Next command: inspect `RwkvLM::forward`, `Linear`/matmul call boundaries, current lm-head loss primitive interface, and available Burn/CubeCL matmul extension hooks.

## Source Boundary Inspection

- Commands:
  - `rtk sed -n '1,180p' crates/rwkv-nn/src/models/lm.rs`
  - `rtk sed -n '1,260p' crates/rwkv-nn/src/kernels/train/lm_head_l2wrap_ce/mod.rs`
  - `rtk sed -n '1,280p' crates/rwkv-nn/src/kernels/train/lm_head_l2wrap_ce/forward.rs`
  - `rtk sed -n '1,240p' crates/rwkv-nn/src/kernels/train/lm_head_l2wrap_ce/kernel.rs`
  - `rtk rg -n "pub struct Linear|impl.*Linear|fn forward\\(|float_matmul|MatmulStrategy|matmul.*tune|epilogue|fusion" ~/.cargo/registry/src ~/.cargo/git/checkouts -S`
- Result:
  - `RwkvLM::forward` calls `forward_logits`, which always materializes `self.unembed.forward(embedded_context_normalized)` before `lm_head_l2wrap_ce(logits, targets)`.
  - `RwkvLM::forward_logits` is public model behavior and still returns raw logits; any loss-only fused path must avoid breaking that public API.
  - `TraceWriter::forward` separately times `lm_head/projection` by calling `model.unembed.forward(embedded_context)`, then times `loss/l2wrap_cross_entropy` by calling `lm_head_l2wrap_ce(logits, targets)`.
  - Current loss primitive interface accepts materialized logits plus targets only; it cannot avoid the projection output tensor.
  - The forward loss row kernel scans the full materialized vocab row for max/sum/target and writes per-row losses. Backward writes a full logits gradient, so a true fused training operator would also need a custom backward that returns hidden-state and unembed-weight gradients without materializing logits gradients.
  - Broad registry grep found Burn public `float_matmul` and Linear module paths, but the result was too broad to identify a supported matmul epilogue hook.
- Interpretation:
  - A forward-only projection-loss fusion would be incomplete for training unless the backward contract is also redesigned.
  - A naive custom row-dot CE kernel would replace a tensor-core TMA matmul with scalar row-dot work across `8192 * 65536 * 768`, which is not a plausible speedup without a GEMM-quality implementation.
  - The next command should inspect the exact Burn/CubeCL matmul implementation and extension points narrowly, not broad grep output.

## Narrow Matmul Extension Inspection

- Commands:
  - `rtk rg --files ~/.cargo/registry/src ~/.cargo/git/checkouts | rtk rg 'burn-cubecl.*(matmul|linear|fusion)|cubek-matmul.*(tune_key|vectorization|write|base|launch)|burn-core.*/linear.rs|burn-backend.*/linear.rs'`
  - `rtk rg -n "epilogue|EventListener|GlobalWrite|GlobalWriter|write_output|MatmulStrategy|Tma|TMA|tune_key|Launch|PlaneWriter|OutWriter|acc_size|num_ops|num_out_buffers" ~/.cargo/registry/src ~/.cargo/git/checkouts -g '*.rs'`
  - `rtk sed -n '1,130p' .../burn-backend-0.21.0-pre.4/src/backend/ops/modules/linear.rs`
  - `rtk sed -n '1,240p' .../burn-cubecl-0.21.0-pre.4/src/kernel/matmul/base.rs`
  - `rtk sed -n '1,620p' .../burn-cubecl-0.21.0-pre.4/src/kernel/matmul/tune/base.rs`
  - `rtk sed -n '1,220p' .../cubek-matmul-0.2.0-pre.4/src/components/global/write/{base,event,unit}.rs`
  - `rtk sed -n '1,520p' .../burn-cubecl-fusion-0.21.0-pre.4/src/optim/matmul/tune.rs`
- Result:
  - Burn `linear` is a thin wrapper around `B::float_matmul(x, weight)` and optional bias add. There is no model-level hook to attach cross-entropy/l2wrap before the logits tensor is produced.
  - CubeCL ordinary matmul calls `cubek::matmul::launch::launch_ref(strategy, ..., out.binding(), ...)`; the only output contract is a full output tensor binding.
  - Ordinary matmul autotune includes TMA strategies such as `SimpleTma*` and `SpecializedTma*`; the kept `lm_head/projection` timing row is therefore already in the optimized Cubek/TMA family.
  - Cubek has internal `GlobalWriter` / `WriteEventListener` traits, but the writer is selected inside Cubek matmul internals. The public Burn/CubeCL matmul launch path does not expose a project-owned writer/epilogue parameter.
  - Burn fused matmul has a separate `FusedMatmulAutotuneKey` keyed by `num_out_buffers` and `num_ops`, but its candidate list covers fallback, vector-matrix, unit, and accelerated fused selectors. It does not register TMA fused candidates, matching the earlier channel-mixer fusion negative evidence.
- Decision:
  - Do not implement projection+loss fusion in this branch. A correct implementation would require a new upstream-style Cubek writer/epilogue or a custom GEMM-quality fused operator plus matching backward contract for hidden-state and unembed-weight gradients.
  - Keep current `lm_head/projection` as an exposed timing row and keep the existing `lm_head_l2wrap_ce` row-loss path with the kept direct target-logit improvement.
  - Next branch should target a different non-duplicate project-owned surface on `10.100.1.253`, with WKV7 state-scan/time-split design as the most plausible remaining large custom-kernel family.

## Duplicate Surface Check After Design Decision

- Commands:
  - `rtk rg -n "WKV7|wkv7|state-scan|state scan|segment|row_tile|shared-lanes|lowrank|recompute|register|regs|time-split|time split|10\\.100\\.1\\.253" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md crates/rwkv-nn/src/kernels/train/time_mixer/wkv7 -S`
  - `rtk rg -n "weight_decay|learning_rate_gate|gated_readout|mix6|value_residual|key_prepare|shared|warp|line_size|vector|GB10|10\\.100\\.1\\.253|selected|negative|revert|Decision" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md -S`
  - `rtk rg -n "Mix6|mix6.*line|mix6.*axis|rwkv7-time-mixer-mix6-forward|line_size_1|selected.*mix6|closed because GB10 tuner" .agents/notes/kernel-tuning -S`
- Result:
  - WKV7 row-tile, shared-lanes, segment recompute, and state-scan design paths are already covered and rejected/closed for this shape unless a new algorithm removes the global f32 transform tensors and high-register segment scan.
  - Weight-decay transform is already kept as a real GB10 optimization replacing the Burn softplus chain.
  - Value-residual vector-axis is already kept as a dispatch-correctness fix, with neutral/slightly negative targeted profiler evidence.
  - Key-prepare warps-per-cube, GatedReadout row-pack, lm-head row-kernel variants, channel-mixer matmul/fusion/line-size variants, and Mix6 line-size/axis retry are closed.
  - Mix6 is not a value-residual-style vector-axis bug: its remote key has `max_line_size=8`, and GB10 autotune selected `line_size_1` as the fastest candidate.
- Next edit: update `.agents/skills/kernel-tuning/SKILL.md` with the projection+loss design guard from this branch, then sync the note and skill to `10.100.1.253`.

## Skill Guard Update

- Edited `.agents/skills/kernel-tuning/SKILL.md`.
- Added known GB10 design guard:
  - keep `lm_head/projection` visible in timing;
  - do not implement projection+loss fusion as a naive row-dot or post-logits tweak;
  - current Burn/CubeCL/Cubek public path has no project-owned TMA epilogue hook;
  - a correct fusion requires an upstream-style Cubek/TMA epilogue or GEMM-quality fused operator plus matching backward contract.
- Keep/revert state: keep the skill update as duplicate-experiment prevention.
- Next command: sync this note and updated skill to `/home/caizus/Projects/Packages/rwkv-rs-stable` on `10.100.1.253`.

## Remote Note And Skill Sync

- Command: scoped `rsync -azR` of `.agents/skills/kernel-tuning/SKILL.md` and this note to `caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable/`.
- Result: sync completed successfully.
- Final decision for this branch: no kernel code change. Keep the design note and skill update; do not implement `lm_head/projection` + loss fusion until the operator boundary provides a TMA-quality epilogue or full fused backward design.
