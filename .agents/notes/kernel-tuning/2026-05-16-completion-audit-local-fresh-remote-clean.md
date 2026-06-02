# Completion Audit: Local Fresh Baseline And Remote Clean

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-completion-audit-local-fresh-remote-clean-20260516` in the existing dirty workspace.
- Dirty-tree constraint: this audit runs in the same broad dirty workspace as the current kernel-tuning work. It is evidence-only and should not edit kernels.
- Prior-note/source search command:
  - `rg -n "speedup=2\\.44|speedup=2\\.10|remote clean|completion|current-goal|LocalTuner|autotune key|ncu|ordered-256|channel_mixer" .agents/notes/kernel-tuning /root/.codex/memories/MEMORY.md -S`
- Matched evidence:
  - `2026-05-16-channel-mixer-edge-rerun-after-projection-timing.md`: remote `10.100.1.253` clean gate, activation `54/54`, timing `76/76`, total `2.10x`.
  - `2026-05-16-local-regenerated-baseline.md`: local fresh baseline activation `54/54`, total `2.44x`, but row-level timing still `22/76` pass with channel mixer `0.81x`.
  - `2026-05-16-cubecl-local-tuner-findings.md` and `2026-05-16-localtuner-mechanism-skill-update.md`: LocalTuner cache/miss/warmup/candidate/accuracy-guard behavior documented and reflected in skill rules.
  - `2026-05-16-autotune-key-audit.md`, `2026-05-16-forward-elementwise-full-hardware-key.md`, `2026-05-16-backward-autotune-hardware-key.md`, `2026-05-16-layernorm-runtime-dispatch.md`: hardware/shape key and runtime dispatch work exists, but needs current-state coverage check before completion.
  - `2026-05-16-layernorm-ordered256-reconstruct.md` and skill update: ordered-256 precision issue was analyzed as implementation/boundary failure, not accepted as math inevitability.
  - `2026-05-16-channel-mixer-local-regenerated-baseline.md`: local fresh-baseline channel mixer remains a meaningful unresolved row-level gap; current branch kept evidence only.
- Machine/GPU: local RTX 5090 plus remote `10.100.1.253` GB10 evidence from notes.
- Shape/dtype: CUDA BF16 `B=16,T=512,D=768`, rows `8192`.
- Objective restatement:
  1. Use branch-per-attempt kernel tuning with notes, preserving failed branches and reverting rejected code.
  2. Design kernel implementation choice by hardware and shape, adding relevant autotune key dimensions: backend/runtime, GPU arch/capability, dtype, `d_model`, rows, block size/warps/vector width, alias/in-place, deterministic boundary.
  3. LayerNorm must not be a universal hard-coded constant; remote GB10 and local RTX 5090 need hardware/shape-specific behavior with trace-backed accuracy.
  4. Understand and document Burn/CubeCL `LocalTuner::execute`: key, cache, warmup/profiling boundary, candidate capability, accuracy guard limitations, hot-path lookup cost.
  5. Investigate CUDA kernel design with profiler evidence: occupancy, register/shared memory pressure, coalescing, BF16 reduction error, and ncu/nsys metrics.
  6. Achieve speedup `>1.0` on local and `10.100.1.253`, and carefully diagnose precision issues rather than reverting blindly.
- Next command: inspect the relevant note files and live source/cache artifacts for each audit row before deciding whether completion is allowed.
