# Local Acceptance Rerun Plan

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-local-acceptance-rerun-plan-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout carries broad unrelated workspace edits plus accumulated kernel-tuning source, skill, and note changes. This attempt owns only this runbook note.
- User constraint: do not run local GPU timing now because the user may be using the local GPU. This plan is for the next local-idle window.
- Scope: runbook only. No kernel source edit, cache clear, compile, benchmark, profiler, or local CUDA workload in this branch.
- Prior search commands:
  - `rg -n "local acceptance|local rerun|local GPU|本机|regenerated-baseline|LayerNorm cache|ordered-256|speedup=2\\.44|speedup=0\\.77|cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md -S`
  - `find target/rwkv-test -maxdepth 1 -type f \\( -name '*local*compare*.log' -o -name '*acceptance*.log' -o -name '*local*.sqlite' \\) 2>/dev/null | sort | tail -50`
  - `find target/autotune -maxdepth 4 -type f 2>/dev/null | sort | sed -n '1,120p'`
- Matched evidence:
  - `2026-05-16-wire-time-mixer-gates-gb10.md`: older local post-gate compare passed activation but total speedup was `0.77x`.
  - `2026-05-16-layernorm-ordered256-post-gates.md`: ordered-256 was reverted after local activation failure; the reverted local compare recovered activation and had total speedup `0.87x`.
  - `2026-05-16-current-goal-audit-after-r9.md` and completion audits mention a local regenerated-baseline total speedup `2.44x`, but row-level timing/provenance remained incomplete.
  - Active local `target/autotune` contains moved-aside rejected LayerNorm caches:
    - `layer-norm-forward.json.log.rejected-ordered256-20260516`
    - `layer-norm-forward.json.log.rejected-ordered256-reconstruct-20260516`
    and an active `layer-norm-forward.json.log` that must be inspected before a local acceptance run.
  - Current local profiler artifact visible without rerun: `target/rwkv-test/nsys-local-post-gates.sqlite`.
- Machine/GPU: local CUDA machine, only when the user confirms it is idle or asks to run despite possible interference.
- Acceptance purpose: close the explicit objective requirement that local timing speedup is `>1.0` with activation passing, under a clean baseline/provenance boundary.

## Local Idle Preflight

Run only after the user says the local GPU is free:

```bash
nvidia-smi
```

Do not continue if another workload is using the GPU heavily.

Confirm active source branch and record dirty-tree status:

```bash
git branch --show-current
git status --short
```

Inspect the active LayerNorm autotune cache before any compare. The active cache must not reference removed ordered-256 candidates:

```bash
python3 - <<'PY'
from pathlib import Path
path = Path("target/autotune/0.10.0/device-0-0-cuda/rwkv_nn-kernels-train-layer_norm-forward-layer-norm-forward.json.log")
print(f"exists={path.exists()} path={path}")
if path.exists():
    text = path.read_text(errors="replace")
    bad = [needle for needle in ["ordered", "block_256_d768_ordered"] if needle in text]
    print(f"bad_markers={bad}")
    print(text[-2000:])
PY
```

If `bad_markers` is non-empty, move aside only the active LayerNorm cache and record the move in the attempt note before rerunning. Do not delete unrelated autotune caches.

## Standard Local Acceptance Run

This is the primary local gate. It uses the repository's default local baseline path selected by `compare-rwkv-nn`, and must be labeled as the canonical local acceptance attempt:

```bash
cargo run --release -p rwkv-test --features cuda -- \
  compare-rwkv-nn \
  --color never \
  --repeat 3 \
  --warmup 1 \
  | tee target/rwkv-test/local-acceptance-after-guards-compare.log
```

Pass criteria:

- `activation_summary` must pass with no failures.
- `timing_summary ... speedup` must be greater than `1.0x`.
- Any timing row with speedup `< 1.0` must be listed and classified. Do not hide row failures behind total speedup.
- If the result is below `1.0x`, query or run a short local `nsys` only after opening a fresh branch/note for the next hypothesis.

## Optional Baseline-Provenance Diagnostic

Use this only if the standard local gate fails but there is evidence the checked-in/default baseline is stale or mismatched. Label it as diagnostic, not final acceptance:

```bash
# Replace <LOCAL_REGENERATED_BASELINE> with the explicitly recorded local regenerated trace path.
cargo run --release -p rwkv-test --features cuda -- \
  compare-rwkv-nn \
  --color never \
  --baseline <LOCAL_REGENERATED_BASELINE> \
  --repeat 3 \
  --warmup 1 \
  | tee target/rwkv-test/local-regenerated-baseline-after-guards-compare.log
```

This diagnostic can explain baseline drift, but it does not replace the standard local gate unless the project explicitly decides the regenerated baseline is the canonical local baseline.

## Keep/Stop Boundary

- Keep this note as the local acceptance runbook while the GPU is unavailable.
- Do not run the local commands from this branch without explicit local-idle confirmation.
- When the local GPU is available, open a fresh `kernel-tuning-local-acceptance-after-guards-<date>` branch, copy this plan into that attempt note, record the exact cache state, then run the standard local acceptance command.
