# LocalTuner Mechanism Skill Update

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-localtuner-skill-mechanics-20260516` in the existing dirty checkout.
- Dirty-tree constraint: this checkout carries broad unrelated uncommitted changes plus previous kernel attempts. This attempt only updates `.agents/skills/kernel-tuning/SKILL.md` and this note.
- Prior-note search command:
  - `rg -n "LocalTuner::execute|LocalTuner|accuracy guard|warmup|measurement|cache|host-side tuner|tuner" .agents/notes/kernel-tuning -S`
- Matched prior evidence:
  - `2026-05-16-skill-guardrails.md` already read `cubecl-runtime-0.10.0-pre.4/src/tune/{local.rs,tuner.rs,tune_benchmark.rs,tune_cache.rs,operation.rs,base.rs}` and records the LocalTuner mechanics.
  - Existing evidence: `LocalTuner::init` caches `TunableSet` by initializer `TypeId`; `execute` checks generated key against memory/persistent cache and runs cached fastest candidate on hit; misses do 3 warmup profiled executions and 10 profiled samples per viable candidate; persistent cache key includes autotune key and tunable-name checksum; `autotune-checks` is not a production trace accuracy guard.
- Changed boundary: this is not a new source-code tuning candidate. It makes the existing LocalTuner mechanism evidence operational in the project skill so later branches do not confuse tuner cache behavior, warmup cost, candidate naming, or accuracy guards.
- Expected keep/revert boundary: keep if the skill gains concise, actionable LocalTuner rules without weakening branch/note discipline or duplicating old residual/channel/layernorm tuning instructions.
- Next command: update `.agents/skills/kernel-tuning/SKILL.md`.

## Skill edit

- File changed: `.agents/skills/kernel-tuning/SKILL.md`.
- Added `Burn/CubeCL Autotune Rules`:
  - distinguishes `LocalTuner::execute` cache-hit and cache-miss paths.
  - records that `LocalTuner::init` caches the `TunableSet` by initializer `TypeId`.
  - warns that cache-miss warmup/profiling is not steady-state `.time.json` evidence.
  - records persistent cache key behavior: autotune key plus tunable-name checksum.
  - states that `autotune-checks` is not a production trace accuracy guard.
  - requires profiler evidence before bypassing `LocalTuner::execute` for host-overhead claims.
  - forbids adding known-invalid BF16 numeric boundaries as candidates before hardware/deterministic gating and trace validation.
- Next command: inspect the edited section for scope and duplication.

## Inspection

- Command: `sed -n '52,105p' .agents/skills/kernel-tuning/SKILL.md`.
- Result: the new `Burn/CubeCL Autotune Rules` section is present before `Performance Rules`, and it does not alter the existing branch/note or remote-acceptance rules.
- Validation: no compile or GPU command is needed for this Markdown-only process-rule update.
- Decision: keep this skill update.
