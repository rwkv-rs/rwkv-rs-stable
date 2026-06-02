# Kernel Tuning Guardrails

- Date: 2026-05-16
- Branch/worktree: `kernel-tuning-hw-shape-dispatch` in the existing dirty workspace; this note records process rules and the branch boundary for continued tuning.
- Reason: the local tuning loop repeated an already-known residual-add experiment and mixed experiments into the shared dirty tree.
- Decision: future kernel tuning must start from a dedicated branch or worktree, search prior notes/memory before rerunning an A/B idea, and record every kernel change/profiler run/revert under `.agents/notes/kernel-tuning/`.
- Recorded duplicate-experiment guard: `residual_add` versus Burn addition already has local A/B evidence for the CUDA BF16 `rwkv_lm` `B=16,T=512,D=768` compare run. Before touching that path again, cite the concrete timing notes and document changed hardware, shape, dtype, baseline provenance, implementation, or measurement method.
- Known negative result to preserve: switching channel mixer from the custom path to the Burn reference path was worse for the same CUDA BF16 trace; keep the custom channel mixer path as the default for that case.
- Known local negative result: BF16 LayerNorm for `D=768` with deterministic minimum block size `512` reproduced activation drift on this machine (`value_from_first_cell` and `lm_head/embedded_context` failed). Keep the local deterministic boundary at `1024`; the older `256` drift remains a separate negative result.

## 2026-05-16 Autotune Mechanism Read

- Branch/worktree: `kernel-tuning-hw-shape-dispatch`
- Files read: `cubecl-runtime-0.10.0-pre.4/src/tune/{local.rs,tuner.rs,tune_benchmark.rs,tune_cache.rs,operation.rs,base.rs}`.
- `LocalTuner::init` caches the `TunableSet` per process by initializer `TypeId`; the candidate set is not rebuilt in the hot path after first init.
- `LocalTuner::execute` generates the key from real inputs, checks the in-memory/persistent cache, and executes the cached fastest candidate directly on cache hit.
- Cache miss triggers tuning: 3 warmup profiled executions, then 10 profiled samples per viable candidate. Native targets block until tuning completes; wasm can return `Pending` and fall back.
- Persistent cache is keyed by `(autotune key, tunable-name checksum)`. Candidate parameters such as block size or vector width are represented by tunable names and group priority; changing candidate names invalidates stale cache via checksum.
- `autotune-checks` is an optional candidate-equivalence check among tunables. It is not a production trace accuracy guard, so BF16-sensitive choices still need trace-backed correctness validation before accepting a faster candidate.
- 2026-05-16 code change: expanded forward elementwise autotune keys for channel mixer, learning-rate gate, and value-residual gate with runtime, hardware capability, rows, embedded dimension, max vector width, in-place/alias state, and deterministic flag.

## 2026-05-16 Branch Per Attempt Rule

- Branch/worktree: `kernel-tuning-skill-branch-rules-20260516` in the existing dirty workspace.
- File changed: `.agents/skills/kernel-tuning/SKILL.md`.
- Reason: the user clarified that kernel tuning iterations should use git branches/trees for rollback, every materially new attempt should start on its own branch, and failed branches should remain as evidence.
- Decision: the skill now requires a separate branch or worktree for each materially new tuning attempt, preserving failed branches, and treating poor timing as a profiling/debugging signal before deciding that an idea is invalid.

## 2026-05-16 Required Prior-Note Preflight

- Branch/worktree: `kernel-tuning-channel-mixer-ncu-20260516` in the existing dirty workspace.
- File changed: `.agents/skills/kernel-tuning/SKILL.md`.
- Reason: the residual-add negative result was already recorded, but the process still allowed the same A/B idea to be rerun. The skill needs to force a prior-note and memory search before every new tuning branch, code change, profiler run, or benchmark run.
- Prior evidence checked: `rg -n "残差|residual|drift|branch|分支|note|笔记|autotune|ncu" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md` found the existing residual-add negative result and branch/notes rules.
- Decision: the skill now treats the prior-note search as a required preflight. If an idea already exists in notes, the new note must record what changed in hardware, shape, dtype, baseline provenance, or implementation before the command is run.

## 2026-05-16 Stricter Branch And Note Rule

- Branch/worktree: `kernel-tuning-skill-note-branch-rules-20260516` in the existing dirty workspace.
- File changed: `.agents/skills/kernel-tuning/SKILL.md`.
- Prior evidence checked: `rg -n "kernel|residual|残差|layer_norm|notes|branch|分支" /root/.codex/memories/MEMORY.md .agents/skills .agents/notes` found the residual-add negative result, the existing branch/note guardrails, and the earlier repeated-experiment warning.
- Reason: the user clarified that kernel tuning should use a new branch/worktree per code change or materially different experiment, and every change/run/revert must have a note. The residual-add retry is the concrete failure mode this rule must prevent.
- Decision: the skill now says every kernel tuning code change needs its own branch/worktree before editing, every profiler/benchmark/revert must be noted, and an already-recorded idea must stop unless a note first records what changed in hardware, shape, dtype, baseline provenance, implementation, or measurement method.

## 2026-05-16 Hard Preflight Gate

- Branch/worktree: `kernel-tuning-skill-hard-preflight-20260516` in the existing dirty workspace.
- File changed: `.agents/skills/kernel-tuning/SKILL.md`.
- Prior evidence checked: `rg -n "residual|残差|value_residual|kernel-tuning|new branch|branch|note|笔记" .agents/skills .agents/notes -S` found the recorded residual-add negative result, previous branch/note rules, and the earlier warning that the residual experiment had already been repeated.
- Reason: the branch/note rules existed, but they still read too much like process guidance. The user clarified that each tuning attempt must use a new branch and that every change/run/revert needs a note, specifically to prevent rerunning an old residual-add negative result.
- Decision: the skill now frames branch creation, note creation/update, and prior-note search as a hard preflight gate before any kernel edit, autotune change, profiler run, or benchmark run. If an idea is already covered by notes, the next branch must document the changed hardware, shape, dtype, baseline provenance, implementation, or measurement method before running anything.

## 2026-05-16 Explicit Experiment Unit

- Branch/worktree: `kernel-tuning-skill-discipline-20260516` in the existing dirty workspace.
- File changed: `.agents/skills/kernel-tuning/SKILL.md`.
- Dirty-tree constraint: this checkout already carries many unrelated uncommitted workspace changes and previous kernel-tuning notes; this process-rule attempt only changes the kernel-tuning skill and this guardrail note.
- Prior evidence checked: `rg -n "branch|note|residual|残差|LayerNorm|layernorm|tuning|speedup|回退|negative" .agents/skills .agents/notes/kernel-tuning -g '*.md'` found the residual-add negative result, the branch/notes rules, LayerNorm drift notes, and the existing hard-preflight entry.
- Reason: the user reiterated that kernel tuning must use new branches and notes for each change, and pointed out that the recorded residual-add negative result had still been rerun. The skill needs an executable checklist rather than only high-level guidance.
- Decision: the skill now defines the branch/worktree plus note as the experiment unit, requires a four-step hard preflight before edits/runs/reruns, includes binary provenance in notes when profiling `target/release/*`, and says stale-binary or wrong-boundary profiler data must be marked invalid before rerunning.

## 2026-05-16 Preflight Clarity And Residual Guard

- Branch/worktree: `kernel-tuning-skill-preflight-clarity-20260516` in the existing dirty workspace.
- File changed: `.agents/skills/kernel-tuning/SKILL.md` and this note.
- Dirty-tree constraint: the checkout already contains broad unrelated uncommitted changes; this attempt is limited to the project skill and this process note.
- Prior evidence checked:
  - `rg -n "residual_add|Burn addition|Burn add|trace_residual_add|custom residual|plain Burn|残差" .agents/notes/kernel-tuning -g '*.md'`
  - `sed -n '1,120p' /root/.codex/memories/extensions/ad_hoc/notes/2026-05-15T21-10-06-rwkv-nn-timing-optimization-iterations.md`
  - `sed -n '1,120p' /root/.codex/memories/extensions/ad_hoc/notes/2026-05-15T21-26-58-rwkv-nn-sync-ablation.md`
- Matched evidence: the ad-hoc timing note records that switching the trace writer back to plain Burn tensor add was worse than the custom residual-add path for the CUDA BF16 `rwkv_lm` `B=16,T=512,D=768` comparison. The sync-ablation note records that alias/in-place reuse for `residual_add` and `channel_mixer_relu_square` improved the previous stable run from about `0.69x` to about `0.77x` but still needed validation.
- Reason: the skill had accumulated repeated branch/note bullets and one residual sentence that was too easy to read as a fresh tuning instruction. The user pointed out that the residual experiment had already been recorded and then rerun.
- Decision: the Branch And Notes section is now a single hard preflight gate. It requires a dedicated branch/worktree plus attempt note before any edit, autotune metadata change, build/profile/benchmark rerun, or revert. The residual-add/Burn-add A/B is now framed as a duplicate-experiment guard: cite the existing notes and document changed conditions before touching that path again.

## 2026-05-16 Attempt Ledger Gate

- Branch/worktree: `kernel-tuning-skill-attempt-ledger-20260516` in the existing dirty workspace.
- File changed: `.agents/skills/kernel-tuning/SKILL.md` and this note.
- Dirty-tree constraint: the checkout already contains broad unrelated uncommitted changes from previous workspace work. This attempt only changes the project skill and this guardrail note.
- Prior evidence checked:
  - `rg -n "kernel-tuning|residual|残差|branch|note|rwkv-nn timing|sync-ablation" /root/.codex/memories/MEMORY.md`
  - `rg -n "residual_add|Burn addition|Burn add|plain Burn|custom residual|残差" .agents/notes/kernel-tuning /root/.codex/memories/extensions/ad_hoc/notes -S`
  - `sed -n '1,130p' .agents/notes/kernel-tuning/2026-05-16-skill-guardrails.md`
- Matched evidence: memory and notes record that custom residual-add beat plain Burn add on 2026-05-15, alias/in-place residual reuse was already part of a separate sync ablation, and `2026-05-16-wire-custom-residual-add.md` later reran residual wiring under a changed implementation boundary and reverted it after negative timing.
- Reason: the user clarified again that kernel tuning must use a new branch and notes for every change, and specifically called out that the recorded residual result had been rerun.
- Decision: the skill now states "no branch plus prewritten note means no tuning command", requires stopping when prior notes match the same boundary, and names both residual notes as the duplicate-experiment guard before any future `residual_add` or trace-writer residual work.

## 2026-05-16 Skill Trigger Enforcement

- Branch/worktree: `kernel-tuning-skill-preflight-enforcement-20260516` in the existing dirty workspace.
- File changed: `.agents/skills/kernel-tuning/SKILL.md` and this note.
- Dirty-tree constraint: the checkout already carries broad unrelated uncommitted changes. This attempt only tightens the project skill metadata and process rules.
- Prior evidence checked:
  - `rg -n "kernel|tuning|残差|residual|branch|notes|layernorm|LayerNorm|channel_mixer" /root/.codex/memories/MEMORY.md`
  - `rg -n "residual|残差|Burn-add|custom-residual|branch/worktree|preflight|duplicate|重复" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md`
  - `nl -ba .agents/skills/kernel-tuning/SKILL.md | sed -n '1,120p'`
- Matched evidence: the skill already requires a branch/worktree plus note as the tuning unit, and the notes record residual-add/Burn-add duplicate guards. The frontmatter trigger text did not yet surface that hard preflight rule, so future agents could load the skill without immediately seeing why branch/note discipline is mandatory.
- Reason: the user clarified that kernel tuning must happen through new branches and notes for every change, and called out the repeated residual experiment as the failure mode to prevent.
- Decision: tighten the skill description and Branch And Notes wording so the branch/note/preflight rule is visible at trigger time and explicitly applies to continuation after context compaction or branch changes.

## 2026-05-16 Remote And Per-Change Note Gate

- Branch/worktree: `kernel-tuning-skill-remote-note-rules-20260516` in the existing dirty workspace.
- File changed: `.agents/skills/kernel-tuning/SKILL.md` and this note.
- Dirty-tree constraint: the checkout already carries broad unrelated uncommitted changes. This attempt is limited to the project skill and this process note.
- Prior evidence checked:
  - `rg -n "kernel|LayerNorm|residual|残差|note|branch|分支" /root/.codex/memories/MEMORY.md`
  - `rg -n "residual|残差|Burn|custom-residual|plain-Burn|repeat|duplicate|重复|branch|note|10\\.100\\.1\\.253|remote" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md`
- Matched evidence: memory records the remote `10.100.1.253` timing loop as the project acceptance target, plus the local-vs-remote LayerNorm conflict. The local notes already record residual-add/Burn-add duplicate guards, the 2026-05-16 residual wiring revert, and repeated skill-rule tightening after the residual A/B was rerun.
- Reason: the user clarified that ongoing kernel tuning should use new branches for materially new attempts, and every change/run/revert must be recorded in notes. The repeated residual experiment is the concrete failure mode the skill must prevent.
- Decision: make the skill stricter about creating a fresh branch/worktree for materially new attempts, appending a note entry for every code change, benchmark/profiler run, invalid run, revert, or keep decision before the next command, and treating remote GB10 validation as the primary timing acceptance source unless the user explicitly asks for local timing.

## 2026-05-16 Strict Change Ledger Order

- Branch/worktree: `kernel-tuning-skill-change-note-discipline-20260516` in the existing dirty workspace.
- File changed: `.agents/skills/kernel-tuning/SKILL.md` and this note.
- Dirty-tree constraint: the checkout already carries broad unrelated uncommitted changes. This attempt is limited to the project skill and this process note.
- Prior evidence checked:
  - `rg -n "residual|残差|branch|分支|note|笔记|kernel tuning|调优" .agents/notes .agents/skills/kernel-tuning -S`
  - `nl -ba .agents/skills/kernel-tuning/SKILL.md | sed -n '1,130p'`
- Matched evidence: the skill already requires fresh branches, prewritten notes, duplicate-experiment checks, and a residual-add/Burn-add duplicate guard. The notes also show several previous skill-rule tightening entries caused by repeating an already recorded residual experiment.
- Reason: the user restated that kernel tuning should use a new branch form and every change must have a note; the current wording had the right rules, but the required command order should be visible as an executable ledger.
- Decision: add a concise ordered ledger rule to the skill: search notes, create branch/worktree, write the initial note, make exactly the next change or run, append its result, then proceed to the next change or run. Do not batch edits/runs and summarize later.
