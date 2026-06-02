# Autotune Key Coverage Audit

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-autotune-key-coverage-audit-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout already carries broad unrelated workspace edits, generated traces, and retained/reverted kernel tuning files. This attempt owns only this audit note unless a later entry explicitly names a skill or source edit.
- User constraint: debug first on remote `10.100.1.253`; do not use local GPU timing because the local GPU may be used for other work.
- Scope: read-only audit of whether current train-kernel implementation choice is keyed by backend/runtime, hardware fingerprint, dtype, rows, model shape, block/warp/vector candidate dimensions, alias/in-place behavior, and deterministic numeric boundary where relevant.
- Prior-note search commands already run:
  - `find .agents/notes/kernel-tuning -maxdepth 1 -type f -name '2026-05-16-*.md' | sort | tail -40`
  - `sed -n '1,240p' .agents/notes/kernel-tuning/2026-05-16-remote-current-r9-nsys-gb10.md`
  - `rg -n "AutotuneKey|hardware|deterministic|block_size|num_warps|line_size|vector|is_in_place|alias|runtime|Runtime|DType|dtype|rows|d_model|Compute|Hardware" crates/rwkv-nn/src/kernels/train crates/rwkv-test/src -g '*.rs'`
- Matched prior evidence:
  - `2026-05-16-remote-current-r9-nsys-gb10.md` is the current remote R9 profiler evidence. It shows the current binary passes R9 timing elsewhere and WKV7 is the only large project-owned custom kernel with clear structural underfill from launch/register metadata.
  - The kernel-tuning skill already records the LayerNorm local/remote conflict, the WKV7 closed experiment families, the residual-add duplicate guard, the `lm_head/projection` near-parity result, and the remote `ncu` permission blocker.
- Machine/GPU boundary: remote `10.100.1.253` / GB10 is the primary performance boundary. This branch does not run GPU commands.
- Shape/dtype boundary: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, heads `12`, head size `64`, vocab `65536`, unless a code path is explicitly shape-generic.
- Hypothesis: the immediate next correct step is to close the design-accounting gap, not rerun duplicate micro-kernel experiments. Current code likely has the key dimensions in most custom autotune keys, but WKV7 and several fixed-dispatch kernels need explicit classification as fixed-policy/runtime-dispatch rather than autotuned candidates.
- Expected keep/revert boundary: if the audit finds a missing key that can cause known incorrect selection, open a separate source-edit branch and note before changing code. If gaps are design/documentation gaps only, record them here and update the kernel-tuning skill with concise operational rules.
- Next command: list the concrete autotune-key structs and fixed-dispatch train kernels, then classify each against the required dimensions.

## Key Coverage Findings

- Commands:
  - `rg -n "struct .*AutotuneKey|impl AutotuneKey|static TUNER|TuneGroup|Tunable::new|TunableSet::new|struct .*Key" crates/rwkv-nn/src/kernels/train -g '*.rs'`
  - `rg -n "LocalTuner|local_tuner|AutotuneKey|TunableSet|TuneGroup|CubeHardwareFingerprint|deterministic_min_block_size|max_line_size|is_in_place|runtime:|dtype:|rows:|d_model:" crates/rwkv-nn/src/kernels/train -g '*.rs'`
  - targeted `sed` reads of the key structs and fixed dispatch sites for LayerNorm, WKV7, ChannelMixer, Mix6, LearningRateGate, ValueResidualGate, WeightDecayTransform, lm-head loss, residual add, KeyPrepare, and GatedReadout.
- Shared hardware key:
  - `CubeHardwareFingerprint` records `load_width`, `plane_size`, max cube units/dimensions, shared memory, max vector size, SM count, tensor-core count, and tensor-core min dimension.
  - This is the available project-side GPU architecture proxy. Literal CUDA compute capability is not currently exposed through the public CubeCL `HardwareProperties` used here.
- Autotuned custom kernels are broadly keyed correctly:
  - LayerNorm forward: runtime, dtype, d_model, rows, num elements, hardware fingerprint, in-place flag, deterministic flag, and deterministic minimum block size. Candidate group is `block_size`; `num_warps` is derived from `block_size / 32`.
  - WKV7 pretrain output: runtime, dtype, batch, context length, rows, d_model, num heads, head size, chunk length, hardware fingerprint, in-place flag, deterministic flag. Candidate group is `row_tile`.
  - ChannelMixer forward elementwise, Mix6 forward, LearningRateGate forward, ValueResidualGate forward, WeightDecayTransform forward: runtime, dtype, num elements, embedded dimension, rows, hardware fingerprint, max line size, in-place flag, deterministic flag. Candidate group is `line_size`.
  - ChannelMixer/Mix6/LearningRateGate/ValueResidualGate backward reducers: runtime, dtype, num elements, d_model, rows, hardware fingerprint, max line size, in-place flag, deterministic flag. Candidate names encode `line_size`, reduction `block_size`, and `bt_tile`, with validity filtered by max line size, divisibility, and max cube units.
  - lm-head/l2wrap CE forward and backward: runtime, dtype, token count, vocab size, hardware fingerprint, in-place flag, deterministic flag. Candidate group is `block_size`; `num_warps` is derived from `block_size / 32`.
  - residual add forward: runtime, dtype, num elements, rows, innermost dim, hardware fingerprint fields, max line size, lhs/rhs in-place flags, deterministic flag. Candidate group is `line_size`.
- Fixed-dispatch kernels are intentionally not autotuned in current code:
  - KeyPrepare forward uses a `head_size == 64` specialized path with fixed `HEAD64_WARPS_PER_CUBE = 4`; the generic fallback uses CubeCL elementwise cube sizing.
  - KeyPrepare backward uses CubeCL elementwise cube sizing and reduction/finalize kernels rather than a project autotune key.
  - GatedReadoutCombine forward uses fixed `HEAD_SIZE = 64`, `BLOCK_SIZE = 64`, and `NUM_WARPS = 2`.
  - WKV7 pretrain saved/state forward and WKV7 backward use fixed `cube_dim = head_size` and `cube_count = (num_heads, batch_size, 1)`.
- Interpretation:
  - The immediate LayerNorm lesson has already been applied where it matters most: BF16 block-size selection is no longer a raw hard-coded constant; the runtime key includes the GB10 deterministic exception and falls back to the conservative local boundary elsewhere.
  - There is no current evidence that a live autotuned custom kernel is missing runtime, dtype, shape, rows, vector width/line-size capability, hardware proxy, in-place/alias flag, or deterministic policy in a way that explains the remaining remote profiler surface.
  - The remaining design gap is classification of fixed-dispatch paths. Future changes to KeyPrepare, GatedReadoutCombine, or WKV7 state/backward launch constants should either stay as explicitly documented fixed policy with trace and profiler evidence, or first introduce a keyed runtime dispatch/autotune wrapper. They should not be silent constant edits.
- Decision: no kernel source edit from this audit. Update the kernel-tuning skill with a concise fixed-dispatch/autotune-key rule so future tuning does not regress into hard-coded constant changes.

## Skill Update And Validation Plan

- Edited `.agents/skills/kernel-tuning/SKILL.md` to record the current train-kernel key coverage result and the fixed-dispatch rule for KeyPrepare, GatedReadoutCombine, and WKV7 state/backward.
- Because `.agents/notes` and `.agents/skills/kernel-tuning` are untracked in this checkout, ordinary `git diff --check -- <path>` does not show content. Use `git diff --no-index --check /dev/null <file>` for the new note, and a `git diff --no-index --check` copy comparison for the skill if needed.
- Next commands:
  - `git diff --no-index --check /dev/null .agents/notes/kernel-tuning/2026-05-16-autotune-key-coverage-audit.md`
  - `git diff --no-index --check /tmp/kernel-tuning-skill-empty .agents/skills/kernel-tuning/SKILL.md`
  - scoped `rsync -azR` of this note and `.agents/skills/kernel-tuning/SKILL.md` to `/home/caizus/Projects/Packages/rwkv-rs-stable` on `10.100.1.253`.

## Validation And Sync Result

- Command: `git diff --no-index --check -- /dev/null .agents/notes/kernel-tuning/2026-05-16-autotune-key-coverage-audit.md`
- Result: passed; no whitespace errors reported.
- Command: `git diff --no-index --check -- /dev/null .agents/skills/kernel-tuning/SKILL.md`
- Result: passed; no whitespace errors reported.
- Remote sync command: `rsync -azR .agents/notes/kernel-tuning/2026-05-16-autotune-key-coverage-audit.md .agents/skills/kernel-tuning/SKILL.md caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable/`
- Result: failed with SSH authentication error: `Permission denied (publickey,password)`.
- Decision: keep the local note and skill update. Do not run remote kernel commands or retry privileged/interactive authentication in this branch.

## Remote Auth Retry Plan

- Follow-up search command: `rg -n "10\\.100\\.1\\.253|spark-35ac|rsync|ssh .*10\\.100\\.1\\.253|caizus@10\\.100\\.1\\.253|/home/caizus/Projects/Packages/rwkv-rs-stable" .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md -S`
- Matched prior evidence: multiple older notes used explicit-key SSH with `~/.ssh/id_ed25519_dgx_spark_windows` and `BatchMode=yes`; plain SSH without the key has failed before.
- Next commands:
  - verify the explicit key exists locally;
  - retry scoped `rsync -azR -e 'ssh -i ~/.ssh/id_ed25519_dgx_spark_windows -o BatchMode=yes -o ConnectTimeout=10'` for this note and the updated kernel-tuning skill.

## Remote Auth Retry Result

- Command: `test -f ~/.ssh/id_ed25519_dgx_spark_windows && ls -l ~/.ssh/id_ed25519_dgx_spark_windows`
- Result: key exists locally with mode `600`.
- Command: `ssh -i ~/.ssh/id_ed25519_dgx_spark_windows -o BatchMode=yes -o ConnectTimeout=10 caizus@10.100.1.253 'hostname; test -d /home/caizus/Projects/Packages/rwkv-rs-stable && echo stable_present=yes'`
- Result: succeeded; host `spark-35ac`, remote stable mirror present.
- Next command: scoped rsync of this note and `.agents/skills/kernel-tuning/SKILL.md` with the explicit key.

## Remote Sync Retry Result

- Command: `rsync -azR -e 'ssh -i ~/.ssh/id_ed25519_dgx_spark_windows -o BatchMode=yes -o ConnectTimeout=10' .agents/notes/kernel-tuning/2026-05-16-autotune-key-coverage-audit.md .agents/skills/kernel-tuning/SKILL.md caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable/`
- Result: passed; remote shell printed the expected zshenv trace and rsync returned exit `0`.
- Next command: verify the remote note exists and the remote skill contains the new fixed-dispatch audit rule.

## Remote Verification Retry

- Command: `ssh -i ~/.ssh/id_ed25519_dgx_spark_windows -o BatchMode=yes -o ConnectTimeout=10 caizus@10.100.1.253 'set -e; cd /home/caizus/Projects/Packages/rwkv-rs-stable; test -s .agents/notes/kernel-tuning/2026-05-16-autotune-key-coverage-audit.md; rg -n "Current train-kernel audit result|fixed-dispatch paths" .agents/skills/kernel-tuning/SKILL.md'`
- Result: invalid verification because remote shell reported `rg: command not found`.
- Next command: rerun the same file-existence and skill-content check with `grep`.

## Remote Verification Result

- Command: `ssh -i ~/.ssh/id_ed25519_dgx_spark_windows -o BatchMode=yes -o ConnectTimeout=10 caizus@10.100.1.253 'set -e; cd /home/caizus/Projects/Packages/rwkv-rs-stable; test -s .agents/notes/kernel-tuning/2026-05-16-autotune-key-coverage-audit.md; grep -n "Current train-kernel audit result" .agents/skills/kernel-tuning/SKILL.md; grep -n "fixed-dispatch paths" .agents/skills/kernel-tuning/SKILL.md'`
- Result: partial verification. The note exists and the skill contains the new audit rule at line `74`; the command exited `1` only because the second grep used lowercase `fixed-dispatch` while the synced line begins with uppercase `Fixed-dispatch`.
- Decision: remote sync is verified by note existence plus the matching skill audit-rule line. Resync this note once more so the remote mirror also contains the auth retry and verification history.

## Final Validation

- Command: `git diff --no-index --check -- /dev/null .agents/notes/kernel-tuning/2026-05-16-autotune-key-coverage-audit.md`
- Result: passed; no whitespace errors reported.
- Command: `git diff --no-index --check -- /dev/null .agents/skills/kernel-tuning/SKILL.md`
- Result: passed; no whitespace errors reported.
- Command: `rsync -azR -e 'ssh -i ~/.ssh/id_ed25519_dgx_spark_windows -o BatchMode=yes -o ConnectTimeout=10' .agents/notes/kernel-tuning/2026-05-16-autotune-key-coverage-audit.md caizus@10.100.1.253:/home/caizus/Projects/Packages/rwkv-rs-stable/`
- Result: passed; remote shell printed the expected zshenv trace and rsync returned exit `0`.
- Final state: this branch changed only the audit note and `.agents/skills/kernel-tuning/SKILL.md`; no kernel source, cache, or timing baseline was changed.
