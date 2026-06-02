# Autotune Key Audit

- Date: 2026-05-16
- Branch/worktree: `kernel-tuning-autotune-key-audit-20260516` in the existing dirty workspace.
- Dirty-tree constraint: this checkout already carries uncommitted kernel, skill, fixture, and crate-structure changes from earlier work. This attempt audits current train-kernel autotune keys before making another key-design change.
- Prior-note search command: `rg -n "struct .*AutotuneKey|impl AutotuneKey|AutotuneKey|runtime_name|compute|num_tensor_cores|max_cube_dim|max_units_per_cube|d_model|rows|block_size|num_warps|line_size|max_line_size|is_in_place|alias|deterministic|dtype|StorageType" crates/rwkv-nn/src/kernels/train .agents/notes/kernel-tuning .agents/skills/kernel-tuning/SKILL.md /root/.codex/memories/MEMORY.md`.
- Matched prior evidence:
  - `2026-05-16-layernorm-hardware-policy.md`, `2026-05-16-lm-head-forward-hardware-key.md`, `2026-05-16-forward-elementwise-full-hardware-key.md`, `2026-05-16-backward-autotune-hardware-key.md`, and `2026-05-16-wkv7-full-hardware-key.md` record earlier key expansions.
  - `kernel-tuning` skill now requires implementation choice to be keyed by backend/runtime, hardware, dtype, shape, candidate parameters, alias/in-place state, and deterministic numeric boundary.
  - `MEMORY.md` records the LayerNorm remote/local conflict: remote `BLOCK_SIZE=256` passed and sped up, local smaller blocks drifted.
- Scope: current autotune-key structs under `crates/rwkv-nn/src/kernels/train`.
- Hypothesis: before adding another performance candidate, verify which current keys still miss required hardware/shape dimensions. Missing keys should become separate implementation branches; this audit branch records the exact gaps.
- Candidate parameters: no live code candidate yet. Audit fields include runtime/backend, compute capability or CubeCL hardware fingerprint, dtype, `d_model`, rows, block size/warp or line-size candidate representation, vector width, in-place/alias state, and deterministic boundary.
- Expected keep/revert boundary: keep this note as audit evidence. If a key gap is found, open a new branch for that specific key family instead of patching multiple unrelated keys here.

## Coverage Matrix

All current `AutotuneKey` structs under `crates/rwkv-nn/src/kernels/train` include backend/runtime through `R::name(...)`, dtype, shape, CubeCL hardware fingerprint fields, alias/in-place state, and a deterministic flag. CubeCL does not expose a portable CUDA compute-capability major/minor field at this call site; current keys use the available hardware fingerprint fields: `load_width`, `plane_size_max`, `max_units_per_cube`, `max_cube_dim`, `max_shared_memory_size`, `max_vector_size`, `num_streaming_multiprocessors`, `num_tensor_cores`, and `min_tensor_cores_dim`.

| Kernel family | Candidate parameter | Shape fields | Hardware/runtime fields | Alias/deterministic | Status |
| --- | --- | --- | --- | --- | --- |
| `layer_norm/forward.rs` | `block_size` via tunable name/group | `d_model`, `rows`, `num_elements` | covered | `is_in_place=false`, `deterministic=true`, `deterministic_min_block_size` | covered; BF16 deterministic guard currently filters small block sizes before tuning |
| `lm_head_l2wrap_ce/forward.rs` | `block_size` via tunable name/group | `num_tokens`, `vocab_size` | covered | covered | covered |
| `lm_head_l2wrap_ce/backward.rs` | `block_size` via tunable name/group | `num_tokens`, `vocab_size` | covered | covered | covered |
| `channel_mixer/forward.rs` mix and relu-square | `line_size` via tunable name/group | `num_elements`, `rows`, `innermost_dim` | covered | relu-square captures mutable alias; mix is non-in-place | covered |
| `channel_mixer/backward.rs` mix backward | `line_size`, reduce `block_size`, and `bt_tile` via tunable name/group | `num_elements`, `d_model`, `rows` | covered | covered | covered |
| `time_mixer/learning_rate_gate/forward.rs` | `line_size` via tunable name/group | `num_elements`, `embedded_dim`, `rows` | covered | covered | covered |
| `time_mixer/learning_rate_gate/backward.rs` | `line_size`, reduce `block_size`, and `bt_tile` via tunable name/group | `num_elements`, `d_model`, `rows` | covered | covered | covered |
| `time_mixer/value_residual_gate/forward.rs` | `line_size` via tunable name/group | `num_elements`, `embedded_dim`, `rows` | covered | covered | covered |
| `time_mixer/value_residual_gate/backward.rs` | `line_size`, reduce `block_size`, and `bt_tile` via tunable name/group | `num_elements`, `d_model`, `rows` | covered | covered | covered |
| `time_mixer/mix6/backward.rs` | `line_size`, reduce `block_size`, and `bt_tile` via tunable name/group | `num_elements`, `d_model`, `rows` | covered | covered | covered |
| `time_mixer/wkv7/forward.rs` pretrain output | `row_tile` via tunable name/group | `batch_size`, `context_len`, `rows`, `d_model`, `num_heads`, `head_size`, `chunk_len` | covered | covered | covered |

## Gaps

- `time_mixer/mix6/forward.rs` still chooses vector width with `best_line_size(...)` and has no `AutotuneKey` / `LocalTuner`. This is a real gap because vector width is an implementation choice and should be keyed by runtime, hardware, dtype, shape, alias state, and deterministic boundary.
- `residual_add/forward.rs` also chooses vector width with `best_line_size(...)` and has no `AutotuneKey` / `LocalTuner`. This path has prior residual-add/Burn-add A/B notes, so any implementation branch must first cite those notes and state why the retry is materially different.
- `time_mixer/key_prepare/{forward,backward}.rs` and WKV7 state/backward launch fixed kernels keyed by checked shape contracts such as `head_size=64` or `shape[3]`. They have no current candidate set to tune. They are not autotune-key gaps until a second implementation, block configuration, vector width, or dispatch policy is introduced.

## Decision

- Keep this branch as audit evidence only.
- Next implementation branch should be scoped to `time_mixer/mix6/forward.rs`: add a forward `AutotuneKey` and line-size tunables, then validate activation and timing before touching residual-add or other kernels.

## Follow-up Implementation Branches

- `kernel-tuning-mix6-forward-hw-key-20260516` addressed the `mix6/forward.rs` vector-width key gap. Local CUDA check passed and trace activation passed; overall timing still failed at `speedup=0.83x`, while the directly relevant `cells/*/time_mixer` group stayed above baseline at `1.10x`.
- `kernel-tuning-residual-add-hw-key-20260516` addressed the custom `residual_add/forward.rs` vector-width key gap without repeating the Burn-add A/B. Local CUDA check passed and trace activation passed; overall timing still failed at `speedup=0.85x`, while `cells/*/time_mixer` stayed above baseline at `1.06x`.
