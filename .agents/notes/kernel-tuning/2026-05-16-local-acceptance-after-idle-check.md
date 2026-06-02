# Local Acceptance After Idle Check

- Date: 2026-05-16
- Branch/worktree: `kernel-tuning-local-acceptance-after-idle-check-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout inherits broad unrelated workspace changes and the current remote-clean tuning tree. This branch owns this note, local acceptance measurement, and any strictly necessary local autotune-cache hygiene.
- Prior-note/source search command:
  - `rg -n "local acceptance|local compare|LayerNorm.*cache|rejected.*cache|autotune cache|ordered256|本机|idle|nvidia-smi|speedup=0\\.|speedup > 1" .agents/notes/kernel-tuning /root/.codex/memories/MEMORY.md -S`
- Matched prior evidence:
  - `2026-05-16-local-gpu-availability-check.md` reports the local RTX 5090 is visible with no active compute apps, around `5%` utilization and `3022/32607 MiB` memory.
  - Prior local compares remain below `1.0`, while the current remote clean gate is `2.10x`.
  - `2026-05-16-layernorm-ordered256-post-gates.md` warns that a rejected ordered-256 LayerNorm autotune cache may pollute later local compares and records one moved-aside rejected cache file.
- Changed boundary: this is a final local acceptance attempt after confirming the GPU appears idle. It does not change kernel code.
- Machine/GPU: local `NVIDIA GeForce RTX 5090`, compute capability `12.0`.
- Shape/dtype: CUDA BF16 `rwkv_lm`, `B=16,T=512,D=768`, rows `8192`, local baseline under `crates/rwkv-test/test_data/rwkv_lm/bf16/case_000000`.
- Candidate parameters: none.
- Expected keep/revert boundary: if activation passes and timing speedup is above `1.0`, record local success and run a completion audit. If activation or timing fails, do not mark the goal complete; analyze the failing rows and decide the next implementation/profiling branch.
- Preflight result:
  - Current `nvidia-smi`: `NVIDIA GeForce RTX 5090`, compute capability `12.0`, utilization `6%`, memory `3022/32607 MiB`.
  - No active compute apps reported by `nvidia-smi --query-compute-apps`.
  - Existing LayerNorm cache files: active `rwkv_nn-kernels-train-layer_norm-forward-layer-norm-forward.json.log` plus moved-aside `...rejected-ordered256-20260516`.
- Next command: inspect active LayerNorm autotune cache to confirm the rejected ordered-256 candidate is not live before running local compare.

## 2026-05-16 Cache Check

- Active LayerNorm autotune cache key:
  - `runtime=cuda`, `dtype=BF16`, `d_model=768`, `rows=8192`, `num_streaming_multiprocessors=170`, `deterministic=true`, `deterministic_min_block_size=1024`.
  - Tunable results skip `block_64`, `block_128`, `block_256`, `block_512`, and `block_768`; selected `block_1024`.
- Interpretation: the rejected ordered-256 candidate is not live in the active LayerNorm cache, and the local deterministic boundary remains `1024`.
- Next command: run standard local `compare-rwkv-nn --repeat 3 --warmup 1` for activation and timing acceptance.

## 2026-05-16 Local Compare Result

- Command:
  - `cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never --repeat 3 --warmup 1`
- Build provenance:
  - The command rebuilt and ran `target/release/rwkv-test` from this branch.
- Correctness:
  - `activation_summary compared=54 passed=54 failed=0 missing=0 extra=0`
- Timing:
  - `timing_summary compared=76 passed=13 failed=63 missing=0 extra=0 ignored=1 actual_total_ms=38.884 baseline_total_ms=35.957 speedup=0.92x`
- Major module rows:
  - `cells/*/time_mixer`: `21.602ms` actual vs `27.836ms` baseline, `1.29x`.
  - `cells/*/channel_mixer`: `9.268ms` actual vs `5.527ms` baseline, `0.60x`.
  - `pre_layer_norm_for_time_mix`: `2.115ms` actual vs `0.495ms` baseline, `0.23x`.
  - `pre_layer_norm_for_channel_mix`: `1.932ms` actual vs `0.501ms` baseline, `0.26x`.
  - `loss/l2wrap_cross_entropy`: `1.108ms` actual vs `0.717ms` baseline, `0.65x`.
  - `lm_head`: `0.131ms` actual vs `0.034ms` baseline, `0.26x`.
- Decision:
  - The local acceptance gate is still open. This result confirms correctness on local RTX 5090, but local timing is still below `1.0`.
  - The failure is not the previous ordered-256 accuracy drift; the active LayerNorm cache selects `block_1024` and activation passes.
- Next command:
  - Inspect the generated actual timing JSON and local baseline timing JSON for the dominant failing rows to confirm whether this is a stable kernel regression target or a local baseline/provenance mismatch.

## 2026-05-16 Post-Run Environment Check

- Post-run GPU check:
  - `NVIDIA GeForce RTX 5090`, compute capability `12.0`, utilization `5%`, memory `3023/32607 MiB`.
  - `nvidia-smi --query-compute-apps` reported no active compute apps.
- Invalid inspection command:
  - Tried to use `jq` to print selected timing JSON files, but `jq` is not installed in this environment.
  - No conclusion is drawn from that failed command.
- Next command:
  - Re-run the selected timing JSON inspection with plain file reads.

## 2026-05-16 Selected Timing JSON Inspection

- Plain file inspection confirmed:
  - `cell_0000/channel_mixer`: actual `847127ns`, samples `[691307,1027068,823006]`; baseline `475501ns`, samples `[505213,459498,461791]`.
  - `cell_0000/pre_layer_norm_for_time_mix`: actual `135639ns`, samples `[127537,148249,131130]`; baseline `121541ns`, samples `[39696,283321,41606]`.
  - Most local baseline pre-layernorm rows are around `33-44us`; current actual pre-layernorm rows are commonly `120-170us` with some outliers.
  - Most local baseline channel_mixer rows are around `453-475us`; current actual channel_mixer rows are around `706-847us`.
  - `lm_head`: actual `131390ns`; baseline `34110ns`.
  - `loss/l2wrap_cross_entropy`: actual `1107912ns`; baseline `716980ns`.
- Interpretation:
  - This is a local timing failure despite clean activation.
  - The failure is dominated by local short-kernel/module boundaries and channel_mixer, while `cells/*/time_mixer` remains faster than baseline.
  - This acceptance branch should not keep running broad compares. The next step needs a dedicated diagnosis branch for local short-kernel/dispatch behavior and baseline provenance.
- Keep/revert state:
  - No kernel code was changed on this branch.
  - Keep this branch and note as negative local acceptance evidence.
