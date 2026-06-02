# Forward Elementwise Autotune Key Expansion

- Date: 2026-05-16
- Branch/worktree: `kernel-tuning-hw-shape-dispatch`
- Machine/GPU: local CUDA machine
- Scope: forward elementwise autotune keys for `channel_mixer`, `learning_rate_gate`, and `value_residual_gate`.
- Shape/dtype target: CUDA BF16 `rwkv_lm` trace family, `B=16,T=512,D=768`.
- Change: added runtime, hardware capability fields, rows, embedded dimension, max vector width, in-place/alias state, and deterministic flag to the forward elementwise autotune keys.
- Candidate parameters: existing line-size candidate sets remain unchanged; line size is represented by tunable names/checksum and group validity, while shape/hardware capabilities now participate in the key.
- Validation: `rtk rustfmt +nightly ...` passed for the touched files; `rtk cargo check -p rwkv-nn --features cuda` passed.
- Trace compare result: local `rtk cargo run --release -p rwkv-test --features cuda -- compare-rwkv-nn --color never` passed activation comparison, `54/54 PASS`, but timing failed. Pre-backward-autotune run was `actual_total_ms=43.693`, baseline `35.957`, total `0.82x`; after the backward experiment, warm-cache forward compare was `actual_total_ms=46.339`, baseline `35.957`, total `0.78x`.
- Decision: key expansion remains correctness-safe, but timing is still below acceptance. Next profiling target is the forward steady-state slow group: `channel_mixer`, pre-layer-norm stages, `lm_head`, and `loss/l2wrap_cross_entropy`.
