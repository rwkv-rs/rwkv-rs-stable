#!/usr/bin/env bash
set -euo pipefail

repo="${RWKV_RS_STABLE_ROOT:-/home/caizus/Projects/Packages/rwkv-rs-stable}"
baseline="${RWKV_RS_TEST_BASELINE:-/home/caizus/Projects/Packages/rwkv-rs-test/test_gen/rwkv_lm/bf16/case_000000}"
projection_baseline="${RWKV_RS_TEST_PROJECTION_R9_BASELINE:-/home/caizus/Projects/Packages/rwkv-rs-test/test_gen_projection_r9_20260516/rwkv_lm/bf16/case_000000}"
ncu="${NCU_BIN:-/usr/local/cuda-13.0/bin/ncu}"
out_dir="${NCU_OUT_DIR:-target/rwkv-test}"
export PATH="/home/caizus/.cargo/bin:/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:${PATH}"

cd "$repo"

echo "host=$(hostname)"
echo "repo=$PWD"
echo "ncu=$ncu"
"$ncu" --version
grep RmProfilingAdminOnly /proc/driver/nvidia/params || true
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader

test -d "$baseline"
test -d "$projection_baseline"
mkdir -p "$out_dir"

cargo build --release -p rwkv-test --features cuda

common_sections=(
  --section SpeedOfLight
  --section Occupancy
  --section SchedulerStats
  --section WarpStateStats
  --section MemoryWorkloadAnalysis
)

"$ncu" \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*(layer_norm_forward|lm_head_l2wrap_ce_forward_row|wkv7_pretrain_forward_output|mix6_forward|channel_mixer_relu_square|gated_readout_combine|key_prepare_forward_64|value_residual_gate_forward).*' \
  --launch-count 240 \
  "${common_sections[@]}" \
  --csv \
  --log-file "$out_dir/ncu-top-owned-gb10.csv" \
  target/release/rwkv-test compare-rwkv-nn \
    --color never \
    --baseline "$baseline" \
    --repeat 1 \
    --warmup 1 || {
      status=$?
      echo "top-owned ncu command exited with status $status; inspect CSV for profiler errors" >&2
      exit "$status"
    }

"$ncu" \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*matmul_entry_lhs_bf16_lhs_size_1_rhs_bf16_rhs_size_1_acc_bf16_acc_size_8.*' \
  --launch-count 12 \
  "${common_sections[@]}" \
  --csv \
  --log-file "$out_dir/ncu-lm-head-projection-matmul-gb10.csv" \
  target/release/rwkv-test compare-rwkv-nn \
    --color never \
    --baseline "$projection_baseline" \
    --repeat 1 \
    --warmup 1 || {
      status=$?
      echo "projection ncu command exited with status $status; inspect CSV for profiler errors" >&2
      exit "$status"
    }

echo "generated:"
ls -lh "$out_dir/ncu-top-owned-gb10.csv" "$out_dir/ncu-lm-head-projection-matmul-gb10.csv"
