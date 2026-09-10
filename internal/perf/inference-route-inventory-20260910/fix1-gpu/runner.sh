#!/usr/bin/env bash
set -euo pipefail
source_dir=${1:?frozen source directory}
evidence_dir=${2:?new evidence directory}
test ! -e "$evidence_dir"
mkdir -p "$evidence_dir"
trap 'result=$?; printf "%s\n" "$result" > "$evidence_dir/runner-exit.txt"' EXIT
cd "$source_dir"
export CUDA_HOME=/usr/local/cuda-13.2 CUDA_PATH=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export CARGO_TARGET_DIR=/root/target-ada-finalist-aldmatrix-cuda132-20260910
export CUDA_VISIBLE_DEVICES=GPU-d1edd7be-e88d-aed6-047d-622163306f0e
export MAMBA_RS_KERNEL_CACHE=/root/ada-final-auto-cuda132-20260910/kernel-cache
export CUDA_CACHE_DISABLE=0 CUDA_CACHE_MAXSIZE=1073741824
export CUDA_CACHE_PATH=/root/ada-final-auto-cuda132-20260910/jit-cache
unset RUSTFLAGS RUSTDOCFLAGS NVIDIA_TF32_OVERRIDE MAMBA_RS_GEMM_MODE MAMBA_RS_BATCH_INVARIANT MAMBA_RS_BI_TENSOR_CORES MAMBA_RS_FAST_GEMM MAMBA_RS_BI_GEMM_FAMILY MAMBA_RS_BI_F32_POLICY MAMBA_RS_BI_HALF_POLICY MAMBA_RS_ARCH_RUNG
test "$(stat -c '%a:%u' "$MAMBA_RS_KERNEL_CACHE")" = '700:0'
date -u +%FT%TZ > "$evidence_dir/start.txt"
cp "$0" "$evidence_dir/runner.sh"
find src kernels tests -type f -print0 | sort -z | xargs -0 sha256sum > "$evidence_dir/source.sha256"
sha256sum Cargo.toml Cargo.lock > "$evidence_dir/build-inputs.sha256"
nvcc --version > "$evidence_dir/toolkit.txt"
nvidia-smi --query-gpu=uuid,name,driver_version,utilization.gpu,memory.used,memory.free --format=csv > "$evidence_dir/device.csv"
cargo test --locked --release --features cuda,hf --test gemm_inference_route_inventory --no-run > "$evidence_dir/routing-build.log" 2>&1
routing_binary=$(sed -n 's/^  Executable .* (\(.*\))$/\1/p' "$evidence_dir/routing-build.log")
test -x "$routing_binary"
sha256sum "$routing_binary" > "$evidence_dir/routing-binary.sha256"
"$routing_binary" --list > "$evidence_dir/routing-tests.list"
test "$(grep -c ': test$' "$evidence_dir/routing-tests.list")" -eq 5
idle_preflight() {
  local label=$1
  local attempt
  for attempt in 1 2 3; do
    nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits > "$evidence_dir/$label.preflight-$attempt.csv"
    if awk -F, 'NF != 2 || $1+0 > 1 || $2+0 < 2048 {exit 1}' "$evidence_dir/$label.preflight-$attempt.csv"; then
      return 0
    fi
    sleep 2
  done
  return 1
}
idle_preflight routing
test_name=deterministic_inference_direct_empty_output_has_no_terminal_record
grep -Fx "$test_name: test" "$evidence_dir/routing-tests.list"
"$routing_binary" "$test_name" --exact --include-ignored --nocapture --test-threads=1 > "$evidence_dir/routing-tests.log" 2>&1
grep -F 'test result: ok. 1 passed; 0 failed; 0 ignored;' "$evidence_dir/routing-tests.log"
sha256sum -c "$evidence_dir/source.sha256" > "$evidence_dir/source-after.log"
date -u +%FT%TZ > "$evidence_dir/completed.txt"
