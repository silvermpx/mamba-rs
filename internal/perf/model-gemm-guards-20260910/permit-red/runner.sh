#!/usr/bin/env bash
set -euo pipefail
source_dir=${1:?frozen source directory}
evidence_dir=${2:?new evidence directory}
shift 2
test "$#" -ge 1
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
printf '%s\n' "$@" > "$evidence_dir/required-tests.txt"
find src kernels tests -type f -print0 | sort -z | xargs -0 sha256sum > "$evidence_dir/source.sha256"
sha256sum Cargo.toml Cargo.lock > "$evidence_dir/build-inputs.sha256"
nvcc --version > "$evidence_dir/toolkit.txt"
nvidia-smi --query-gpu=uuid,name,driver_version,utilization.gpu,memory.used,memory.free --format=csv > "$evidence_dir/device.csv"
cargo test --locked --release --features cuda,hf --lib --no-run > "$evidence_dir/build.log" 2>&1
test_binary=$(sed -n 's/^  Executable .* (\(.*\))$/\1/p' "$evidence_dir/build.log")
test -x "$test_binary"
sha256sum "$test_binary" > "$evidence_dir/binary.sha256"
"$test_binary" --list > "$evidence_dir/tests.list"
for test_name in "$@"; do
  grep -Fx "$test_name: test" "$evidence_dir/tests.list"
done
for test_name in "$@"; do
  idle=0
  for attempt in 1 2 3; do
    nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits > "$evidence_dir/$test_name.preflight.$attempt.csv"
    if awk -F, 'NF != 2 || $1+0 > 1 || $2+0 < 2048 {exit 1}' "$evidence_dir/$test_name.preflight.$attempt.csv"; then idle=1; break; fi
    sleep 2
  done
  test "$idle" -eq 1
  set +e
  "$test_binary" "$test_name" --exact --ignored --nocapture --test-threads=1 > "$evidence_dir/$test_name.log" 2>&1
  test_result=$?
  set -e
  printf '%s\n' "$test_result" > "$evidence_dir/$test_name.exit"
  test "$test_result" -eq 101
  grep -F 'test result: FAILED. 0 passed; 1 failed; 0 ignored;' "$evidence_dir/$test_name.log"
done
sha256sum -c "$evidence_dir/source.sha256" > "$evidence_dir/source-after.log"
sha256sum -c "$evidence_dir/build-inputs.sha256" > "$evidence_dir/build-inputs-after.log"
date -u +%FT%TZ > "$evidence_dir/completed.txt"
