#!/usr/bin/env bash
set -euo pipefail
source_dir=${1:?unchanged frozen source}
built_evidence=${2:?original completed build evidence}
evidence_dir=${3:?new evidence directory}
test ! -e "$evidence_dir"
mkdir -p "$evidence_dir"
trap 'result=$?; printf "%s\n" "$result" > "$evidence_dir/runner-exit.txt"' EXIT
cd "$source_dir"
export CUDA_HOME=/usr/local/cuda-13.2 CUDA_PATH=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export CUDA_VISIBLE_DEVICES=GPU-d1edd7be-e88d-aed6-047d-622163306f0e
export MAMBA_RS_KERNEL_CACHE=/root/ada-final-auto-cuda132-20260910/kernel-cache
export CUDA_CACHE_DISABLE=0 CUDA_CACHE_MAXSIZE=1073741824
export CUDA_CACHE_PATH=/root/ada-final-auto-cuda132-20260910/jit-cache
unset RUSTFLAGS RUSTDOCFLAGS NVIDIA_TF32_OVERRIDE MAMBA_RS_GEMM_MODE MAMBA_RS_BATCH_INVARIANT MAMBA_RS_BI_TENSOR_CORES MAMBA_RS_FAST_GEMM MAMBA_RS_BI_GEMM_FAMILY MAMBA_RS_BI_F32_POLICY MAMBA_RS_BI_HALF_POLICY MAMBA_RS_ARCH_RUNG
test "$(stat -c '%a:%u' "$MAMBA_RS_KERNEL_CACHE")" = '700:0'
date -u +%FT%TZ > "$evidence_dir/start.txt"
cp "$0" "$evidence_dir/runner.sh"
cp "$built_evidence/required-tests.txt" "$evidence_dir/required-tests.txt"
sha256sum -c "$built_evidence/source.sha256" > "$evidence_dir/source-before.log"
sha256sum -c "$built_evidence/build-inputs.sha256" > "$evidence_dir/build-inputs-before.log"
sha256sum -c "$built_evidence/binaries.sha256" > "$evidence_dir/binaries-before.log"
lib_binary=$(sed -n 's/^  Executable unittests src\/lib.rs (\(.*\))$/\1/p' "$built_evidence/build.log")
route_binary=$(sed -n 's/^  Executable tests\/inference_graph_route.rs (\(.*\))$/\1/p' "$built_evidence/build.log")
test -x "$lib_binary"
test -x "$route_binary"
idle_preflight() {
  local label=$1 attempt
  for attempt in 1 2 3; do
    nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits > "$evidence_dir/$label.preflight-$attempt.csv"
    if awk -F, 'NF != 2 || $1+0 > 1 || $2+0 < 2048 {exit 1}' "$evidence_dir/$label.preflight-$attempt.csv"; then return 0; fi
    sleep 2
  done
  return 1
}
completed=0
while read -r target lane test_name extra; do
  test "$lane" = gpu || continue
  test -z "$extra"
  case "$target" in
    lib) test_binary=$lib_binary ;;
    route) test_binary=$route_binary ;;
    *) exit 2 ;;
  esac
  "$test_binary" --list | grep -Fx "$test_name: test"
  idle_preflight "$test_name"
  set +e
  "$test_binary" "$test_name" --exact --include-ignored --nocapture --test-threads=1 > "$evidence_dir/$test_name.log" 2>&1
  test_result=$?
  set -e
  printf '%s\n' "$test_result" > "$evidence_dir/$test_name.exit"
  test "$test_result" -eq 0
  grep -F 'test result: ok. 1 passed; 0 failed; 0 ignored;' "$evidence_dir/$test_name.log"
  completed=$((completed + 1))
done < "$evidence_dir/required-tests.txt"
test "$completed" -eq 13
sha256sum -c "$built_evidence/source.sha256" > "$evidence_dir/source-after.log"
sha256sum -c "$built_evidence/binaries.sha256" > "$evidence_dir/binaries-after.log"
date -u +%FT%TZ > "$evidence_dir/completed.txt"
