#!/usr/bin/env bash
set -euo pipefail
source_dir=${1:?frozen source}
evidence_dir=${2:?new evidence directory}
board=${3:?ada or sm120}
case "$board" in
  ada)
    gpu=GPU-d1edd7be-e88d-aed6-047d-622163306f0e
    target=/root/target-ada-finalist-aldmatrix-cuda132-20260910
    cache=/root/ada-final-auto-cuda132-20260910/kernel-cache
    jit=/root/ada-final-auto-cuda132-20260910/jit-cache
    ;;
  sm120)
    gpu=GPU-a10ad830-a2cf-054d-e0fa-54add30a2cf8
    target=/root/target-sm120-final-sweep-cuda132
    cache=/root/sm120-release-matrix-cuda132-20260910/kernel-cache
    jit=/root/sm120-qualified-jit-cuda132-20260910
    ;;
  *) exit 2 ;;
esac
test ! -e "$evidence_dir"
mkdir -p "$evidence_dir" "$cache" "$jit"
chmod 700 "$cache"
cd "$source_dir"
export CUDA_HOME=/usr/local/cuda-13.2 CUDA_PATH=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64" CARGO_TARGET_DIR="$target"
export CUDA_VISIBLE_DEVICES="$gpu" MAMBA_RS_KERNEL_CACHE="$cache"
export CUDA_CACHE_DISABLE=0 CUDA_CACHE_PATH="$jit" CUDA_CACHE_MAXSIZE=1073741824
unset NVIDIA_TF32_OVERRIDE MAMBA_RS_BATCH_INVARIANT MAMBA_RS_BI_TENSOR_CORES MAMBA_RS_FAST_GEMM MAMBA_RS_BI_GEMM_FAMILY MAMBA_RS_BI_F32_POLICY MAMBA_RS_BI_HALF_POLICY MAMBA_RS_ARCH_RUNG
nvidia-smi --query-gpu=uuid,name,driver_version,memory.total --format=csv > "$evidence_dir/device.csv"
find src kernels tests -type f -print0 | sort -z | xargs -0 sha256sum > "$evidence_dir/assembly-source.sha256"
cp "$0" "$evidence_dir/runner.sh"
sha256sum "$0" > "$evidence_dir/runner.sha256"
target_name=inference_graph_route
test_name=decode_graphs_reject_complete_route_drift
cargo test --locked --release --features cuda --test "$target_name" --no-run > "$evidence_dir/build.log" 2>&1
cargo test --locked --release --features cuda --test "$target_name" -- --list > "$evidence_dir/tests.list" 2>&1
grep -Fx "$test_name: test" "$evidence_dir/tests.list"
binary=$(sed -n 's/^  Executable .* (\(.*\))$/\1/p' "$evidence_dir/build.log")
test -x "$binary"
sha256sum "$binary" > "$evidence_dir/test-binary.sha256"
sleep 2
for sample in 1 2 3 4 5; do
  telemetry=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.free --format=csv,noheader,nounits)
  printf '%s\n' "$telemetry" >> "$evidence_dir/preflight.log"
  awk -F, 'NF != 3 || $1+0 > 1 || $2+0 > 1 || $3+0 < 2048 {exit 1}' <<< "$telemetry"
  if [[ "$sample" != 5 ]]; then sleep 1; fi
done
date -u +%FT%TZ > "$evidence_dir/start.txt"
set +e
cargo test --locked --release --features cuda --test "$target_name" "$test_name" -- --exact --nocapture --test-threads=1 > "$evidence_dir/model-graph.log" 2>&1
result=$?
set -e
printf '%s\n' "$result" > "$evidence_dir/exit.txt"
date -u +%FT%TZ > "$evidence_dir/end.txt"
test "$result" = 0
grep -F 'test result: ok. 1 passed;' "$evidence_dir/model-graph.log"
date -u +%FT%TZ > "$evidence_dir/completed.txt"
