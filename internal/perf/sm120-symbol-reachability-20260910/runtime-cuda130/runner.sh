#!/usr/bin/env bash
set -euo pipefail
source_dir=/root/mamba-sm120-repaired.dzkoBl
evidence_dir=/root/sm120-reachability-runtime-cuda130-20260910
cd "$source_dir"
mkdir -p "$evidence_dir"
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:/root/.cargo/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export CARGO_TARGET_DIR=/root/target-sm120-final-sweep-cuda130
export MAMBA_RS_KERNEL_CACHE=/root/sm120-release-matrix-cuda130-20260910/kernel-cache
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_PATH=/root/sm120-qualified-jit-cuda130-20260910
export CUDA_CACHE_MAXSIZE=1073741824
mkdir -p "$CUDA_CACHE_PATH"
export CUDA_VISIBLE_DEVICES=GPU-a10ad830-a2cf-054d-e0fa-54add30a2cf8
unset NVIDIA_TF32_OVERRIDE
nvidia-smi --query-gpu=uuid,name,driver_version,memory.total --format=csv > "$evidence_dir/device.csv"
find src kernels tests -type f -print0 | sort -z | xargs -0 sha256sum > "$evidence_dir/assembly-source.sha256"
test_target=gemm_bi_tf32_contract
test_case=sm120_exact_nt_d768_out_reachable_fallback_matches_fixed_split_cpu_bits
cargo test --locked --release --features cuda --test "$test_target" --no-run > "$evidence_dir/build.log" 2>&1
cargo test --locked --release --features cuda --test "$test_target" -- --list > "$evidence_dir/tests.list" 2>&1
grep -Fx "$test_case: test" "$evidence_dir/tests.list"
sleep 2
for sample in 1 2 3 4 5; do
    telemetry=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.free --format=csv,noheader,nounits)
    printf '%s\n' "$telemetry" >> "$evidence_dir/preflight.log"
    awk -F, 'NF != 3 || $1+0 > 1 || $2+0 > 1 || $3+0 < 2048 {exit 1}' <<< "$telemetry"
    if [[ "$sample" != 5 ]]; then sleep 1; fi
done
cargo test --locked --release --features cuda --test "$test_target" "$test_case" -- --ignored --exact --nocapture --test-threads=1 > "$evidence_dir/runtime.log" 2>&1
grep -F 'test result: ok. 1 passed;' "$evidence_dir/runtime.log"
export GEMM_BI_QUAL_WINDOWS=3
export GEMM_BI_QUAL_VARIANT=sm120-reachability-cuda130-s2
export GEMM_BI_QUAL_CELL_IDS=f32_policy_exact/nt/d768_out_proj/contiguous,f32_policy_allow_tf32/nt/d768_out_proj/contiguous
export MAMBA_FINAL_CACHE_ROOT="$MAMBA_RS_KERNEL_CACHE"
bash "$source_dir/run-focused-cuda.sh" "$source_dir" "$CUDA_HOME" "$CARGO_TARGET_DIR" "$evidence_dir/matrix-two" gemm_bi_performance_matrix gemm_bi_deterministic_performance_matrix
