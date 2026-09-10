#!/usr/bin/env bash
set -euo pipefail
source_dir=${1:?frozen source}
evidence_dir=/root/ada-final-models-732c1146-20260910
cd "$source_dir"
mkdir -p "$evidence_dir"
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:/root/.cargo/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export CUDA_VISIBLE_DEVICES=0
export CARGO_TARGET_DIR=/root/target-ada-finalist-aldmatrix-cuda132-20260910
export MAMBA_RS_KERNEL_CACHE=/root/ada-final-models-732c1146-20260910/kernel-cache
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_PATH=/root/ada-final-models-732c1146-20260910/jit-cache
export CUDA_CACHE_MAXSIZE=1073741824
unset MAMBA_RS_BATCH_INVARIANT MAMBA_RS_BI_TENSOR_CORES MAMBA_RS_FAST_GEMM
unset MAMBA_RS_BI_GEMM_FAMILY MAMBA_RS_BI_F32_POLICY MAMBA_RS_BI_HALF_POLICY
unset MAMBA_RS_ARCH_RUNG NVIDIA_TF32_OVERRIDE
nvidia-smi --query-gpu=uuid,name,driver_version,memory.total --format=csv > "$evidence_dir/device.csv"
find src kernels tests -type f -print0 | sort -z | xargs -0 sha256sum > "$evidence_dir/assembly-source.sha256"
cp "$0" "$evidence_dir/runner.sh"
preflight() {
    sleep 2
    for sample in 1 2 3 4 5; do
        telemetry=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.free --format=csv,noheader,nounits)
        printf '%s\n' "$telemetry"
        awk -F, 'NF != 3 || $1+0 > 1 || $2+0 > 1 || $3+0 < 2048 {exit 1}' <<< "$telemetry"
        if [[ "$sample" != 5 ]]; then sleep 1; fi
    done
}
for entry in \
    f32_training_graph_parity:m1_f32_training_graph_matches_eager \
    f32_training_graph_parity:m3_f32_training_graph_matches_eager \
    training_graph_parity:training_graph_bf16_multi_replay_matches_eager \
    m3_training_graph_parity:m3_training_graph_bf16_multi_replay_matches_eager \
    m3_training_graph_safety:m3_bf16_training_graph_replays_deterministically \
    m3_training_graph_safety:m3_f16_training_graph_replays_deterministically \
    m3_training_graph_safety:m3_f32_training_graph_replays_deterministically; do
    test_target=${entry%%:*}
    test_case=${entry#*:}
    cargo test --locked --release --features cuda --test "$test_target" --no-run > "$evidence_dir/$test_case.build.log" 2>&1
    cargo test --locked --release --features cuda --test "$test_target" -- --list > "$evidence_dir/$test_case.tests.list" 2>&1
    grep -Fx "$test_case: test" "$evidence_dir/$test_case.tests.list"
    preflight > "$evidence_dir/$test_case.preflight.log"
    test_flags=(--exact --nocapture --test-threads=1)
    if [[ "$test_target" = m3_training_graph_safety ]]; then test_flags+=(--ignored); fi
    cargo test --locked --release --features cuda --test "$test_target" "$test_case" -- "${test_flags[@]}" > "$evidence_dir/$test_case.log" 2>&1
    grep -F 'test result: ok. 1 passed;' "$evidence_dir/$test_case.log"
done
date -u +%FT%TZ > "$evidence_dir/completed.txt"
