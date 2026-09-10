#!/usr/bin/env bash
set -euo pipefail
source_dir=${1:?frozen source}
evidence_dir=${2:?new evidence directory}
test ! -e "$evidence_dir"
mkdir -p "$evidence_dir"
cd "$source_dir"
export CUDA_VISIBLE_DEVICES=GPU-a10ad830-a2cf-054d-e0fa-54add30a2cf8
export CUDA_CACHE_DISABLE=0 CUDA_CACHE_MAXSIZE=1073741824
export CUDA_CACHE_PATH=/root/sm120-qualified-jit-cuda130-20260910
export MAMBA_FINAL_CACHE_ROOT=/root/sm120-release-matrix-cuda130-20260910/kernel-cache
mkdir -p "$CUDA_CACHE_PATH" "$MAMBA_FINAL_CACHE_ROOT"
chmod 700 "$MAMBA_FINAL_CACHE_ROOT"
unset NVIDIA_TF32_OVERRIDE
export GEMM_BI_QUAL_WINDOWS=3
export GEMM_BI_QUAL_VARIANT=sm120-cuda130-missing14-post-cohort
export GEMM_BI_QUAL_CELL_IDS=bf16_policy_tc/nn/prism_in_proj/contiguous,bf16_policy_tc/nt/d768_out_proj/contiguous,bf16_policy_tc/nt/prism_in_proj/contiguous,bf16_policy_tc/tn/prism_in_proj/contiguous,f16_policy_tc/nn/prism_in_proj/contiguous,f16_policy_tc/nt/d768_out_proj/contiguous,f16_policy_tc/nt/prism_in_proj/contiguous,f16_policy_tc/tn/prism_in_proj/contiguous,f32_policy_allow_tf32/nn/prism_in_proj/contiguous,f32_policy_allow_tf32/nt/prism_in_proj/contiguous,f32_policy_allow_tf32/tn/prism_in_proj/contiguous,f32_policy_exact/nn/prism_in_proj/contiguous,f32_policy_exact/nt/prism_in_proj/contiguous,f32_policy_exact/tn/prism_in_proj/contiguous
find src kernels tests -type f -print0 | sort -z | xargs -0 sha256sum > "$evidence_dir/assembly-source.sha256"
cp "$0" "$evidence_dir/runner.sh"
sha256sum "$0" > "$evidence_dir/runner.sha256"
date -u +%FT%TZ > "$evidence_dir/start.txt"
bash "$source_dir/run-focused-cuda.sh" "$source_dir" /usr/local/cuda-13.0 /root/target-sm120-final-sweep-cuda130 "$evidence_dir" gemm_bi_performance_matrix gemm_bi_deterministic_performance_matrix
date -u +%FT%TZ > "$evidence_dir/completed.txt"
