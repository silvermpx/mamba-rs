#!/usr/bin/env bash
set -euo pipefail
cd /root/mamba-rs-triad
logdir=/root/triad-ada-current21-20260906
mkdir -p "$logdir"
exec > >(tee -a "$logdir/runner-v1.log") 2>&1
trap 'rc=$?; echo "RUNNER_EXIT=$rc $(date -u +%FT%TZ)"' EXIT
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export CARGO_TARGET_DIR=/root/mamba-rs-triad/target-132-triad-current21
export CUDA_VISIBLE_DEVICES=GPU-d1edd7be-e88d-aed6-047d-622163306f0e
unset NVIDIA_TF32_OVERRIDE
export MAMBA_RS_KERNEL_CACHE
MAMBA_RS_KERNEL_CACHE=$(mktemp -d /root/triad-ada-cache-XXXXXX)
chmod 700 "$MAMBA_RS_KERNEL_CACHE"
export MAMBA_RS_CACHE_TRACE=1 CUDA_CACHE_DISABLE=1
export GEMM_BI_QUAL_VARIANT=ada-current21 GEMM_BI_QUAL_PATH_ORDER=ab
echo "ENV $(date -u +%FT%TZ)"
env | sort | grep -E '^(CUDA_|CARGO_TARGET_DIR|LD_LIBRARY_PATH|PATH|MAMBA_RS_|GEMM_BI_)'
nvcc --version
cargo --version
rustc --version
nvidia-smi
sha256sum Cargo.toml Cargo.lock src/mamba_ssm/gpu/gemm_bi_triad/mod.rs src/mamba_ssm/gpu/gemm_bi_triad/modules.rs tests/gemm_bi_performance_matrix.rs
qual_ids=()
for route in f32_policy_exact f32_policy_allow_tf32 bf16_policy_tc f16_policy_tc; do
  for op in nn tn nt; do
    for shape in d128_in_proj d128_out_proj d768_in_proj d768_out_proj prism_in_proj; do
      qual_ids+=("$route/$op/$shape/contiguous")
    done
  done
done
vendor_ids=()
for dtype in f32 bf16 f16; do
  for op in nn tn nt; do
    for shape in d128_in_proj d128_out_proj d768_in_proj d768_out_proj prism_in_proj; do
      vendor_ids+=("cublas/$dtype/$op/$shape")
    done
  done
done
export GEMM_BI_QUAL_CELL_IDS GEMM_BI_CUBLAS_CELL_IDS
GEMM_BI_QUAL_CELL_IDS=$(IFS=,; echo "${qual_ids[*]}")
GEMM_BI_CUBLAS_CELL_IDS=$(IFS=,; echo "${vendor_ids[*]}")
idle_gate() {
  nvidia-smi --query-gpu=uuid,utilization.gpu,memory.used --format=csv,noheader
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
    echo "FAIL concurrent CUDA process"; return 1
  fi
  local utilization
  utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  if [ "$utilization" != 0 ]; then echo "FAIL GPU non-idle $utilization"; return 1; fi
}
run_block() {
  local label=$1 windows=$2 testname=$3
  idle_gate
  echo "START $label $(date -u +%FT%TZ)"
  GEMM_BI_QUAL_WINDOWS="$windows" cargo test --release --locked --features cuda --test gemm_bi_performance_matrix "$testname" -- --exact --ignored --nocapture --test-threads=1 >"$logdir/$label-v1.log" 2>&1
  echo "END $label EXIT=0 $(date -u +%FT%TZ)"
  sha256sum "$logdir/$label-v1.log"
  sleep 2
}
run_block smoke60 1 gemm_bi_deterministic_performance_matrix
run_block auto60 21 gemm_bi_deterministic_performance_matrix
run_block vendor45 21 gemm_bi_cublas_performance_denominators
idle_gate
