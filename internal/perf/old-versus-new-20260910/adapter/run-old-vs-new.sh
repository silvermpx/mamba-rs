#!/usr/bin/env bash
# Old main (d8f2efbe) versus the release endpoint: two mirrored blocks
# (old, new, new, old), one process per arm, idle-GPU check before each,
# private kernel caches, every GEMM control cleared. Run on the box as root.
set -euo pipefail
source /root/triad-env.sh
OLD=/root/mamba-rs-main-d8f2efbe
NEW=/root/mamba-rs-new-c65c1c85
EX=m1_exact_f32_inference_baseline
out=/root/monolith-baseline-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$out"
for v in MAMBA_RS_GEMM_MODE MAMBA_RS_BATCH_INVARIANT MAMBA_RS_BI_GEMM_FAMILY MAMBA_RS_BI_TENSOR_CORES MAMBA_RS_FAST_GEMM MAMBA_RS_BI_F32_POLICY MAMBA_RS_BI_HALF_POLICY MAMBA_RS_ARCH_RUNG; do unset $v; done
export CUDA_CACHE_DISABLE=0
{
  echo "date $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=name,driver_version,uuid --format=csv,noheader; nvcc --version | tail -1
  echo "old $OLD $(sha256sum $OLD/examples/$EX.rs | cut -c1-64)"; echo "new $NEW $(sha256sum $NEW/examples/$EX.rs | cut -c1-64)"
  sha256sum $OLD/target/release/examples/$EX $NEW/target/release/examples/$EX
} > "$out/identity.txt"
arm() {  # arm <label> <tree> <block>
  local label=$1 tree=$2 block=$3
  local util mem
  read -r util mem < <(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ',')
  echo "block $block arm $label pre: util=${util}% mem=${mem}MiB" >> "$out/order.txt"
  if [ "$util" -gt 1 ] || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then echo "GPU busy before $label" >> "$out/order.txt"; exit 3; fi
  mkdir -p "$out/cache-$label"; chmod 0700 "$out/cache-$label"
  CUDA_CACHE_PATH="$out/cache-$label" MAMBA_RS_KERNEL_CACHE=off "$tree/target/release/examples/$EX" > "$out/$block-$label.log" 2>&1
  echo "block $block arm $label rc=$?" >> "$out/order.txt"
}
for block in 1 2; do arm old $OLD $block; arm new $NEW $block; arm new $NEW $block; arm old $OLD $block; done
echo "done $(date -u +%FT%TZ)" >> "$out/order.txt"
echo "$out"
