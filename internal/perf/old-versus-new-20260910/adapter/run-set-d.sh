#!/usr/bin/env bash
# Set D: kernel-level GEMM timing, old main against the release, each with
# its own cuBLAS Fast and Pedantic arms in the same process. Two mirrored
# blocks (old, new, new, old) of separate processes with private caches.
set -uo pipefail
source /root/triad-env.sh
OLD=/root/mamba-rs-main-d8f2efbe
NEW=/root/mamba-rs-new-c65c1c85
out=/root/monolith-baseline-setd-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$out"
for v in MAMBA_RS_GEMM_MODE MAMBA_RS_BATCH_INVARIANT MAMBA_RS_BI_GEMM_FAMILY MAMBA_RS_BI_TENSOR_CORES MAMBA_RS_FAST_GEMM MAMBA_RS_BI_F32_POLICY MAMBA_RS_BI_HALF_POLICY MAMBA_RS_ARCH_RUNG; do unset $v; done
export CUDA_CACHE_DISABLE=0
log() { echo "$(date -u +%H:%M:%S) $*" >> "$out/order.txt"; }
log "build old"
(cd $OLD && CARGO_TARGET_DIR=$OLD/target cargo build --release --features cuda --example gemm_kernel_old_vs_new > "$out/build-old.log" 2>&1) || { log "build old FAILED"; exit 2; }
log "build new"
(cd $NEW && CARGO_TARGET_DIR=$NEW/target cargo build --release --features cuda --example gemm_kernel_old_vs_new > "$out/build-new.log" 2>&1) || { log "build new FAILED"; exit 2; }
{
  echo "date $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=name,driver_version,uuid --format=csv,noheader; nvcc --version | tail -1
  echo "old gemm_kernel_old_vs_new $(sha256sum $OLD/examples/gemm_kernel_old_vs_new.rs | cut -c1-64)"
  echo "new gemm_kernel_old_vs_new $(sha256sum $NEW/examples/gemm_kernel_old_vs_new.rs | cut -c1-64)"
  sha256sum $OLD/target/release/examples/gemm_kernel_old_vs_new $NEW/target/release/examples/gemm_kernel_old_vs_new
} > "$out/identity.txt"
quiet() {
  local util mem q=0 tries=0
  while [ "$q" -lt 3 ]; do
    read -r util mem < <(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ',')
    if [ "$util" -le 1 ] && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then q=$((q+1)); else q=0; fi
    tries=$((tries+1)); [ "$tries" -gt 120 ] && { log "GPU busy: util=${util}% mem=${mem}MiB"; exit 3; }
    sleep 1
  done
}
arm() {
  local set=$1 block=$2 pos=$3 label=$4; shift 4
  quiet
  local cache="$out/cache-$label"; mkdir -p "$cache/driver" "$cache/kernels"; chmod 0700 "$cache" "$cache/driver" "$cache/kernels"
  log "set $set block $block pos $pos arm $label start"
  CUDA_CACHE_PATH="$cache/driver" MAMBA_RS_KERNEL_CACHE="$cache/kernels" "$@" > "$out/$set-$block-$pos-$label.log" 2>&1
  log "set $set block $block pos $pos arm $label rc=$?"
}
for block in 1 2; do
  arm D $block 1 old $OLD/target/release/examples/gemm_kernel_old_vs_new
  arm D $block 2 new $NEW/target/release/examples/gemm_kernel_old_vs_new
  arm D $block 3 new $NEW/target/release/examples/gemm_kernel_old_vs_new
  arm D $block 4 old $OLD/target/release/examples/gemm_kernel_old_vs_new
done
log "done"
echo "$out"
