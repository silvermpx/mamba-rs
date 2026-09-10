#!/usr/bin/env bash
# Old main (d8f2efbe) versus the release endpoint on one board. Builds every
# instrument first, then times three sets, each as mirrored blocks
# (old, new, new, old) of separate processes with private caches, an idle
# check before each process and every GEMM control cleared:
#   A. exact-F32 inference step on the training family (common lane)
#   B. inference step on the inference family, f32 and bf16
#   C. training step: cuBLAS Fast, cuBLAS Pedantic, deterministic scalar, +TC
set -uo pipefail
source /root/triad-env.sh
OLD=/root/mamba-rs-main-d8f2efbe
NEW=/root/mamba-rs-new-c65c1c85
out=/root/monolith-baseline-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$out"
for v in MAMBA_RS_GEMM_MODE MAMBA_RS_BATCH_INVARIANT MAMBA_RS_BI_GEMM_FAMILY MAMBA_RS_BI_TENSOR_CORES MAMBA_RS_FAST_GEMM MAMBA_RS_BI_F32_POLICY MAMBA_RS_BI_HALF_POLICY MAMBA_RS_ARCH_RUNG; do unset $v; done
export CUDA_CACHE_DISABLE=0
log() { echo "$(date -u +%H:%M:%S) $*" >> "$out/order.txt"; }
build() {
  log "build old"
  (cd $OLD && CARGO_TARGET_DIR=$OLD/target cargo build --release --features cuda --example m1_exact_f32_inference_baseline --example m1_inference_family_baseline > "$out/build-old.log" 2>&1 && CARGO_TARGET_DIR=$OLD/target cargo test --release --features cuda --test gemm_bi_determinism --no-run >> "$out/build-old.log" 2>&1) || { log "build old FAILED"; exit 2; }
  log "build new"
  (cd $NEW && CARGO_TARGET_DIR=$NEW/target cargo build --release --features cuda --example m1_exact_f32_inference_baseline --example m1_inference_family_baseline > "$out/build-new.log" 2>&1 && CARGO_TARGET_DIR=$NEW/target cargo bench --features cuda --bench gemm_bi_trainer_step_bench --no-run >> "$out/build-new.log" 2>&1) || { log "build new FAILED"; exit 2; }
  {
    echo "date $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=name,driver_version,uuid --format=csv,noheader; nvcc --version | tail -1
    for f in m1_exact_f32_inference_baseline m1_inference_family_baseline; do echo "old $f $(sha256sum $OLD/examples/$f.rs | cut -c1-64)"; echo "new $f $(sha256sum $NEW/examples/$f.rs | cut -c1-64)"; done
    echo "new bench $(sha256sum $NEW/benches/gemm_bi_trainer_step_bench.rs | cut -c1-64)"
    sha256sum $OLD/target/release/examples/m1_exact_f32_inference_baseline $NEW/target/release/examples/m1_exact_f32_inference_baseline $OLD/target/release/examples/m1_inference_family_baseline $NEW/target/release/examples/m1_inference_family_baseline
  } > "$out/identity.txt"
}
quiet() {
  local util mem q=0 tries=0
  while [ "$q" -lt 3 ]; do
    read -r util mem < <(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ',')
    if [ "$util" -le 1 ] && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then q=$((q+1)); else q=0; fi
    tries=$((tries+1)); [ "$tries" -gt 120 ] && { log "GPU busy: util=${util}% mem=${mem}MiB"; exit 3; }
    sleep 1
  done
}
arm() {  # arm <set> <block> <pos> <label> <command...>
  local set=$1 block=$2 pos=$3 label=$4; shift 4
  quiet
  local cache="$out/cache-$label"; mkdir -p "$cache/driver" "$cache/kernels"; chmod 0700 "$cache" "$cache/driver" "$cache/kernels"
  log "set $set block $block pos $pos arm $label start"
  CUDA_CACHE_PATH="$cache/driver" MAMBA_RS_KERNEL_CACHE="$cache/kernels" "$@" > "$out/$set-$block-$pos-$label.log" 2>&1
  log "set $set block $block pos $pos arm $label rc=$?"
}
build
for block in 1 2; do
  arm A $block 1 old $OLD/target/release/examples/m1_exact_f32_inference_baseline
  arm A $block 2 new $NEW/target/release/examples/m1_exact_f32_inference_baseline
  arm A $block 3 new $NEW/target/release/examples/m1_exact_f32_inference_baseline
  arm A $block 4 old $OLD/target/release/examples/m1_exact_f32_inference_baseline
done
for block in 1 2; do
  arm B $block 1 old $OLD/target/release/examples/m1_inference_family_baseline
  arm B $block 2 new $NEW/target/release/examples/m1_inference_family_baseline
  arm B $block 3 new $NEW/target/release/examples/m1_inference_family_baseline
  arm B $block 4 old $OLD/target/release/examples/m1_inference_family_baseline
done
oldtest="$(ls -t $OLD/target/release/deps/gemm_bi_determinism-* | grep -v '\.d$' | head -1)"
newbench="$(ls -t $NEW/target/release/deps/gemm_bi_trainer_step_bench-* | grep -v '\.d$' | head -1)"
log "training binaries: $oldtest $newbench"
for block in 1; do
  arm C $block 1 old "$oldtest" bench_sgemm_bi_vs_tf32 --ignored --nocapture --test-threads=1
  arm C $block 2 new "$newbench" bench_gemm_bi_vs_tf32
  arm C $block 3 new "$newbench" bench_gemm_bi_vs_tf32
  arm C $block 4 old "$oldtest" bench_sgemm_bi_vs_tf32 --ignored --nocapture --test-threads=1
done
log "done"
echo "$out"
