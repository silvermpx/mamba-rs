#!/usr/bin/env bash
set -uo pipefail
set -C
toolkit=$1
tag=$2
feature=$3
attempt=${4:-1}
export CUDA_HOME=/usr/local/cuda-$toolkit CUDA_PATH=/usr/local/cuda-$toolkit
export LD_LIBRARY_PATH=$CUDA_HOME/lib64 PATH=$CUDA_HOME/bin:/root/.cargo/bin:$PATH
export CARGO_TARGET_DIR=/root/target-ada-half-s3-force-cuda${tag}-20260907
export MAMBA_RS_KERNEL_CACHE=/root/mamba-kcache-ada-half-s3-force-final-cuda${tag}-attempt${attempt}-20260907
cd /root/mamba-ada-half-s3-force-20260907 || exit $?
evidence=/root/evidence-ada-half-s3-force-20260907/cuda${tag}-attempt${attempt}
mkdir -p /root/evidence-ada-half-s3-force-20260907 || exit $?
mkdir "$evidence" || exit $?
mkdir -m 700 "$MAMBA_RS_KERNEL_CACHE" || exit $?
run() {
    local label=$1
    shift
    date -u +%FT%TZ
    printf 'COMMAND %q ' "$@"
    printf '\n'
    "$@" > "$evidence/$label.log" 2>&1
    local status=$?
    printf 'EXIT %s %d\n' "$label" "$status"
    tail -8 "$evidence/$label.log"
    if [ "$status" -ne 0 ]; then return "$status"; fi
}
testcmd=(cargo test --release --features cuda,cudarc/cuda-${feature})
run identity bash -euo pipefail -c 'hostname; echo EXIT_hostname=0; date -u; echo EXIT_date=0; nvidia-smi -q; echo EXIT_nvidia_smi=0; nvcc --version; echo EXIT_nvcc=0; rustc -Vv; echo EXIT_rustc=0; cargo -V; echo EXIT_cargo=0; sha256sum "$CUDA_HOME/bin/ptxas"; echo EXIT_ptxas_hash=0; shopt -s nullglob; libs=("$CUDA_HOME"/targets/x86_64-linux/lib/libnvrtc.so.*); test ${#libs[@]} -ge 1; sha256sum "${libs[@]}"; echo EXIT_nvrtc_hashes=0' || exit $?
run library "${testcmd[@]}" --lib || exit $?
run force-static "${testcmd[@]}" --test gemm_bi_fixed_performance || exit $?
run source "${testcmd[@]}" --test arch_compile_gates fixed_sm89_half_ -- --nocapture || exit $?
run compile-sm89 "${testcmd[@]}" --test arch_compile_gates compiles_for_sm89 -- --exact --nocapture || exit $?
run cold "${testcmd[@]}" --test gemm_bi_fixed_sm89_pipeline fixed_sm89_half_swizzle_and_pipeline_holders_are_independently_live -- --ignored --exact --nocapture || exit $?
run warm "${testcmd[@]}" --test gemm_bi_fixed_sm89_pipeline fixed_sm89_half_swizzle_and_pipeline_holders_are_independently_live -- --ignored --exact --nocapture || exit $?
run half-full "${testcmd[@]}" --test gemm_bi_fixed_sm89_pipeline -- --include-ignored --test-threads=1 --nocapture || exit $?
run retained-rna "${testcmd[@]}" --test gemm_bi_fixed_correctness fixed_sm89_rna_wide_actual_auto -- --ignored --test-threads=1 --nocapture || exit $?
run retained-exact "${testcmd[@]}" --test gemm_bi_fixed_sm89_exact_n64 -- --include-ignored --test-threads=1 --nocapture || exit $?
run retained-tf32-c "${testcmd[@]}" --test gemm_bi_fixed_performance fixed_sm89_tf32_c_auto_prefix_special_bias_graph_bits -- --ignored --exact --nocapture || exit $?
run eager-physical env MAMBA_FIXED_ADA_VENDOR=1 MAMBA_FIXED_VENDOR_EXACT_CC=8.9 MAMBA_FIXED_ADA_ROWS=bf16,f16 MAMBA_FIXED_ADA_CELLS=hot_a MAMBA_FIXED_ADA_BIAS=0,1 MAMBA_FIXED_ADA_WINDOWS=1 MAMBA_FIXED_VENDOR_TILES=Tc128Sm89S3 MAMBA_FIXED_VENDOR_PATHS=eager "${testcmd[@]}" --test gemm_bi_fixed_performance fixed_ada_forced_rungs_paired_precision_cublas -- --ignored --exact --nocapture || exit $?
if [ "$tag" = 132 ]; then
    run triad-sm120 "${testcmd[@]}" --test arch_compile_gates compiles_for_sm120 -- --exact --nocapture || exit $?
    run triad-sm120-generic "${testcmd[@]}" --test arch_compile_gates compiles_generic_sm120_triad_modules_with_exact_ptx_contract -- --exact --nocapture || exit $?
    run retained-cohort "${testcmd[@]}" --test gemm_bi_tf32_cohort_binding tf32_cohort_binds_on_this_board -- --ignored --exact --nocapture || exit $?
    run retained-bias-cohort "${testcmd[@]}" --test gemm_bi_tf32_cohort_binding sm89_tf32_bias_cohort_serves_the_qualified_wide_epilogues -- --ignored --exact --nocapture || exit $?
fi
mapfile -t pipeline_bins < <(find "$CARGO_TARGET_DIR/release/deps" -maxdepth 1 -type f -executable -name 'gemm_bi_fixed_sm89_pipeline-*')
if [ "${#pipeline_bins[@]}" -ne 1 ]; then printf 'ambiguous/missing pipeline binary count=%s\n' "${#pipeline_bins[@]}"; exit 98; fi
pipeline_bin=${pipeline_bins[0]}
run sanitizer-binary sha256sum "$pipeline_bin" || exit $?
for tool in memcheck racecheck synccheck; do
    run sanitizer-$tool compute-sanitizer --tool "$tool" --report-api-errors no --error-exitcode 99 "$pipeline_bin" --ignored --exact fixed_sm89_half_pipeline_sanitizer_smoke --nocapture || exit $?
done
fixed_cache=
for entry in "$MAMBA_RS_KERNEL_CACHE"/*.bin; do
    if grep -aq 'gemm_bi_nn_fixed_sm89_tc128_s3_v1_bf16' "$entry"; then
        if [ -n "$fixed_cache" ]; then printf 'ambiguous Fixed cache\n'; exit 98; fi
        fixed_cache=$entry
    fi
done
if [ -z "$fixed_cache" ]; then printf 'missing Fixed cache\n'; exit 98; fi
tail -c +92 "$fixed_cache" > "$evidence/fixed.ptx" || exit $?
run ptxas ptxas -arch=sm_89 -O3 -v "$evidence/fixed.ptx" -o "$evidence/fixed.cubin" || exit $?
run resources cuobjdump --dump-resource-usage "$evidence/fixed.cubin" || exit $?
run sass cuobjdump --dump-sass "$evidence/fixed.cubin" || exit $?
run artifact-proof ruby analyze-artifact.rb "$fixed_cache" "$evidence/fixed.ptx" "$evidence/sass.log" "$evidence/resources.log" "$evidence/ptxas.log" 188 || exit $?
run identities-final bash -euo pipefail -c 'find src kernels tests -type f -print0 | sort -z | xargs -0 sha256sum; echo EXIT_source_hashes=0; sha256sum Cargo.toml Cargo.lock build.rs; echo EXIT_build_input_hashes=0; find "$CARGO_TARGET_DIR/release/deps" -maxdepth 1 -type f -executable -print0 | sort -z | xargs -0 sha256sum; echo EXIT_binary_hashes=0; find "$MAMBA_RS_KERNEL_CACHE" -type f -print0 | sort -z | xargs -0 sha256sum; echo EXIT_cache_hashes=0' || exit $?
date -u +%FT%TZ
printf 'TOOLKIT_COMPLETE %s\n' "$toolkit"
