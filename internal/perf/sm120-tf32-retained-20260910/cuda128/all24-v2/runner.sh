#!/usr/bin/env bash
set -euo pipefail
toolkit=${1:?toolkit}
source_dir=${2:?frozen source}
packet=${3:-existing23}
start_group=${4:-G01}
label=${toolkit/./}
case "$packet" in existing23|g10|all24|warmup_fix) ;; *) exit 2 ;; esac
evidence_dir=/root/sm120-tf32-retained-cuda${label}-${packet}-${start_group}-20260910
test ! -e "$evidence_dir"
mkdir -p "$evidence_dir"
cd "$source_dir"
export CUDA_HOME=/usr/local/cuda-$toolkit
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export CARGO_TARGET_DIR=/root/target-sm120-final-sweep-cuda${label}
export MAMBA_RS_KERNEL_CACHE=/root/sm120-release-matrix-cuda${label}-20260910/kernel-cache
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_PATH=/root/sm120-qualified-jit-cuda${label}-20260910
export CUDA_CACHE_MAXSIZE=1073741824
mkdir -p "$CUDA_CACHE_PATH" "$MAMBA_RS_KERNEL_CACHE"
export CUDA_VISIBLE_DEVICES=GPU-a10ad830-a2cf-054d-e0fa-54add30a2cf8
export MAMBA_RS_SM120_TF32_SELECTOR_QUALIFICATION=1
unset NVIDIA_TF32_OVERRIDE
nvidia-smi --query-gpu=uuid,name,driver_version,memory.total --format=csv > "$evidence_dir/device.csv"
nvcc --version > "$evidence_dir/nvcc-version.txt"
find src kernels tests -type f -print0 | sort -z | xargs -0 sha256sum > "$evidence_dir/assembly-source.sha256"
sha256sum "$0" > "$evidence_dir/runner.sha256"
cp "$0" "$evidence_dir/runner.sh"
test_target=gemm_bi_sm120_tf32_selector_qualification
test_case=sm120_tf32_projection_selector_qualification
cargo test --locked --release --features cuda --test "$test_target" --no-run > "$evidence_dir/build.log" 2>&1
cargo test --locked --release --features cuda --test "$test_target" -- --list > "$evidence_dir/tests.list" 2>&1
grep -Fx "$test_case: test" "$evidence_dir/tests.list"
if [[ "$packet" != existing23 ]]; then
    mapping_test=tn_m8192_k128_n128_maps_to_exact_tn_key
    grep -Fx "$mapping_test: test" "$evidence_dir/tests.list"
    cargo test --locked --release --features cuda --test "$test_target" "$mapping_test" -- --exact --nocapture > "$evidence_dir/g10-mapping.log" 2>&1
    grep -F 'test result: ok. 1 passed;' "$evidence_dir/g10-mapping.log"
fi
if [[ "$packet" == warmup_fix || "$packet" == all24 ]]; then
    for host_test in runtime_qualification_requires_gpu_quiet_gates post_quiet_paired_warmup_order_is_fixed_and_symmetric completion_schema_is_stable_and_machine_readable; do
        grep -Fx "$host_test: test" "$evidence_dir/tests.list"
        cargo test --locked --release --features cuda --test "$test_target" "$host_test" -- --exact --nocapture > "$evidence_dir/$host_test.log" 2>&1
        grep -F 'test result: ok. 1 passed;' "$evidence_dir/$host_test.log"
    done
fi
binary=$(sed -n 's/^  Executable .* (\(.*\))$/\1/p' "$evidence_dir/build.log")
test -x "$binary"
sha256sum "$binary" > "$evidence_dir/test-binary.sha256"
preflight() {
    sleep 2
    for sample in 1 2 3 4 5; do
        telemetry=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.free --format=csv,noheader,nounits)
        printf '%s\n' "$telemetry"
        awk -F, 'NF != 3 || $1+0 > 1 || $2+0 > 1 || $3+0 < 2048 {exit 1}' <<< "$telemetry"
        if [[ "$sample" != 5 ]]; then sleep 1; fi
    done
}
for group in G01 G02 G03 G04 G05 G06 G07 G08 G09 G10 G11; do
    if [[ "$group" < "$start_group" ]]; then continue; fi
    if [[ "$packet" == existing23 && "$group" == G10 ]]; then continue; fi
    if [[ "$packet" == g10 && "$group" != G10 ]]; then continue; fi
    if [[ "$packet" == warmup_fix && "$group" != G06 && "$group" != G09 ]]; then continue; fi
    case "$group" in
        G01) cells=nn_d768_in_proj,nn_large_deep; symbol=gemm_bi_nn_sm120_tma_mma_tf32_v1_m64n128_bk32_s2 ;;
        G02) cells=nn_d768_out_proj,nn_prism_in_proj,nn_large,nn_batch_in_proj; symbol=gemm_bi_nn_sm120_tma_mma_tf32_v1_m64n64_bk32_s2 ;;
        G03) cells=tn_d768_in_proj,tn_d768_out_proj,tn_prism_in_proj,tn_large_deep,tn_large; symbol=gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s3_pair_streamk ;;
        G04) cells=nt_d768_in_proj,nt_d768_out_proj,nt_prism_in_proj; symbol=gemm_bi_nt_sm120_tma_mma_tf32_v1_m64n64_bk32_s2 ;;
        G05) cells=nt_large_deep,nt_large,nt_batch_in_proj; symbol=gemm_bi_nt_sm120_tma_mma_tf32_v1_m64n128_bk32_s2 ;;
        G06) cells=nn_d128_in_proj; symbol=gemm_bi_nn_sm80_mma_tf32_v1_m64n64_bk32_s3 ;;
        G07) cells=tn_d128_in_proj,tn_underfill; symbol=gemm_bi_tn_sm80_mma_tf32_v1_m16n32_bk32_s4 ;;
        G08) cells=nt_d128_in_proj; symbol=gemm_bi_nt_sm80_mma_tf32_v1_m16n16_bk32_s4 ;;
        G09) cells=tn_d128_out_proj; symbol=gemm_bi_tn_sm80_mma_tf32_v1_m16n16_bk32_s4 ;;
        G10) cells=tn_m8192_k128_n128; symbol=gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_pair ;;
        G11) cells=nt_split_candidate; symbol=gemm_bi_nt_sm80_mma_tf32_v1_m64n64_bk32_s2 ;;
    esac
    export MAMBA_RS_SM120_TF32_SELECTOR_CELLS="$cells"
    export MAMBA_RS_TF32_SELECTOR_CANDIDATES="$symbol"
    export MAMBA_RS_SM120_TF32_SELECTOR_JSONL="$evidence_dir/$group.jsonl"
    test ! -e "$MAMBA_RS_SM120_TF32_SELECTOR_JSONL"
    printf '%s %s %s\n' "$group" "$cells" "$symbol" >> "$evidence_dir/selection.txt"
    preflight > "$evidence_dir/$group.preflight.log"
    date -u +%FT%TZ > "$evidence_dir/$group.start.txt"
    set +e
    cargo test --locked --release --features cuda --test "$test_target" "$test_case" -- --exact --ignored --nocapture --test-threads=1 > "$evidence_dir/$group.log" 2>&1
    result=$?
    set -e
    date -u +%FT%TZ > "$evidence_dir/$group.end.txt"
    printf '%s\n' "$result" > "$evidence_dir/$group.exit.txt"
    test "$result" = 0
    grep -F 'test result: ok. 1 passed;' "$evidence_dir/$group.log"
    expected=$(awk -F, '{print NF}' <<< "$cells")
    actual=$(wc -l < "$MAMBA_RS_SM120_TF32_SELECTOR_JSONL")
    # JsonlSink writes one final completion record after the cell records.
    test "$actual" -eq "$((expected + 1))"
done
date -u +%FT%TZ > "$evidence_dir/completed.txt"
