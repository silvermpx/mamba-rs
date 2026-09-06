#!/usr/bin/env bash
set -uo pipefail

repo=/root/mamba-rs-triad
evidence="$repo/internal/perf/fixed-full-ada-20260906-318b3fbd09c8"
source_evidence="$repo/internal/perf/fixed-smoke-ada-20260906T125444Z-318b3fbd09c8"
binary="$repo/target-132-fixed-smoke/release/deps/gemm_bi_fixed_performance-fd5155b4b71de901"
expected_binary_sha=d0368f56ed46c35f536ca2331aeb7e1f34bbc03026c9820d409316eb26f05ca7

source /root/triad-env.sh
cd "$repo"
mkdir -p "$evidence"

{
    date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
    printf 'source_git=%s\n' 318b3fbd09c8fe0bfac7eaabefdec2bba457f212
    printf 'rows=%s\n' bf16,f16,bf16_f32,f16_f32,tf32
    printf 'cells=all bias=all tiles=all-sm89 paths=eager,graph windows=21\n'
    sha256sum "$binary"
    sha256sum -c --quiet "$source_evidence/manifest.sha256"
} 2>&1 | tee "$evidence/runner.log"
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    printf 'source_or_binary_preflight=FAIL\n' | tee -a "$evidence/runner.log"
    exit 1
fi

actual_binary_sha=$(sha256sum "$binary" | awk '{print $1}')
if [[ "$actual_binary_sha" != "$expected_binary_sha" ]]; then
    printf 'binary_hash=FAIL expected=%s actual=%s\n' "$expected_binary_sha" "$actual_binary_sha" | tee -a "$evidence/runner.log"
    exit 1
fi

unset NVIDIA_TF32_OVERRIDE
unset MAMBA_FIXED_VENDOR_TILES
unset MAMBA_FIXED_ADA_CELLS
unset MAMBA_FIXED_ADA_BIAS
export MAMBA_FIXED_ADA_VENDOR=1
export MAMBA_FIXED_VENDOR_EXACT_CC=8.9
export MAMBA_FIXED_ADA_WINDOWS=21
export MAMBA_FIXED_VENDOR_PATHS=eager,graph

overall=0
for row in bf16 f16 bf16_f32 f16_f32 tf32; do
    compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')
    if [[ -n "$compute_pids" ]]; then
        printf 'row=%s gpu_preflight=FAIL compute_pids=%s\n' "$row" "$compute_pids" | tee -a "$evidence/runner.log"
        overall=1
        break
    fi
    export MAMBA_FIXED_ADA_ROWS="$row"
    printf 'row=%s start_utc=%s\n' "$row" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$evidence/runner.log"
    "$binary" fixed_ada_forced_rungs_paired_precision_cublas \
        --ignored --exact --nocapture --test-threads=1 \
        2>&1 | tee "$evidence/${row}-full21.log"
    status=${PIPESTATUS[0]}
    printf 'row=%s exit=%s end_utc=%s\n' "$row" "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$evidence/runner.log"
    if [[ $status -ne 0 ]]; then
        overall=1
    fi
done

(
    cd "$evidence"
    find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 sha256sum > SHA256SUMS
)
exit "$overall"
