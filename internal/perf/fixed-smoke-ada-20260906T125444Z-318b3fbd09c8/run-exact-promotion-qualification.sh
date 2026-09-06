#!/usr/bin/env bash
set -uo pipefail

repo=/root/mamba-rs-triad
evidence="$repo/internal/perf/fixed-exact-promotions-ada-20260906-318b3fbd09c8"
source_evidence="$repo/internal/perf/fixed-smoke-ada-20260906T125444Z-318b3fbd09c8"
binary="$repo/target-132-fixed-smoke/release/deps/gemm_bi_fixed_performance-fd5155b4b71de901"
expected_binary_sha=d0368f56ed46c35f536ca2331aeb7e1f34bbc03026c9820d409316eb26f05ca7

source /root/triad-env.sh
cd "$repo"
mkdir -p "$evidence"

{
    date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
    printf 'source_git=%s\n' 318b3fbd09c8fe0bfac7eaabefdec2bba457f212
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
export MAMBA_FIXED_ADA_VENDOR=1
export MAMBA_FIXED_VENDOR_EXACT_CC=8.9
export MAMBA_FIXED_ADA_ROWS=f32_exact,f32_exact_fast
export MAMBA_FIXED_ADA_BIAS=0,1
export MAMBA_FIXED_VENDOR_PATHS=eager,graph

gpu_idle() {
    [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]]
}

run_case() {
    local label=$1
    local cells=$2
    local tile=$3
    local windows=$4
    local log="$evidence/${label}.log"
    if ! gpu_idle; then
        printf 'case=%s gpu_preflight=FAIL\n' "$label" | tee -a "$evidence/runner.log"
        return 1
    fi
    export MAMBA_FIXED_ADA_CELLS="$cells"
    export MAMBA_FIXED_VENDOR_TILES="$tile"
    export MAMBA_FIXED_ADA_WINDOWS="$windows"
    printf 'case=%s cells=%s tile=%s windows=%s start_utc=%s\n' \
        "$label" "$cells" "$tile" "$windows" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        | tee -a "$evidence/runner.log"
    "$binary" fixed_ada_forced_rungs_paired_precision_cublas \
        --ignored --exact --nocapture --test-threads=1 \
        2>&1 | tee "$log"
    local status=${PIPESTATUS[0]}
    printf 'case=%s exit=%s end_utc=%s\n' "$label" "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        | tee -a "$evidence/runner.log"
    return "$status"
}

promotion_gate() {
    local label=$1
    local expected_records=$2
    local log="$evidence/${label}.log"
    sed -n '/^{/p' "$log" | jq -s -e --argjson expected "$expected_records" '
        [ .[] | select(.schema == "MambaBiFixedExplicitForcedRungV2") ] as $records
        | [ .[] | select(.schema == "MambaBiFixedExplicitForcedRungCompleteV2") ] as $complete
        | ($records | length) == $expected
          and ($records | all(
              .raw_storage_bits_equal == true
              and .auto_bits_equal == true
              and .repeat_bits_equal == true
              and (.path != "graph" or .graph_replay_bits_equal == true)
              and .forced_over_auto_p50 < 1.0
              and .forced_over_auto_p95 < 1.0
          ))
          and ($complete | length) == 1
          and $complete[0].passed == true
          and $complete[0].rejected == 0
          and $complete[0].records == $expected
    ' >/dev/null
}

overall=0
if run_case ad-copyplan-confirm101 hot_a,hot_d F32Sm89N64CopyPlan 101; then
    if promotion_gate ad-copyplan-confirm101 32; then
        printf 'case=ad-copyplan-confirm101 promotion_gate=PASS\n' | tee -a "$evidence/runner.log"
    else
        printf 'case=ad-copyplan-confirm101 promotion_gate=FAIL\n' | tee -a "$evidence/runner.log"
        overall=1
    fi
else
    overall=1
fi

if run_case c-n128-screen21 hot_c F32N128S2 21; then
    if promotion_gate c-n128-screen21 16; then
        printf 'case=c-n128-screen21 promotion_gate=PASS\n' | tee -a "$evidence/runner.log"
        if run_case c-n128-confirm101 hot_c F32N128S2 101; then
            if promotion_gate c-n128-confirm101 16; then
                printf 'case=c-n128-confirm101 promotion_gate=PASS\n' | tee -a "$evidence/runner.log"
            else
                printf 'case=c-n128-confirm101 promotion_gate=FAIL\n' | tee -a "$evidence/runner.log"
                overall=1
            fi
        else
            overall=1
        fi
    else
        printf 'case=c-n128-screen21 promotion_gate=FAIL confirm101=SKIP\n' | tee -a "$evidence/runner.log"
        overall=1
    fi
else
    overall=1
fi

(
    cd "$evidence"
    find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 sha256sum > SHA256SUMS
)
exit "$overall"
