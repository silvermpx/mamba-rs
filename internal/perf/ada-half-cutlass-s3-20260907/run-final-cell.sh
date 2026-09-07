#!/bin/sh
set -eu

dtype=${1:-bf16}
windows=${2:-21}
case "$dtype:$windows" in bf16:21|bf16:101|f16:21|f16:101) ;; *) exit 2 ;; esac

source_root=/root/mamba-ada-half-cutlass-s3-20260907
target_root=/root/target-ada-half-cutlass-s3-20260907-final
evidence_root="$source_root/internal/perf/ada-half-cutlass-s3-20260907"
binary="$target_root/benchmark"
raw="$target_root/paired${windows}-${dtype}-cuda132.log"
body="$target_root/paired${windows}-${dtype}-body.log"
pre="$target_root/paired${windows}-${dtype}-pre.log"
immediate="$target_root/paired${windows}-${dtype}-post-immediate.log"
final="$target_root/paired${windows}-${dtype}-post-final.log"
expected_uuid=GPU-d1edd7be-e88d-aed6-047d-622163306f0e

export CUDA_HOME=/usr/local/cuda-13.2
export PATH="/usr/local/cuda-13.2/bin:/root/.cargo/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}"

gpu_state() {
    nvidia-smi --query-gpu=uuid,utilization.gpu,utilization.memory \
        --format=csv,noheader,nounits
}

compute_apps() {
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
        --format=csv,noheader,nounits
}

pre_state=$(gpu_state)
pre_apps=$(compute_apps)
test "$pre_state" = "$expected_uuid, 0, 0"
test -z "$pre_apps"
{
    date -u +UTC=%Y-%m-%dT%H:%M:%SZ
    nvidia-smi --query-gpu=uuid,name,driver_version,utilization.gpu,utilization.memory,temperature.gpu,pstate,clocks.current.sm,power.draw,memory.used --format=csv,noheader,nounits
    printf 'COMPUTE_APPS_BEGIN\n%s\nCOMPUTE_APPS_END\n' "$pre_apps"
    cat /proc/loadavg
} > "$pre"

set +e
"$binary" "$dtype" 4621 768 2304 0 --warmup 128 --graph-ops 20 --windows "$windows" > "$body" 2>&1
test_status=$?
set -e

{
    date -u +UTC=%Y-%m-%dT%H:%M:%SZ
    nvidia-smi --query-gpu=uuid,name,driver_version,utilization.gpu,utilization.memory,temperature.gpu,pstate,clocks.current.sm,power.draw,memory.used --format=csv,noheader,nounits
    printf 'COMPUTE_APPS_BEGIN\n'
    compute_apps
    printf 'COMPUTE_APPS_END\n'
} > "$immediate"

quiet=0
attempt=0
while test "$attempt" -lt 60; do
    post_state=$(gpu_state)
    post_apps=$(compute_apps)
    if test "$post_state" = "$expected_uuid, 0, 0" && test -z "$post_apps"; then
        quiet=1
        break
    fi
    attempt=$((attempt + 1))
    sleep 1
done
{
    date -u +UTC=%Y-%m-%dT%H:%M:%SZ
    nvidia-smi --query-gpu=uuid,name,driver_version,utilization.gpu,utilization.memory,temperature.gpu,pstate,clocks.current.sm,power.draw,memory.used --format=csv,noheader,nounits
    printf 'COMPUTE_APPS_BEGIN\n%s\nCOMPUTE_APPS_END\n' "$post_apps"
    printf 'QUIET_POLL_ATTEMPTS=%d\n' "$attempt"
} > "$final"

{
    cat "$pre"
    cat "$target_root/ptxas-resources-cuda132.jsonl"
    cat "$body"
    printf 'TEST_EXIT=%d\n' "$test_status"
    printf 'POST_IMMEDIATE_BEGIN\n'
    cat "$immediate"
    printf 'POST_IMMEDIATE_END\n'
    printf 'POST_FINAL_BEGIN\n'
    cat "$final"
    printf 'POST_FINAL_END\n'
    printf 'TELEMETRY_EXIT=%d\n' "$((1 - quiet))"
} > "$raw"

test "$test_status" -eq 0
test "$quiet" -eq 1
cat "$raw"
