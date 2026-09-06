#!/usr/bin/env bash

set -u
set -o pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
. "$script_dir/run-direct-pair-v2.sh"

quiet='2026/09/06 23:12:47.998, GPU-d1edd7be-e88d-aed6-047d-622163306f0e, NVIDIA RTX 6000 Ada Generation, 8.9, 0 %, 0 %, 90 MiB, 1800 MHz, 810 MHz, 40.08 W, 39, P5, Default'
residual='2026/09/06 23:14:10.133, GPU-d1edd7be-e88d-aed6-047d-622163306f0e, NVIDIA RTX 6000 Ada Generation, 8.9, 14 %, 0 %, 90 MiB, 1800 MHz, 10001 MHz, 260.07 W, 54, P0, Default'
memory_busy='2026/09/06 23:14:10.133, GPU-d1edd7be-e88d-aed6-047d-622163306f0e, NVIDIA RTX 6000 Ada Generation, 8.9, 0 %, 14 %, 90 MiB, 1800 MHz, 10001 MHz, 260.07 W, 54, P0, Default'
wrong_identity='2026/09/06 23:14:10.133, GPU-ffffffff-ffff-ffff-ffff-ffffffffffff, NVIDIA RTX 6000 Ada Generation, 8.9, 0 %, 0 %, 90 MiB, 1800 MHz, 810 MHz, 40.00 W, 39, P5, Default'
assertions=0

expect_status() {
    local expected=$1
    shift
    local actual=0
    "$@" >/dev/null 2>&1 || actual=$?
    if [ "$actual" -ne "$expected" ]; then
        echo "expected status $expected, got $actual: $*" >&2
        exit 1
    fi
    assertions=$((assertions + 1))
}

expect_status 0 direct_pair_validate_gpu_snapshot PRE "$quiet" 0
expect_status 0 direct_pair_validate_gpu_snapshot POST "$quiet" 0
expect_status 71 direct_pair_validate_gpu_snapshot PRE "$residual" 0
expect_status 0 direct_pair_validate_gpu_snapshot POST "$residual" 0
expect_status 71 direct_pair_validate_gpu_snapshot PRE "$memory_busy" 0
expect_status 0 direct_pair_validate_gpu_snapshot POST "$memory_busy" 0
expect_status 71 direct_pair_validate_gpu_snapshot PRE "$wrong_identity" 0
expect_status 71 direct_pair_validate_gpu_snapshot POST "$wrong_identity" 0
expect_status 9 direct_pair_validate_gpu_snapshot PRE "$quiet" 9
expect_status 9 direct_pair_validate_gpu_snapshot POST "$residual" 9

echo "PASS assertions=$assertions"
