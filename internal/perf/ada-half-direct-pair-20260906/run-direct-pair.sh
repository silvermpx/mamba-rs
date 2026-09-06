#!/usr/bin/env bash

set -u
set -o pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <cuda128|cuda130> <21|101>" >&2
    exit 64
fi

toolkit=$1
windows=$2
case "$windows" in
    21 | 101) ;;
    *)
        echo "windows must be 21 or 101, got $windows" >&2
        exit 64
        ;;
esac

source_root=/root/mamba-ada-half-direct-pair-20260906
source_sha=3873a40ac14a05615c5758bb3181948e6e4944486400387552de08a8ecb90d3a
case "$toolkit" in
    cuda128)
        cuda_root=/usr/local/cuda-12.8
        target_dir=/root/target-ada-half-direct-pair-cuda128-20260906
        cache_dir=/root/mamba-kcache-ada-half-direct-pair-cuda128-20260906
        binary="$target_dir/release/deps/gemm_bi_fixed_performance-e01d837be76567b7"
        binary_sha=b887ffbd2d5c91ebd7cd1dcd8eeb3daad5c9400f5b967f58dc46150328067a78
        control="$source_root/identity-cuda128.json"
        control_sha=8ab62c7f1da94b7f61ef81efac7c4d54ae8f98830748e6e8bee27e94558763e3
        ;;
    cuda130)
        cuda_root=/usr/local/cuda-13.0
        target_dir=/root/target-ada-half-direct-pair-cuda130-20260906
        cache_dir=/root/mamba-kcache-ada-half-direct-pair-cuda130-20260906
        binary="$target_dir/release/deps/gemm_bi_fixed_performance-28fdf8ce2728ddd5"
        binary_sha=d21a1a3cb284979289e05f042529be7f98f924e79147aa527766812f794540d5
        control="$source_root/identity-cuda130.json"
        control_sha=0489790af9824b9459b4fe9f56641fd884052e70c2b8c0b946aa9252eeaef301
        ;;
    *)
        echo "toolkit must be cuda128 or cuda130, got $toolkit" >&2
        exit 64
        ;;
esac

nvidia_smi=${DIRECT_PAIR_NVIDIA_SMI:-/usr/bin/nvidia-smi}

telemetry() {
    local phase=$1
    local smi_exit=0
    local apps_exit=0
    local driver_exit=0
    local date_exit=0
    local gpu_output
    local apps_output

    echo "${phase}_TELEMETRY"
    date --iso-8601=ns || date_exit=$?
    gpu_output=$("$nvidia_smi" \
        --query-gpu=timestamp,uuid,name,compute_cap,utilization.gpu,utilization.memory,memory.used,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,pstate,compute_mode \
        --format=csv,noheader 2>&1) || smi_exit=$?
    printf '%s\n' "$gpu_output"
    if [ "$smi_exit" -eq 0 ] && ! printf '%s\n' "$gpu_output" | grep -Eq 'GPU-d1edd7be-e88d-aed6-047d-622163306f0e, NVIDIA RTX 6000 Ada Generation, 8\.9, 0 %, 0 %,'; then
        echo "${phase}_GPU_NOT_QUIET_OR_WRONG_IDENTITY"
        smi_exit=71
    fi
    echo "${phase}_ACTIVE_COMPUTE_APPS"
    apps_output=$("$nvidia_smi" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1) || apps_exit=$?
    printf '%s\n' "$apps_output"
    if [ "$apps_exit" -eq 0 ] && [ -n "$apps_output" ]; then
        echo "${phase}_ACTIVE_COMPUTE_APPS_PRESENT"
        apps_exit=72
    fi
    LD_LIBRARY_PATH="$cuda_root/lib64" /usr/bin/python3 - <<'PY' || driver_exit=$?
import ctypes
import sys

cuda = ctypes.CDLL("libcuda.so.1")
cuda.cuInit.argtypes = [ctypes.c_uint]
cuda.cuInit.restype = ctypes.c_int
cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
cuda.cuDeviceGet.restype = ctypes.c_int
cuda.cuDeviceGetAttribute.argtypes = [
    ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int
]
cuda.cuDeviceGetAttribute.restype = ctypes.c_int

def checked(code, operation):
    if code != 0:
        raise RuntimeError(f"{operation} failed with CUresult {code}")

checked(cuda.cuInit(0), "cuInit")
device = ctypes.c_int()
checked(cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")

def attribute(number):
    value = ctypes.c_int()
    checked(
        cuda.cuDeviceGetAttribute(ctypes.byref(value), number, device.value),
        f"cuDeviceGetAttribute({number})",
    )
    return value.value

major = attribute(75)
minor = attribute(76)
sms = attribute(16)
print(f"DRIVER_CC_SM_COUNT={major}.{minor},{sms}")
if (major, minor, sms) != (8, 9, 142):
    sys.exit(f"unexpected CUDA Driver identity {major}.{minor},{sms}")
PY
    echo "${phase}_DATE_EXIT=$date_exit ${phase}_NVIDIA_SMI_EXIT=$smi_exit ${phase}_APPS_EXIT=$apps_exit ${phase}_DRIVER_EXIT=$driver_exit"
    if [ "$date_exit" -ne 0 ] || [ "$smi_exit" -ne 0 ] || [ "$apps_exit" -ne 0 ] || [ "$driver_exit" -ne 0 ]; then
        echo "${phase}_TELEMETRY_EXIT=70"
        return 70
    fi
    echo "${phase}_TELEMETRY_EXIT=0"
}

export CUDA_HOME=$cuda_root
export CUDA_PATH=$cuda_root
export LD_LIBRARY_PATH=$cuda_root/lib64
export PATH=$cuda_root/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CARGO_TARGET_DIR=$target_dir
export MAMBA_RS_KERNEL_CACHE=$cache_dir
export MAMBA_FIXED_ADA_DIRECT_PAIR=1
export MAMBA_FIXED_VENDOR_EXACT_CC=8.9
export MAMBA_FIXED_ADA_ROWS=bf16,f16
export MAMBA_FIXED_ADA_CELLS=hot_a,hot_b,hot_c,hot_d,hot_e
export MAMBA_FIXED_ADA_BIAS=0,1
export MAMBA_FIXED_VENDOR_PATHS=eager,graph
export MAMBA_FIXED_ADA_WINDOWS=$windows
unset MAMBA_FIXED_VENDOR_TILES MAMBA_FIXED_ADA_VENDOR

echo "RUN_STAGE=direct-pair TOOLKIT=$toolkit WINDOWS=$windows"
echo "COMMAND: env CUDA_HOME=$CUDA_HOME CUDA_PATH=$CUDA_PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH PATH=$PATH CARGO_TARGET_DIR=$CARGO_TARGET_DIR MAMBA_RS_KERNEL_CACHE=$MAMBA_RS_KERNEL_CACHE MAMBA_FIXED_ADA_DIRECT_PAIR=1 MAMBA_FIXED_VENDOR_EXACT_CC=8.9 MAMBA_FIXED_ADA_ROWS=bf16,f16 MAMBA_FIXED_ADA_CELLS=hot_a,hot_b,hot_c,hot_d,hot_e MAMBA_FIXED_ADA_BIAS=0,1 MAMBA_FIXED_VENDOR_PATHS=eager,graph MAMBA_FIXED_ADA_WINDOWS=$windows env -u MAMBA_FIXED_VENDOR_TILES $binary fixed_ada_half_forced_direct_pair --exact --ignored --nocapture --test-threads=1"
printf '%s  %s\n' "$source_sha" "$source_root/tests/gemm_bi_fixed_performance.rs" | sha256sum -c - || exit 66
printf '%s  %s\n' "$binary_sha" "$binary" | sha256sum -c - || exit 66
printf '%s  %s\n' "$control_sha" "$control" | sha256sum -c - || exit 66

telemetry PRE || exit $?

echo BENCHMARK_START
start_seconds=$SECONDS
"$binary" fixed_ada_half_forced_direct_pair --exact --ignored --nocapture --test-threads=1
test_exit=$?
elapsed_seconds=$((SECONDS - start_seconds))
echo BENCHMARK_END

post_exit=0
telemetry POST || post_exit=$?
final_date_exit=0
date --iso-8601=ns || final_date_exit=$?
echo "ELAPSED_SECONDS=$elapsed_seconds TEST_EXIT=$test_exit POST_EXIT=$post_exit FINAL_DATE_EXIT=$final_date_exit"
if [ "$test_exit" -ne 0 ]; then
    exit "$test_exit"
fi
if [ "$post_exit" -ne 0 ]; then
    exit "$post_exit"
fi
exit "$final_date_exit"
