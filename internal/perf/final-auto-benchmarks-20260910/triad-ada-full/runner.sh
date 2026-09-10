#!/usr/bin/env bash
set -euo pipefail
source_dir=${1:?frozen source}
toolkit=${2:?toolkit}
evidence_dir=${3:?new evidence directory}
board=${4:?ada or sm120}
phase=${5:?smoke or full}
production_sha=${6:?production Git SHA}
label=${toolkit/./}
case "$board" in
    sm120)
        gpu=GPU-a10ad830-a2cf-054d-e0fa-54add30a2cf8
        target=/root/target-sm120-final-sweep-cuda${label}
        cache=/root/sm120-release-matrix-cuda${label}-20260910/kernel-cache
        jit=/root/sm120-qualified-jit-cuda${label}-20260910
        ;;
    ada)
        gpu=GPU-d1edd7be-e88d-aed6-047d-622163306f0e
        target=/root/target-ada-finalist-aldmatrix-cuda${label}-20260910
        cache=/root/ada-final-auto-cuda${label}-20260910/kernel-cache
        jit=/root/ada-final-auto-cuda${label}-20260910/jit-cache
        ;;
    *) exit 2 ;;
esac
test ! -e "$evidence_dir"
mkdir -p "$evidence_dir" "$cache" "$jit"
chmod 700 "$cache"
cd "$source_dir"
export CUDA_HOME=/usr/local/cuda-$toolkit CUDA_PATH=/usr/local/cuda-$toolkit
export PATH="$CUDA_HOME/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export CARGO_TARGET_DIR="$target" MAMBA_RS_KERNEL_CACHE="$cache"
export CUDA_CACHE_DISABLE=0 CUDA_CACHE_PATH="$jit" CUDA_CACHE_MAXSIZE=1073741824
export CUDA_VISIBLE_DEVICES="$gpu"
export GEMM_BI_FINAL_AUTO_GIT_SHA="$production_sha"
export GEMM_BI_QUAL_VARIANT="final-auto-${board}-${phase}-cuda${label}"
export GEMM_BI_FINAL_AUTO_JSONL="$evidence_dir/triad.jsonl"
unset NVIDIA_TF32_OVERRIDE
case "$phase" in
    smoke)
        export GEMM_BI_QUAL_WINDOWS=3
        : "${GEMM_BI_QUAL_CELL_IDS:?strict smoke cells must be supplied}"
        ;;
    full)
        export GEMM_BI_QUAL_WINDOWS=21
        unset GEMM_BI_QUAL_CELL_IDS
        ;;
    *) exit 2 ;;
esac
nvidia-smi --query-gpu=uuid,name,driver_version,memory.total --format=csv > "$evidence_dir/device.csv"
nvcc --version > "$evidence_dir/nvcc-version.txt"
find src kernels tests -type f -print0 | sort -z | xargs -0 sha256sum > "$evidence_dir/assembly-source.sha256"
sha256sum "$0" > "$evidence_dir/runner.sha256"
cp "$0" "$evidence_dir/runner.sh"
target_name=gemm_bi_performance_matrix
test_name=gemm_bi_production_auto_paired_cublas_release_matrix
cargo test --locked --release --features cuda --test "$target_name" --no-run > "$evidence_dir/build.log" 2>&1
cargo test --locked --release --features cuda --test "$target_name" -- --list > "$evidence_dir/tests.list" 2>&1
grep -Fx "$test_name: test" "$evidence_dir/tests.list"
cargo test --locked --release --features cuda --test "$target_name" final_auto_ -- --nocapture --test-threads=1 > "$evidence_dir/host-tests.log" 2>&1
grep -F 'test result: ok. 5 passed;' "$evidence_dir/host-tests.log"
binary=$(sed -n 's/^  Executable .* (\(.*\))$/\1/p' "$evidence_dir/build.log")
test -x "$binary"
sha256sum "$binary" > "$evidence_dir/test-binary.sha256"
sleep 2
for sample in 1 2 3 4 5; do
    telemetry=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.free --format=csv,noheader,nounits)
    printf '%s\n' "$telemetry" >> "$evidence_dir/preflight.log"
    awk -F, 'NF != 3 || $1+0 > 1 || $2+0 > 1 || $3+0 < 2048 {exit 1}' <<< "$telemetry"
    if [[ "$sample" != 5 ]]; then sleep 1; fi
done
date -u +%FT%TZ > "$evidence_dir/start.txt"
set +e
cargo test --locked --release --features cuda --test "$target_name" "$test_name" -- --exact --ignored --nocapture --test-threads=1 > "$evidence_dir/triad.log" 2>&1
result=$?
set -e
date -u +%FT%TZ > "$evidence_dir/end.txt"
printf '%s\n' "$result" > "$evidence_dir/exit.txt"
test "$result" = 0
grep -F 'test result: ok. 1 passed;' "$evidence_dir/triad.log"
if [[ "$phase" == full ]]; then test "$(wc -l < "$evidence_dir/triad.jsonl")" -eq 325; fi
date -u +%FT%TZ > "$evidence_dir/completed.txt"
