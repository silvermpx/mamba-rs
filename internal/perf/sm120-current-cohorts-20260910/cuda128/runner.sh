#!/usr/bin/env bash
set -euo pipefail
source_dir=${1:?frozen source}
toolkit=${2:?toolkit}
evidence_dir=${3:?new evidence directory}
label=${toolkit/./}
test ! -e "$evidence_dir"
mkdir -p "$evidence_dir"
cd "$source_dir"
export CUDA_HOME=/usr/local/cuda-$toolkit CUDA_PATH=/usr/local/cuda-$toolkit
export PATH="$CUDA_HOME/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export CUDA_VISIBLE_DEVICES=GPU-a10ad830-a2cf-054d-e0fa-54add30a2cf8
export CARGO_TARGET_DIR=/root/target-sm120-final-sweep-cuda${label}
export MAMBA_RS_KERNEL_CACHE=/root/sm120-release-matrix-cuda${label}-20260910/kernel-cache
export CUDA_CACHE_DISABLE=0 CUDA_CACHE_MAXSIZE=1073741824
export CUDA_CACHE_PATH=/root/sm120-qualified-jit-cuda${label}-20260910
mkdir -p "$MAMBA_RS_KERNEL_CACHE" "$CUDA_CACHE_PATH"
chmod 700 "$MAMBA_RS_KERNEL_CACHE"
unset NVIDIA_TF32_OVERRIDE
nvidia-smi --query-gpu=uuid,name,driver_version,memory.total --format=csv > "$evidence_dir/device.csv"
nvcc --version > "$evidence_dir/nvcc-version.txt"
find src kernels tests -type f -print0 | sort -z | xargs -0 sha256sum > "$evidence_dir/assembly-source.sha256"
cp "$0" "$evidence_dir/runner.sh"
sha256sum "$0" > "$evidence_dir/runner.sha256"
target=gemm_bi_tf32_cohort_binding
test_name=tf32_cohort_binds_on_this_board
cargo test --locked --release --features cuda --test "$target" --no-run > "$evidence_dir/build.log" 2>&1
cargo test --locked --release --features cuda --test "$target" -- --list > "$evidence_dir/tests.list" 2>&1
grep -Fx "$test_name: test" "$evidence_dir/tests.list"
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
cargo test --locked --release --features cuda --test "$target" "$test_name" -- --exact --ignored --nocapture --test-threads=1 > "$evidence_dir/post-auto.log" 2>&1
result=$?
set -e
printf '%s\n' "$result" > "$evidence_dir/exit.txt"
date -u +%FT%TZ > "$evidence_dir/end.txt"
test "$result" = 0
grep -F 'test result: ok. 1 passed;' "$evidence_dir/post-auto.log"
date -u +%FT%TZ > "$evidence_dir/completed.txt"
