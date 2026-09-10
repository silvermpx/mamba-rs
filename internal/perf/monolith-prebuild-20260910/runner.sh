#!/usr/bin/env bash
set -euo pipefail
source_dir=${1:?immutable main source directory}
evidence_dir=${2:?new evidence directory}
test ! -e "$evidence_dir"
mkdir -p "$evidence_dir"
trap 'result=$?; printf "%s\n" "$result" > "$evidence_dir/runner-exit.txt"' EXIT
cd "$source_dir"
export CUDA_HOME=/usr/local/cuda-13.2 CUDA_PATH=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export CARGO_TARGET_DIR=/root/target-monolith-main-prebuild-cuda132-20260910
unset RUSTDOCFLAGS RUSTFLAGS NVIDIA_TF32_OVERRIDE MAMBA_RS_GEMM_MODE MAMBA_RS_BATCH_INVARIANT MAMBA_RS_BI_TENSOR_CORES MAMBA_RS_FAST_GEMM MAMBA_RS_BI_GEMM_FAMILY MAMBA_RS_BI_F32_POLICY MAMBA_RS_BI_HALF_POLICY MAMBA_RS_ARCH_RUNG
date -u +%FT%TZ > "$evidence_dir/start.txt"
printf '%s\n' d8f2efbeaecc04a53cec9890097f2913cd3f09e6 > "$evidence_dir/source-commit.txt"
cp "$0" "$evidence_dir/runner.sh"
find src kernels tests xtask .cargo -type f -print0 | sort -z | xargs -0 sha256sum > "$evidence_dir/source.sha256"
sha256sum Cargo.toml Cargo.lock > "$evidence_dir/build-inputs.sha256"
nvcc --version > "$evidence_dir/toolkit.txt"
rustc --version > "$evidence_dir/rustc.txt"
nvidia-smi --query-gpu=uuid,name,driver_version,utilization.gpu,memory.used,memory.free --format=csv > "$evidence_dir/device.csv"
cargo test --locked --release --features cuda --test m1_gpu_benchmark --no-run > "$evidence_dir/build.log" 2>&1
binary=$(sed -n 's/^  Executable .* (\(.*\))$/\1/p' "$evidence_dir/build.log")
test -x "$binary"
sha256sum "$binary" > "$evidence_dir/binary.sha256"
"$binary" --list > "$evidence_dir/tests.list"
grep -Fx 'm1_gpu_benchmark: test' "$evidence_dir/tests.list"
sha256sum -c "$evidence_dir/source.sha256" > "$evidence_dir/source-after.log"
date -u +%FT%TZ > "$evidence_dir/completed.txt"
