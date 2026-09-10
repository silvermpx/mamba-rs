#!/usr/bin/env bash
set -euo pipefail
source_dir=${1:?immutable source}
evidence_dir=${2:?new evidence directory}
test ! -e "$evidence_dir"
mkdir -p "$evidence_dir"
trap 'result=$?; printf "%s\n" "$result" > "$evidence_dir/runner-exit.txt"' EXIT
cd "$source_dir"
export CUDA_HOME=/usr/local/cuda-13.2 CUDA_PATH=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export CARGO_TARGET_DIR=/root/target-ada-finalist-aldmatrix-cuda132-20260910
unset RUSTFLAGS RUSTDOCFLAGS NVIDIA_TF32_OVERRIDE MAMBA_RS_GEMM_MODE MAMBA_RS_BATCH_INVARIANT MAMBA_RS_BI_TENSOR_CORES MAMBA_RS_FAST_GEMM MAMBA_RS_BI_GEMM_FAMILY MAMBA_RS_BI_F32_POLICY MAMBA_RS_BI_HALF_POLICY MAMBA_RS_ARCH_RUNG
date -u +%FT%TZ > "$evidence_dir/start.txt"
cp "$0" "$evidence_dir/runner.sh"
find src kernels tests -type f -print0 | sort -z | xargs -0 sha256sum > "$evidence_dir/source.sha256"
sha256sum Cargo.toml Cargo.lock > "$evidence_dir/build-inputs.sha256"
nvcc --version > "$evidence_dir/toolkit.txt"
cargo test --locked --release --features cuda --lib --no-run > "$evidence_dir/lib-build.log" 2>&1
lib_binary=$(sed -n 's/^  Executable .* (\(.*\))$/\1/p' "$evidence_dir/lib-build.log")
test -x "$lib_binary"
sha256sum "$lib_binary" > "$evidence_dir/lib-binary.sha256"
"$lib_binary" --list > "$evidence_dir/lib-tests.list"
for test_filter in gemm_bi_inference::identity::tests inference_appends_tags_without_reencoding_any_existing_contract cold_architecture_probe_is_rejected_before_recording_or_capture; do
  grep -F "$test_filter" "$evidence_dir/lib-tests.list"
  "$lib_binary" "$test_filter" --nocapture --test-threads=1 > "$evidence_dir/$test_filter.log" 2>&1
  grep -E 'test result: ok\. [1-9][0-9]* passed; 0 failed; 0 ignored;' "$evidence_dir/$test_filter.log"
done
sha256sum -c "$evidence_dir/source.sha256" > "$evidence_dir/source-after.log"
date -u +%FT%TZ > "$evidence_dir/completed.txt"
