#!/usr/bin/env bash
set -euo pipefail
source_dir=${1:?frozen source directory}
evidence_dir=${2:?new evidence directory}
expectation=${3:?red or green}
test "$expectation" = red || test "$expectation" = green
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
cargo test --locked --release --features cuda --lib --no-run > "$evidence_dir/build.log" 2>&1
test_binary=$(sed -n 's/^  Executable .* (\(.*\))$/\1/p' "$evidence_dir/build.log")
test -x "$test_binary"
sha256sum "$test_binary" > "$evidence_dir/binary.sha256"
"$test_binary" --list > "$evidence_dir/tests.list"
test_name=mamba_ssm::gpu::gemm_bi_triad::qualification::tests::qualification_policy_guard_preserves_mode_setup_error
grep -Fx "$test_name: test" "$evidence_dir/tests.list"
set +e
"$test_binary" "$test_name" --exact --nocapture --test-threads=1 > "$evidence_dir/test.log" 2>&1
test_result=$?
set -e
printf '%s\n' "$test_result" > "$evidence_dir/test-exit.txt"
if test "$expectation" = red; then
  test "$test_result" -ne 0
  grep -F 'test result: FAILED. 0 passed; 1 failed;' "$evidence_dir/test.log"
else
  test "$test_result" -eq 0
  grep -F 'test result: ok. 1 passed;' "$evidence_dir/test.log"
  "$test_binary" qualification_policy_guard --nocapture --test-threads=1 > "$evidence_dir/lease-tests.log" 2>&1
  grep -E 'test result: ok\. [1-9][0-9]* passed;' "$evidence_dir/lease-tests.log"
  cargo check --locked --release --features cuda --all-targets > "$evidence_dir/all-targets-check.log" 2>&1
fi
sha256sum -c "$evidence_dir/source.sha256" > "$evidence_dir/source-after.log"
date -u +%FT%TZ > "$evidence_dir/completed.txt"
