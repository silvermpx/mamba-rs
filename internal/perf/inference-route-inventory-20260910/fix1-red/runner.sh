#!/usr/bin/env bash
set -euo pipefail
source_dir=${1:?immutable source}
evidence_dir=${2:?new evidence directory}
test_name=${3:?exact host test name}
expected_exit=${4:?expected exit code}
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
cargo test --locked --release --features cuda,hf --lib --no-run > "$evidence_dir/lib-build.log" 2>&1
lib_binary=$(sed -n 's/^  Executable .* (\(.*\))$/\1/p' "$evidence_dir/lib-build.log")
test -x "$lib_binary"
sha256sum "$lib_binary" > "$evidence_dir/lib-binary.sha256"
"$lib_binary" --list > "$evidence_dir/lib-tests.list"
grep -Fx "$test_name: test" "$evidence_dir/lib-tests.list"
set +e
"$lib_binary" "$test_name" --exact --nocapture --test-threads=1 > "$evidence_dir/test.log" 2>&1
test_exit=$?
set -e
printf '%s\n' "$test_exit" > "$evidence_dir/test-exit.txt"
test "$test_exit" -eq "$expected_exit"
if test "$expected_exit" -eq 0; then
  grep -F 'test result: ok. 1 passed; 0 failed; 0 ignored;' "$evidence_dir/test.log"
else
  test "$expected_exit" -eq 101
  grep -F 'test result: FAILED. 0 passed; 1 failed; 0 ignored;' "$evidence_dir/test.log"
fi
sha256sum -c "$evidence_dir/source.sha256" > "$evidence_dir/source-after.log"
date -u +%FT%TZ > "$evidence_dir/completed.txt"
