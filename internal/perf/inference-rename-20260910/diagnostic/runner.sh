#!/usr/bin/env bash
set -euo pipefail
source_dir=${1:?frozen source directory}
evidence_dir=${2:?new evidence directory}
test ! -e "$evidence_dir"
mkdir -p "$evidence_dir"
trap 'result=$?; printf "%s\n" "$result" > "$evidence_dir/runner-exit.txt"' EXIT
cd "$source_dir"
export CUDA_HOME=/usr/local/cuda-13.2 CUDA_PATH=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export CARGO_TARGET_DIR=/root/target-ada-finalist-aldmatrix-cuda132-20260910
date -u +%FT%TZ > "$evidence_dir/start.txt"
cp "$0" "$evidence_dir/runner.sh"
find src kernels tests -type f -print0 | sort -z | xargs -0 sha256sum > "$evidence_dir/source.sha256"
sha256sum Cargo.toml Cargo.lock > "$evidence_dir/build-inputs.sha256"
nvcc --version > "$evidence_dir/toolkit.txt"
cargo test --locked --release --features cuda --lib --no-run > "$evidence_dir/build.log" 2>&1
binary=$(sed -n 's/^  Executable .* (\(.*\))$/\1/p' "$evidence_dir/build.log")
test -x "$binary"
sha256sum "$binary" > "$evidence_dir/binary.sha256"
"$binary" --list > "$evidence_dir/tests.list"
identity=mamba_ssm::gpu::gemm_bi_triad::modules::tests::inference_composed_source_identity_is_frozen
parser=mamba_ssm::gpu::context::tests::bi_gemm_family_environment_accepts_only_semantic_family_names
grep -Fx "$identity: test" "$evidence_dir/tests.list"
grep -Fx "$parser: test" "$evidence_dir/tests.list"
"$binary" "$identity" --exact --nocapture --test-threads=1 > "$evidence_dir/identity.log" 2>&1
grep -F 'test result: ok. 1 passed;' "$evidence_dir/identity.log"
set +e
"$binary" "$parser" --exact --nocapture --test-threads=1 > "$evidence_dir/parser-red.log" 2>&1
parser_result=$?
set -e
printf '%s\n' "$parser_result" > "$evidence_dir/parser-exit.txt"
test "$parser_result" = 101
grep -F 'test result: FAILED. 0 passed; 1 failed;' "$evidence_dir/parser-red.log"
grep -F 'inference' "$evidence_dir/parser-red.log"
sha256sum -c "$evidence_dir/source.sha256" > "$evidence_dir/source-after.log"
date -u +%FT%TZ > "$evidence_dir/completed.txt"
