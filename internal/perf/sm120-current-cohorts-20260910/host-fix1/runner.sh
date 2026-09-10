#!/usr/bin/env bash
set -euo pipefail
source_dir=${1:?frozen source}
evidence_dir=${2:?new evidence directory}
toolkit=${3:-13.0}
test ! -e "$evidence_dir"
mkdir -p "$evidence_dir"
cd "$source_dir"
export CUDA_HOME=/usr/local/cuda-$toolkit CUDA_PATH=/usr/local/cuda-$toolkit
export PATH="$CUDA_HOME/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export CARGO_TARGET_DIR=/root/target-ada-finalist-aldmatrix-cuda${toolkit/./}-20260910
cp "$0" "$evidence_dir/runner.sh"
sha256sum src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs tests/gemm_bi_tf32_cohort_binding.rs > "$evidence_dir/source.sha256"
date -u +%FT%TZ > "$evidence_dir/start.txt"
cargo test --locked --release --features cuda --lib --no-run > "$evidence_dir/build.log" 2>&1
cargo test --locked --release --features cuda --lib sm120_tf32_ -- --list > "$evidence_dir/tests.list" 2>&1
grep -F 'sm120_tf32_live_cohort_manifest_is_exact_and_unique: test' "$evidence_dir/tests.list"
cargo test --locked --release --features cuda --lib sm120_tf32_ -- --nocapture --test-threads=1 > "$evidence_dir/host-tests.log" 2>&1
grep -F 'test result: ok.' "$evidence_dir/host-tests.log"
date -u +%FT%TZ > "$evidence_dir/completed.txt"
