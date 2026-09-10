#!/usr/bin/env bash
set -euo pipefail
source_dir=${1:?frozen source}
evidence_dir=${2:?new evidence directory}
test ! -e "$evidence_dir"
mkdir -p "$evidence_dir"
trap 'result=$?; printf "%s\n" "$result" > "$evidence_dir/runner-exit.txt"' EXIT
cd "$source_dir"
export CUDA_HOME=/usr/local/cuda-13.2 CUDA_PATH=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export CARGO_TARGET_DIR=/root/target-ada-finalist-aldmatrix-cuda132-20260910
unset RUSTFLAGS RUSTDOCFLAGS
date -u +%FT%TZ > "$evidence_dir/start.txt"
cp "$0" "$evidence_dir/runner.sh"
find src kernels tests -type f -print0 | sort -z | xargs -0 sha256sum > "$evidence_dir/source.sha256"
sha256sum Cargo.toml Cargo.lock > "$evidence_dir/build-inputs.sha256"
cargo test --locked --release --features cuda,hf --test gemm_bi_tf32_contract --no-run > "$evidence_dir/build.log" 2>&1
test_binary=$(sed -n 's/^  Executable tests\/gemm_bi_tf32_contract.rs (\(.*\))$/\1/p' "$evidence_dir/build.log")
test -x "$test_binary"
sha256sum "$test_binary" > "$evidence_dir/binary.sha256"
failed=0
for test_name in physical_trace_provenance_has_no_crate_visible_mint_or_src_bypass physical_904_census_keeps_test_authorities_gated_private_and_exact physical_owner_rejects_private_method_authorities; do
  "$test_binary" --list | grep -Fx "$test_name: test"
  set +e
  "$test_binary" "$test_name" --exact --nocapture --test-threads=1 > "$evidence_dir/$test_name.log" 2>&1
  test_result=$?
  set -e
  printf '%s\n' "$test_result" > "$evidence_dir/$test_name.exit"
  if test "$test_result" -eq 0; then
    grep -F 'test result: ok. 1 passed; 0 failed; 0 ignored;' "$evidence_dir/$test_name.log"
  else
    failed=$((failed + 1))
    tail -25 "$evidence_dir/$test_name.log"
  fi
done
sha256sum -c "$evidence_dir/source.sha256" > "$evidence_dir/source-after.log"
sha256sum -c "$evidence_dir/build-inputs.sha256" > "$evidence_dir/build-inputs-after.log"
date -u +%FT%TZ > "$evidence_dir/completed.txt"
printf '%s\n' "$failed" > "$evidence_dir/failed-count.txt"
test "$failed" -eq 0
