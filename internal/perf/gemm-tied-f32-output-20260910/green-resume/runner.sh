#!/usr/bin/env bash
set -euo pipefail
source_dir=${1:?unchanged frozen source}
prior=${2:?first GREEN packet}
evidence_dir=${3:?new continuation packet}
test ! -e "$evidence_dir"
mkdir -p "$evidence_dir"
trap 'result=$?; printf "%s\n" "$result" > "$evidence_dir/runner-exit.txt"' EXIT
cd "$source_dir"
export CUDA_HOME=/usr/local/cuda-13.2 CUDA_PATH=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export CARGO_TARGET_DIR=/root/target-ada-finalist-aldmatrix-cuda132-20260910
export CUDA_VISIBLE_DEVICES=GPU-d1edd7be-e88d-aed6-047d-622163306f0e
export MAMBA_RS_KERNEL_CACHE=/root/ada-final-auto-cuda132-20260910/kernel-cache
export CUDA_CACHE_DISABLE=0 CUDA_CACHE_MAXSIZE=1073741824
export CUDA_CACHE_PATH=/root/ada-final-auto-cuda132-20260910/jit-cache
unset RUSTFLAGS RUSTDOCFLAGS NVIDIA_TF32_OVERRIDE MAMBA_RS_GEMM_MODE MAMBA_RS_BATCH_INVARIANT MAMBA_RS_BI_TENSOR_CORES MAMBA_RS_FAST_GEMM MAMBA_RS_BI_GEMM_FAMILY MAMBA_RS_BI_F32_POLICY MAMBA_RS_BI_HALF_POLICY MAMBA_RS_ARCH_RUNG
date -u +%FT%TZ > "$evidence_dir/start.txt"
cp "$0" "$evidence_dir/runner.sh"
sha256sum -c "$prior/source.sha256" > "$evidence_dir/source-before.log"
sha256sum -c "$prior/lib-binary.sha256" > "$evidence_dir/binary-before.log"
lib_binary=$(sed -n 's/^  Executable .* (\(.*\))$/\1/p' "$prior/lib-build.log")
test -x "$lib_binary"
tail -n +2 "$prior/required-lib-tests.txt" > "$evidence_dir/required-lib-tests.txt"
while IFS= read -r test_name; do
  grep -Fx "$test_name: test" "$prior/lib-tests.list"
  idle=0
  for attempt in 1 2 3; do
    nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits > "$evidence_dir/$test_name.preflight-$attempt.csv"
    if awk -F, 'NF != 2 || $1+0 > 1 || $2+0 < 2048 {exit 1}' "$evidence_dir/$test_name.preflight-$attempt.csv"; then
      idle=1
      break
    fi
    sleep 2
  done
  test "$idle" = 1
  "$lib_binary" "$test_name" --exact --include-ignored --nocapture --test-threads=1 > "$evidence_dir/$test_name.log" 2>&1
  grep -F 'test result: ok. 1 passed; 0 failed; 0 ignored;' "$evidence_dir/$test_name.log"
done < "$evidence_dir/required-lib-tests.txt"
RUSTDOCFLAGS='-D rustdoc::broken_intra_doc_links' cargo doc --locked --release --features cuda,hf --no-deps --lib > "$evidence_dir/rustdoc.log" 2>&1
sha256sum -c "$prior/source.sha256" > "$evidence_dir/source-after.log"
sha256sum -c "$prior/lib-binary.sha256" > "$evidence_dir/binary-after.log"
date -u +%FT%TZ > "$evidence_dir/completed.txt"
