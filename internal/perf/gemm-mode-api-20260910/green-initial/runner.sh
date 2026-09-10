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
export CUDA_VISIBLE_DEVICES=GPU-d1edd7be-e88d-aed6-047d-622163306f0e
export MAMBA_RS_KERNEL_CACHE=/root/ada-final-auto-cuda132-20260910/kernel-cache
export CUDA_CACHE_DISABLE=0 CUDA_CACHE_MAXSIZE=1073741824
export CUDA_CACHE_PATH=/root/ada-final-auto-cuda132-20260910/jit-cache
unset RUSTFLAGS RUSTDOCFLAGS NVIDIA_TF32_OVERRIDE MAMBA_RS_GEMM_MODE MAMBA_RS_BATCH_INVARIANT MAMBA_RS_BI_TENSOR_CORES MAMBA_RS_FAST_GEMM MAMBA_RS_BI_GEMM_FAMILY MAMBA_RS_BI_F32_POLICY MAMBA_RS_BI_HALF_POLICY MAMBA_RS_ARCH_RUNG
test "$(stat -c '%a:%u' "$MAMBA_RS_KERNEL_CACHE")" = '700:0'
date -u +%FT%TZ > "$evidence_dir/start.txt"
cp "$0" "$evidence_dir/runner.sh"
find src kernels tests -type f -print0 | sort -z | xargs -0 sha256sum > "$evidence_dir/source.sha256"
find internal/perf/ada-f32-nt-copyplan-siblings-batch-a-20260909/evidence -type f -print0 | sort -z | xargs -0 sha256sum > "$evidence_dir/fixture.sha256"
sha256sum Cargo.toml Cargo.lock > "$evidence_dir/build-inputs.sha256"
nvcc --version > "$evidence_dir/toolkit.txt"
nvidia-smi --query-gpu=uuid,name,driver_version,utilization.gpu,memory.used,memory.free --format=csv > "$evidence_dir/device.csv"
cargo check --locked --release --features cuda --all-targets > "$evidence_dir/all-targets-check.log" 2>&1
cargo test --locked --release --features cuda --test gemm_mode_api --no-run > "$evidence_dir/api-build.log" 2>&1
api_binary=$(sed -n 's/^  Executable .* (\(.*\))$/\1/p' "$evidence_dir/api-build.log")
test -x "$api_binary"
sha256sum "$api_binary" > "$evidence_dir/api-binary.sha256"
"$api_binary" --list > "$evidence_dir/api-tests.list"
for name in gemm_mode_names_and_default gpu_module_reexports_the_canonical_gemm_mode gpu_context_constructors_use_explicit_modes_and_deterministic_defaults gpu_context_supports_all_nine_canonical_mode_transitions legacy_mode_adapters_preserve_the_documented_call_order vendor_mode_round_trip_preserves_custom_deterministic_policy; do
  grep -Fx "$name: test" "$evidence_dir/api-tests.list"
done
nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits > "$evidence_dir/api-preflight.csv"
awk -F, 'NF != 2 || $1+0 > 1 || $2+0 < 2048 {exit 1}' "$evidence_dir/api-preflight.csv"
"$api_binary" --include-ignored --nocapture --test-threads=1 > "$evidence_dir/api-tests.log" 2>&1
grep -E 'test result: ok\. [1-9][0-9]* passed;' "$evidence_dir/api-tests.log"
cargo test --locked --release --features cuda --test gemm_mode_live --no-run > "$evidence_dir/live-build.log" 2>&1
live_binary=$(sed -n 's/^  Executable .* (\(.*\))$/\1/p' "$evidence_dir/live-build.log")
test -x "$live_binary"
sha256sum "$live_binary" > "$evidence_dir/live-binary.sha256"
"$live_binary" --list > "$evidence_dir/live-tests.list"
for name in gpu_mode_change_rejects_capture_but_allows_same_mode_noop gpu_mode_change_is_rejected_during_route_recording; do
  grep -Fx "$name: test" "$evidence_dir/live-tests.list"
done
"$live_binary" --include-ignored --nocapture --test-threads=1 > "$evidence_dir/live-tests.log" 2>&1
grep -E 'test result: ok\. 2 passed;' "$evidence_dir/live-tests.log"
cargo test --locked --release --features cuda --lib --no-run > "$evidence_dir/lib-build.log" 2>&1
lib_binary=$(sed -n 's/^  Executable .* (\(.*\))$/\1/p' "$evidence_dir/lib-build.log")
test -x "$lib_binary"
sha256sum "$lib_binary" > "$evidence_dir/lib-binary.sha256"
"$lib_binary" --list > "$evidence_dir/lib-tests.list"
"$lib_binary" gemm_mode --nocapture --test-threads=1 > "$evidence_dir/mode-unit-tests.log" 2>&1
grep -E 'test result: ok\. [1-9][0-9]* passed;' "$evidence_dir/mode-unit-tests.log"
for name in mamba_ssm::gpu::context::tests::deterministic_environment_defaults_and_validates_custom_policy mamba_ssm::gpu::blas::physical_graph_tests::context_aware_vendor_compute_maps_all_dtypes; do
  grep -Fx "$name: test" "$evidence_dir/lib-tests.list"
  "$lib_binary" "$name" --exact --include-ignored --nocapture --test-threads=1 > "$evidence_dir/${name##*::}.log" 2>&1
  grep -E 'test result: ok\. 1 passed;' "$evidence_dir/${name##*::}.log"
done
cargo test --locked --release --features cuda --test kernel_identity -- --test-threads=1 > "$evidence_dir/kernel-identity.log" 2>&1
grep -E 'test result: ok\. [1-9][0-9]* passed;' "$evidence_dir/kernel-identity.log"
RUSTDOCFLAGS='-D rustdoc::broken_intra_doc_links' cargo doc --locked --release --features cuda --no-deps --lib > "$evidence_dir/rustdoc.log" 2>&1
cargo test --locked --release --features cuda --doc > "$evidence_dir/doctests.log" 2>&1
sha256sum -c "$evidence_dir/source.sha256" > "$evidence_dir/source-after.log"
sha256sum -c "$evidence_dir/fixture.sha256" > "$evidence_dir/fixture-after.log"
date -u +%FT%TZ > "$evidence_dir/completed.txt"
