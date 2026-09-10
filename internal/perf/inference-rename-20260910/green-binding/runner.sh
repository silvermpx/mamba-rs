#!/usr/bin/env bash
set -euo pipefail
source_dir=${1:?frozen source directory}
evidence_dir=${2:?new evidence directory}
phase=${3:-all}
case "$phase" in all|gpu|remaining|binding) ;; *) exit 2 ;; esac
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
unset NVIDIA_TF32_OVERRIDE MAMBA_RS_BATCH_INVARIANT MAMBA_RS_BI_TENSOR_CORES MAMBA_RS_FAST_GEMM MAMBA_RS_BI_GEMM_FAMILY MAMBA_RS_BI_F32_POLICY MAMBA_RS_BI_HALF_POLICY MAMBA_RS_ARCH_RUNG
test "$(stat -c '%a:%u' "$MAMBA_RS_KERNEL_CACHE")" = '700:0'
date -u +%FT%TZ > "$evidence_dir/start.txt"
printf '%s\n' "$phase" > "$evidence_dir/phase.txt"
cp "$0" "$evidence_dir/runner.sh"
find src kernels tests -type f -print0 | sort -z | xargs -0 sha256sum > "$evidence_dir/source.sha256"
find internal/perf/ada-f32-nt-copyplan-siblings-batch-a-20260909/evidence -type f -print0 | sort -z | xargs -0 sha256sum > "$evidence_dir/fixture.sha256"
sha256sum Cargo.toml Cargo.lock > "$evidence_dir/build-inputs.sha256"
nvcc --version > "$evidence_dir/toolkit.txt"
nvidia-smi --query-gpu=uuid,name,driver_version,utilization.gpu,memory.used,memory.free --format=csv > "$evidence_dir/device.csv"
if [[ "$phase" = all ]]; then
cargo check --locked --release --features cuda --all-targets > "$evidence_dir/all-targets-check.log" 2>&1
cargo test --locked --release --features cuda --lib --no-run > "$evidence_dir/lib-build.log" 2>&1
lib_binary=$(sed -n 's/^  Executable .* (\(.*\))$/\1/p' "$evidence_dir/lib-build.log")
test -x "$lib_binary"
sha256sum "$lib_binary" > "$evidence_dir/lib-binary.sha256"
"$lib_binary" --list > "$evidence_dir/lib-tests.list"
for name in \
  mamba_ssm::gpu::gemm_bi_triad::modules::tests::inference_composed_source_identity_is_frozen \
  mamba_ssm::gpu::gemm_bi_triad::modules::tests::module_sources_have_exact_deterministic_boundaries \
  mamba_ssm::gpu::context::tests::bi_gemm_family_environment_accepts_only_semantic_family_names; do
  grep -Fx "$name: test" "$evidence_dir/lib-tests.list"
  log="$evidence_dir/${name##*::}.log"
  "$lib_binary" "$name" --exact --nocapture --test-threads=1 > "$log" 2>&1
  grep -F 'test result: ok. 1 passed;' "$log"
done
"$lib_binary" 'mamba_ssm::gpu::gemm_bi_inference::' --test-threads=1 > "$evidence_dir/inference-selectors.log" 2>&1
grep -E 'test result: ok\. [1-9][0-9]* passed;' "$evidence_dir/inference-selectors.log"
"$lib_binary" sm120_tf32_ --test-threads=1 > "$evidence_dir/sm120-cohorts.log" 2>&1
grep -E 'test result: ok\. [1-9][0-9]* passed;' "$evidence_dir/sm120-cohorts.log"
cargo test --locked --release --features cuda --test kernel_identity -- --test-threads=1 > "$evidence_dir/kernel-identity.log" 2>&1
grep -E 'test result: ok\. [1-9][0-9]* passed;' "$evidence_dir/kernel-identity.log"
fi
run_gpu_case() {
  local target=$1 name=$2
  cargo test --locked --release --features cuda --test "$target" --no-run > "$evidence_dir/$target-build.log" 2>&1
  local binary
  binary=$(sed -n 's/^  Executable .* (\(.*\))$/\1/p' "$evidence_dir/$target-build.log")
  test -x "$binary"
  sha256sum "$binary" >> "$evidence_dir/gpu-binaries.sha256"
  "$binary" --list > "$evidence_dir/$target-tests.list"
  grep -Fx "$name: test" "$evidence_dir/$target-tests.list"
  nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits > "$evidence_dir/$name-preflight.csv"
  awk -F, 'NF != 2 || $1+0 > 1 || $2+0 < 2048 {exit 1}' "$evidence_dir/$name-preflight.csv"
  "$binary" "$name" --exact --include-ignored --nocapture --test-threads=1 > "$evidence_dir/$name.log" 2>&1
  grep -F 'test result: ok. 1 passed;' "$evidence_dir/$name.log"
}
if [[ "$phase" != remaining ]]; then
  run_gpu_case gemm_bi_tf32_cohort_binding tf32_cohort_binds_on_this_board
fi
if [[ "$phase" != binding ]]; then
run_gpu_case gemm_bi_sm89_tf32_joint_selector_qualification live::sm89_tf32_joint_post_admission_auto_uses_the_toolkit_winner_map
run_gpu_case gemm_bi_fixed_correctness fixed_sm89_rna_wide_actual_auto_hot_a_route_and_graph
run_gpu_case gemm_bi_fixed_sm89_pipeline fixed_sm89_half_pipeline_auto_hot_cell_prefix_view_graph_bits
run_gpu_case gemm_bi_fixed_sm89_exact_n64 fixed_sm89_exact_n64_auto_prefix_view_graph_bits
fi
sha256sum -c "$evidence_dir/source.sha256" > "$evidence_dir/source-after.log"
sha256sum -c "$evidence_dir/fixture.sha256" > "$evidence_dir/fixture-after.log"
date -u +%FT%TZ > "$evidence_dir/completed.txt"
