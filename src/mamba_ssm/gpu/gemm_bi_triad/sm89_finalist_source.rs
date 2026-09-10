use std::collections::BTreeSet;

pub(super) const SM89_FINALIST_SYMBOL: &str =
    "gemm_bi_nt_sm89_mma_tf32_compact8_v1_m128n64_bk32_s2";

const SM80_SOURCE: &str = include_str!("../../../../kernels/gemm_bi_triad/sm80.cu");
const COMPACT_HELPER: &str = include_str!("../../../../kernels/gemm_bi_triad/sm89_nt_compact.cuh");
const PREAMBLES: [&str; 5] = [
    include_str!("../../../../kernels/_typed_prelude.cuh"),
    include_str!("../../../../kernels/gemm_bi_triad/contract.cuh"),
    include_str!("../../../../kernels/gemm_bi_triad/common.cuh"),
    include_str!("../../../../kernels/gemm_bi_triad/epilogue.cuh"),
    include_str!("../../../../kernels/gemm_bi_triad/mma16.cuh"),
];

const ORIGINAL_SYMBOL: &str = "gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s2";

const ASYNC_MAINLOOP_MARKER: &str = r#"template <SgbTf32Op Op, int BM, int BN, int Stages,
          int MAtoms, int NAtoms, bool NarrowA, bool NarrowB>
__device__ __forceinline__ void gemm_bi_tf32_async_mainloop("#;

const WIDE_BRANCH: &str = r#"    if (wide_a && wide_b) {
        gemm_bi_tf32_async_mainloop<
            Op, BM, BN, Stages, MAtoms, NAtoms, false, false>(
            storage, problem, tile_count, thread_plan, accumulators);
    } else if (gemm_bi_tf32_can_stage_async_4(a, b, params)) {"#;

const A_LOAD: &str = r#"#pragma unroll
        for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
            int row = warp_m + m_atom * 16 + group;
            a_fragments[m_atom][0] =
                gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row, k8 + thread));
            a_fragments[m_atom][1] =
                gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row + 8, k8 + thread));
            a_fragments[m_atom][2] =
                gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row, k8 + thread + 4));
            a_fragments[m_atom][3] =
                gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row + 8, k8 + thread + 4));
        }
"#;

const A_LDMATRIX_LOAD: &str = r#"#pragma unroll
        for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
            if constexpr (Op == SgbTf32Nt && BM == 128 && BN == 64 && Stages == 2) {
                int lane = (int)threadIdx.x & 31;
                int row = warp_m + m_atom * 16 + (lane & 15);
                int reduction = k8 + ((lane >> 4) << 2);
                unsigned address = (unsigned)__cvta_generic_to_shared(
                    &gemm_bi_tf32_a_slot<Op>(storage, stage, row, reduction));
                unsigned raw0, raw1, raw2, raw3;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(raw0), "=r"(raw1), "=r"(raw2), "=r"(raw3)
                    : "r"(address));
                a_fragments[m_atom][0] = gemm_bi_tf32_rna(__uint_as_float(raw0));
                a_fragments[m_atom][1] = gemm_bi_tf32_rna(__uint_as_float(raw1));
                a_fragments[m_atom][2] = gemm_bi_tf32_rna(__uint_as_float(raw2));
                a_fragments[m_atom][3] = gemm_bi_tf32_rna(__uint_as_float(raw3));
            } else {
                int row = warp_m + m_atom * 16 + group;
                a_fragments[m_atom][0] =
                    gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row, k8 + thread));
                a_fragments[m_atom][1] =
                    gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row + 8, k8 + thread));
                a_fragments[m_atom][2] =
                    gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row, k8 + thread + 4));
                a_fragments[m_atom][3] =
                    gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row + 8, k8 + thread + 4));
            }
        }
"#;

const SLICED_BRANCH: &str = r#"    if (wide_a && wide_b) {
        if constexpr (Op == SgbTf32Nt && BM == 128 && BN == 64 && Stages == 2) {
            bool use_stage_sliced =
                (params.m == 2048 && params.k == 1536 && params.n == 768
                    && params.lda == 768 && params.ldb == 768 && params.ldc == 1536)
                || (params.m == 4096 && params.k == 3072 && params.n == 1536
                    && params.lda == 1536 && params.ldb == 1536 && params.ldc == 3072);
            if (use_stage_sliced) {
                gemm_bi_tf32_nt_compact_sliced_mainloop(
                    storage, problem, tile_count, thread_plan, accumulators);
            } else {
                gemm_bi_tf32_async_mainloop<
                    Op, BM, BN, Stages, MAtoms, NAtoms, false, false>(
                    storage, problem, tile_count, thread_plan, accumulators);
            }
        } else {
            gemm_bi_tf32_async_mainloop<
                Op, BM, BN, Stages, MAtoms, NAtoms, false, false>(
                storage, problem, tile_count, thread_plan, accumulators);
        }
    } else if (gemm_bi_tf32_can_stage_async_4(a, b, params)) {"#;

const SLICED_HELPERS: &str = r#"__device__ __forceinline__ void gemm_bi_tf32_nt_compact_stage_slice(
    SgbTf32Storage<SgbTf32Nt, 128, 64, 2>* storage, int stage,
    const SgbTf32Problem& problem, int reduction_base, int issue) {
    {
        int linear = (int)threadIdx.x + issue * 256;
        int row = linear >> 3;
        int reduction = (linear & 7) * 4;
        int global_row = problem.tile_row + row;
        int global_reduction = reduction_base + reduction;
        int valid = global_row < problem.params.m
            ? problem.params.n - global_reduction : 0;
        valid = valid < 0 ? 0 : (valid > 4 ? 4 : valid);
        int bytes = valid * 4;
        long long valid_offset =
            (long long)global_row * problem.params.lda + global_reduction;
        const float* source = gemm_bi_cp_async_source(
            problem.a, bytes == 0 ? 0 : valid_offset, bytes);
        unsigned destination = (unsigned)__cvta_generic_to_shared(
            &gemm_bi_tf32_a_slot<SgbTf32Nt>(storage, stage, row, reduction));
        gemm_bi_tf32_cp_async_zfill<false, 128>(
            destination, source, bytes);
    }
    if (issue < 2) {
        int linear = (int)threadIdx.x + issue * 256;
        int column = linear >> 3;
        int reduction = (linear & 7) * 4;
        int global_column = problem.tile_column + column;
        int global_reduction = reduction_base + reduction;
        int valid = global_column < problem.params.k
            ? problem.params.n - global_reduction : 0;
        valid = valid < 0 ? 0 : (valid > 4 ? 4 : valid);
        int bytes = valid * 4;
        long long valid_offset =
            (long long)global_column * problem.params.ldb + global_reduction;
        const float* source = gemm_bi_cp_async_source(
            problem.b, bytes == 0 ? 0 : valid_offset, bytes);
        unsigned destination = (unsigned)__cvta_generic_to_shared(
            &gemm_bi_tf32_b_slot<SgbTf32Nt>(storage, stage, reduction, column));
        gemm_bi_tf32_cp_async_zfill<false, 128>(
            destination, source, bytes);
    }
}

__device__ __forceinline__ void gemm_bi_tf32_nt_compact_stage_async(
    SgbTf32Storage<SgbTf32Nt, 128, 64, 2>* storage, int stage,
    const SgbTf32Problem& problem, int reduction_base) {
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        gemm_bi_tf32_nt_compact_stage_slice(
            storage, stage, problem, reduction_base, issue);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void gemm_bi_tf32_nt_compact_compute_sliced(
    SgbTf32Storage<SgbTf32Nt, 128, 64, 2>* storage, int read_stage,
    int warp_m, int warp_n, int group, int thread,
    bool has_next, int write_stage, const SgbTf32Problem& problem,
    int next_reduction, float (&accumulators)[2][4][4]) {
    const int k_offsets[4] = {0, 8, 16, 24};
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        if (has_next) {
            gemm_bi_tf32_nt_compact_stage_slice(
                storage, write_stage, problem, next_reduction, issue);
        }
        int k8 = k_offsets[issue];
        unsigned a_fragments[2][4];
        unsigned b_fragments[4][2];
#pragma unroll
        for (int m_atom = 0; m_atom < 2; ++m_atom) {
            int lane = (int)threadIdx.x & 31;
            int row = warp_m + m_atom * 16 + (lane & 15);
            int reduction = k8 + ((lane >> 4) << 2);
            unsigned address = (unsigned)__cvta_generic_to_shared(
                &gemm_bi_tf32_a_slot<SgbTf32Nt>(
                    storage, read_stage, row, reduction));
            unsigned raw0, raw1, raw2, raw3;
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(raw0), "=r"(raw1), "=r"(raw2), "=r"(raw3)
                : "r"(address));
            a_fragments[m_atom][0] = gemm_bi_tf32_rna(__uint_as_float(raw0));
            a_fragments[m_atom][1] = gemm_bi_tf32_rna(__uint_as_float(raw1));
            a_fragments[m_atom][2] = gemm_bi_tf32_rna(__uint_as_float(raw2));
            a_fragments[m_atom][3] = gemm_bi_tf32_rna(__uint_as_float(raw3));
        }
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
            int column = warp_n + n_atom * 8 + group;
            b_fragments[n_atom][0] = gemm_bi_tf32_rna(
                gemm_bi_tf32_b_slot<SgbTf32Nt>(
                    storage, read_stage, k8 + thread, column));
            b_fragments[n_atom][1] = gemm_bi_tf32_rna(
                gemm_bi_tf32_b_slot<SgbTf32Nt>(
                    storage, read_stage, k8 + thread + 4, column));
        }
#pragma unroll
        for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < 4; ++n_atom) {
                gemm_bi_tf32_mma_m16n8k8(
                    accumulators[m_atom][n_atom],
                    a_fragments[m_atom], b_fragments[n_atom]);
            }
        }
    }
}

__device__ __forceinline__ void gemm_bi_tf32_nt_compact_sliced_mainloop(
    SgbTf32Storage<SgbTf32Nt, 128, 64, 2>* storage,
    const SgbTf32Problem& problem, unsigned tile_count,
    const SgbTf32ThreadPlan& thread_plan,
    float (&accumulators)[2][4][4]) {
    if (tile_count != 0) {
        gemm_bi_tf32_nt_compact_stage_async(storage, 0, problem, 0);
    } else {
        asm volatile("cp.async.commit_group;\n" ::);
    }
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        unsigned next = tile + 1;
        bool has_next = next < tile_count;
        gemm_bi_tf32_nt_compact_compute_sliced(
            storage, static_cast<int>(tile & 1U),
            thread_plan.warp_m, thread_plan.warp_n,
            thread_plan.group, thread_plan.thread,
            has_next, static_cast<int>(next & 1U), problem,
            static_cast<int>(next * 32U), accumulators);
        asm volatile("cp.async.commit_group;\n" ::);
        __syncthreads();
    }
}

"#;

#[derive(Clone, Copy)]
struct Transformation {
    label: &'static str,
    from: &'static str,
    to: &'static str,
}

const TRANSFORMATIONS: [Transformation; 9] = [
    Transformation {
        label: "compact finalist storage",
        from: concat!(
            "    static constexpr int bk32 = 32;\n",
            "    static constexpr int ARows = Op == SgbTf32Tn ? bk32 : BM;\n",
            "    static constexpr int AStride = Op == SgbTf32Tn ? BM + 8 : 36;\n",
            "    static constexpr int BRows = Op == SgbTf32Nt ? BN : bk32;\n",
            "    static constexpr int BStride = Op == SgbTf32Nt ? 36\n",
            "        : (BN == 64 ? 72 : (BN == 32 ? 40 : 24));"
        ),
        to: concat!(
            "    static constexpr int bk32 = 32;\n",
            "    static constexpr bool compact_eight_warp_s2 =\n",
            "        Op == SgbTf32Nt && BM == 128 && BN == 64 && Stages == 2;\n",
            "    static constexpr int ARows = Op == SgbTf32Tn ? bk32 : BM;\n",
            "    static constexpr int AStride = Op == SgbTf32Tn ? BM + 8\n",
            "        : (compact_eight_warp_s2 ? 32 : 36);\n",
            "    static constexpr int BRows = Op == SgbTf32Nt ? BN : bk32;\n",
            "    static constexpr int BStride = Op == SgbTf32Nt\n",
            "        ? (compact_eight_warp_s2 ? 32 : 36)\n",
            "        : (BN == 64 ? 72 : (BN == 32 ? 40 : 24));"
        ),
    },
    Transformation {
        label: "compact finalist A slot",
        from: concat!(
            "    if constexpr (Op == SgbTf32Tn) {\n",
            "        return storage->a[stage][reduction][row];\n",
            "    }\n",
            "    return storage->a[stage][row][reduction];"
        ),
        to: concat!(
            "    if constexpr (Op == SgbTf32Tn) {\n",
            "        return storage->a[stage][reduction][row];\n",
            "    }\n",
            "    if constexpr (Op == SgbTf32Nt && BM == 128 && BN == 64 && Stages == 2) {\n",
            "        return storage->a[stage][row][gemm_bi_nt_compact8_xor_k(row, reduction)];\n",
            "    }\n",
            "    return storage->a[stage][row][reduction];"
        ),
    },
    Transformation {
        label: "compact finalist B slot",
        from: concat!(
            "    if constexpr (Op == SgbTf32Nt) {\n",
            "        return storage->b[stage][column][reduction];\n",
            "    }\n",
            "    return storage->b[stage][reduction][column];"
        ),
        to: concat!(
            "    if constexpr (Op == SgbTf32Nt) {\n",
            "        if constexpr (BM == 128 && BN == 64 && Stages == 2) {\n",
            "            return storage->b[stage][column][gemm_bi_nt_compact8_xor_k(column, reduction)];\n",
            "        }\n",
            "        return storage->b[stage][column][reduction];\n",
            "    }\n",
            "    return storage->b[stage][reduction][column];"
        ),
    },
    Transformation {
        label: "compact finalist storage extent",
        from: "== 55296, \"NT M128N64 s2 storage\"",
        to: "== 49152, \"NT compact-eight-warp M128N64 s2 storage\"",
    },
    Transformation {
        label: "compact finalist accumulator ownership",
        from: concat!(
            "__device__ __forceinline__ void gemm_bi_tf32_kernel(\n",
            "    float* output, const float* a, const float* b, const float* bias,\n",
            "    Sm80Tf32KernelParams params) {\n",
            "    constexpr int MAtoms = BM == 128 ? 4 : (BM == 64 ? 2 : 1);"
        ),
        to: concat!(
            "__device__ __forceinline__ void gemm_bi_tf32_kernel(\n",
            "    float* output, const float* a, const float* b, const float* bias,\n",
            "    Sm80Tf32KernelParams params) {\n",
            "    constexpr bool compact_eight_warp_s2 =\n",
            "        Op == SgbTf32Nt && BM == 128 && BN == 64 && Stages == 2;\n",
            "    constexpr int MAtoms = compact_eight_warp_s2 ? 2\n",
            "        : (BM == 128 ? 4 : (BM == 64 ? 2 : 1));"
        ),
    },
    Transformation {
        label: "compact finalist compute and row ownership",
        from: concat!(
            "    bool compute = BM != 128 || warp < 4;\n",
            "    int warp_m = BM == 128 ? (warp >> 1) * 64\n",
            "        : (BM == 64 ? (warp >> 1) * 32 : 0);"
        ),
        to: concat!(
            "    bool compute = compact_eight_warp_s2 || BM != 128 || warp < 4;\n",
            "    int warp_m = compact_eight_warp_s2 ? (warp >> 1) * 32\n",
            "        : (BM == 128 ? (warp >> 1) * 64\n",
            "        : (BM == 64 ? (warp >> 1) * 32 : 0));"
        ),
    },
    Transformation {
        label: "compact finalist A-only ldmatrix load",
        from: A_LOAD,
        to: A_LDMATRIX_LOAD,
    },
    Transformation {
        label: "compact finalist target export",
        from: concat!(
            "GEMM_BI_TF32_DEFINE_KERNEL(",
            "gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s2, ",
            "SgbTf32Nt, 128, 64, 2, 256, 1)"
        ),
        to: concat!(
            "GEMM_BI_TF32_DEFINE_KERNEL(",
            "gemm_bi_nt_sm89_mma_tf32_compact8_v1_m128n64_bk32_s2, ",
            "SgbTf32Nt, 128, 64, 2, 256, 1)"
        ),
    },
    Transformation {
        label: "compact finalist target signature",
        from: concat!(
            "TF32_ASSERT_KERNEL_SIGNATURE(",
            "gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s2);"
        ),
        to: concat!(
            "TF32_ASSERT_KERNEL_SIGNATURE(",
            "gemm_bi_nt_sm89_mma_tf32_compact8_v1_m128n64_bk32_s2);"
        ),
    },
];

pub(super) fn compose_sm89_finalist_source() -> Result<String, String> {
    compose_from_parts(PREAMBLES, COMPACT_HELPER, SM80_SOURCE)
}

fn transform_sm80_source(source: &str) -> Result<String, String> {
    let mut transformed = source.to_owned();
    for transformation in TRANSFORMATIONS {
        replace_exact(
            &mut transformed,
            transformation.from,
            transformation.to,
            transformation.label,
        )?;
    }
    install_stage_sliced_winners(&mut transformed)?;
    Ok(transformed)
}

fn install_stage_sliced_winners(source: &mut String) -> Result<(), String> {
    replace_exact(
        source,
        ASYNC_MAINLOOP_MARKER,
        &format!("{SLICED_HELPERS}{ASYNC_MAINLOOP_MARKER}"),
        "stage-sliced helper insertion",
    )?;
    replace_exact(
        source,
        WIDE_BRANCH,
        SLICED_BRANCH,
        "stage-sliced runtime gate",
    )
}

fn finalist_body_from(helper: &str, source: &str) -> Result<String, String> {
    reject_quoted_include("kernels/gemm_bi_triad/sm89_nt_compact.cuh", helper)?;
    let transformed = transform_sm80_source(source)?;
    let exports = macro_names(&transformed, "GEMM_BI_TF32_DEFINE_KERNEL")?;
    let assertions = macro_names(&transformed, "TF32_ASSERT_KERNEL_SIGNATURE")?;
    if exports != assertions {
        return Err("SM89 finalist export and signature-assert sets differ".into());
    }
    if !exports.contains(SM89_FINALIST_SYMBOL) || exports.contains(ORIGINAL_SYMBOL) {
        return Err("SM89 finalist target export replacement is incomplete".into());
    }
    Ok(format!("{helper}\n{transformed}"))
}

fn replace_exact(source: &mut String, from: &str, to: &str, label: &str) -> Result<(), String> {
    let count = source.matches(from).count();
    if count != 1 {
        return Err(format!(
            "{label} anchor count changed: expected 1, observed {count}"
        ));
    }
    *source = source.replacen(from, to, 1);
    Ok(())
}

fn macro_names(source: &str, macro_name: &str) -> Result<BTreeSet<String>, String> {
    let prefix = format!("{macro_name}(");
    let mut names = BTreeSet::new();
    for line in source.lines() {
        let Some(arguments) = line.trim_start().strip_prefix(&prefix) else {
            continue;
        };
        let name = arguments
            .split([',', ')'])
            .next()
            .map(str::trim)
            .filter(|name| {
                !name.is_empty()
                    && name
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
            })
            .ok_or_else(|| format!("{macro_name} invocation has an invalid first argument"))?;
        if !names.insert(name.to_owned()) {
            return Err(format!(
                "{macro_name} contains duplicate invocation for {name}"
            ));
        }
    }
    if names.is_empty() {
        return Err(format!("{macro_name} has no concrete invocations"));
    }
    Ok(names)
}

fn reject_quoted_include(logical_name: &str, source: &str) -> Result<(), String> {
    for line in source.lines() {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix('#') else {
            continue;
        };
        let rest = rest.trim_start();
        let Some(target) = rest.strip_prefix("include").map(str::trim_start) else {
            continue;
        };
        if target.starts_with('"') {
            return Err(format!(
                "{logical_name} contains unexpected quoted include {target}"
            ));
        }
    }
    Ok(())
}

fn normalized_part(source: &str) -> String {
    source.lines().collect::<Vec<_>>().join("\n")
}

fn compose_from_parts(preambles: [&str; 5], helper: &str, source: &str) -> Result<String, String> {
    for (logical_name, preamble) in [
        ("kernels/_typed_prelude.cuh", preambles[0]),
        ("kernels/gemm_bi_triad/contract.cuh", preambles[1]),
        ("kernels/gemm_bi_triad/common.cuh", preambles[2]),
        ("kernels/gemm_bi_triad/epilogue.cuh", preambles[3]),
        ("kernels/gemm_bi_triad/mma16.cuh", preambles[4]),
    ] {
        reject_quoted_include(logical_name, preamble)?;
    }
    reject_quoted_include("kernels/gemm_bi_triad/sm89_nt_compact.cuh", helper)?;
    reject_quoted_include("kernels/gemm_bi_triad/sm80.cu", source)?;

    let body = finalist_body_from(helper, source)?;
    Ok(preambles
        .into_iter()
        .map(normalized_part)
        .chain(std::iter::once(normalized_part(&body)))
        .collect::<Vec<_>>()
        .join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "cuda"))]
    const TEST_HELPER_NAME: &str = "gemm_bi_nt_test_compact_xor_k";
    #[cfg(not(feature = "cuda"))]
    const FROZEN_HELPER: &str = include_str!("../../../../tests/gemm_bi_tf32_nt_compact_xor.cu");
    const FROZEN_SLICED_ADAPTER: &str =
        include_str!("../../../../tests/support/triad_tf32_nt_compact_a_ldmatrix_sliced_source.rs");

    #[cfg(not(feature = "cuda"))]
    mod frozen_candidate {
        include!("../../../../tests/gemm_bi_tf32_nt_compact_xor.rs");

        pub(super) fn compact_a_ldmatrix_source() -> Result<String, String> {
            let compact = compact_eight_warp_s2_candidate_source()?;
            triad_tf32_nt_compact_a_ldmatrix_source::candidate_source(&compact)
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn normalized_frozen_candidate() -> String {
        let mut source = frozen_candidate::compact_a_ldmatrix_source()
            .expect("compose frozen compact A-only ldmatrix candidate")
            .replacen(FROZEN_HELPER, COMPACT_HELPER, 1)
            .replace(TEST_HELPER_NAME, "gemm_bi_nt_compact8_xor_k")
            .replace(
                "gemm_bi_nt_test_compact_a_ldmatrix_sm80_mma_tf32_v1_m128n64_bk32_s2",
                SM89_FINALIST_SYMBOL,
            );
        install_stage_sliced_winners(&mut source).expect("install frozen stage-sliced winners");
        source
    }

    #[test]
    fn missing_and_duplicate_anchors_fail_closed() {
        for transformation in TRANSFORMATIONS {
            let missing = SM80_SOURCE.replacen(transformation.from, "", 1);
            let missing_error = transform_sm80_source(&missing).unwrap_err();
            assert!(missing_error.contains(transformation.label));
            assert!(missing_error.contains("observed 0"));

            let duplicate = format!("{SM80_SOURCE}\n{}", transformation.from);
            let duplicate_error = transform_sm80_source(&duplicate).unwrap_err();
            assert!(duplicate_error.contains(transformation.label));
            assert!(duplicate_error.contains("observed 2"));
        }
    }

    #[test]
    fn sliced_helpers_match_the_measured_adapter_and_seams_fail_closed() {
        let measured_helpers = FROZEN_SLICED_ADAPTER
            .split_once("const SLICED_HELPERS: &str = r#\"")
            .expect("measured sliced-helper prefix")
            .1
            .split_once("\"#;\n\npub const fn a_copy_coordinate")
            .expect("measured sliced-helper suffix")
            .0;
        assert_eq!(SLICED_HELPERS, measured_helpers);

        let mut compact = SM80_SOURCE.to_owned();
        for transformation in TRANSFORMATIONS {
            replace_exact(
                &mut compact,
                transformation.from,
                transformation.to,
                transformation.label,
            )
            .unwrap();
        }
        for (anchor, label) in [
            (ASYNC_MAINLOOP_MARKER, "stage-sliced helper insertion"),
            (WIDE_BRANCH, "stage-sliced runtime gate"),
        ] {
            let mut missing = compact.replacen(anchor, "", 1);
            let missing_error = install_stage_sliced_winners(&mut missing).unwrap_err();
            assert!(missing_error.contains(label));
            assert!(missing_error.contains("observed 0"));

            let mut duplicate = format!("{compact}\n{anchor}");
            let duplicate_error = install_stage_sliced_winners(&mut duplicate).unwrap_err();
            assert!(duplicate_error.contains(label));
            assert!(duplicate_error.contains("observed 2"));
        }
    }

    #[test]
    fn reversed_transformation_restores_immutable_sm80_source() {
        let transformed = transform_sm80_source(SM80_SOURCE).unwrap();
        let restored = super::reverse_sm80_transformation_for_test(&transformed).unwrap();
        assert_eq!(restored, SM80_SOURCE);
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn production_body_matches_frozen_candidate_after_allowed_normalization() {
        let production = finalist_body_from(COMPACT_HELPER, SM80_SOURCE).unwrap();
        assert_eq!(production, normalized_frozen_candidate());
    }

    #[test]
    fn exports_and_signature_asserts_are_balanced() {
        let body = finalist_body_from(COMPACT_HELPER, SM80_SOURCE).unwrap();
        let exports = macro_names(&body, "GEMM_BI_TF32_DEFINE_KERNEL").unwrap();
        let assertions = macro_names(&body, "TF32_ASSERT_KERNEL_SIGNATURE").unwrap();
        assert_eq!(exports, assertions);
        assert!(exports.contains(SM89_FINALIST_SYMBOL));
        assert_eq!(exports.len(), 18);
    }

    #[test]
    fn composed_source_uses_exact_five_preambles_and_rejects_new_quoted_include() {
        let composed = compose_sm89_finalist_source().unwrap();
        for marker in [
            "__device__ __forceinline__ float to_f(float v)",
            "Three operand layouts for training:",
            "__device__ __forceinline__ float4 ld_global_L2_128B",
            "__device__ __forceinline__ void gemm_bi_store_pair_rne",
            "gemm_bi_cp_async_source",
        ] {
            assert!(
                composed.contains(marker),
                "missing preamble marker {marker}"
            );
        }
        assert!(composed.contains(SM89_FINALIST_SYMBOL));

        let parts = super::source_parts_for_test();
        let poisoned = format!("{}\n#include \"unexpected.cuh\"", parts[2]);
        let error = super::compose_from_parts_for_test(
            [parts[0], parts[1], &poisoned, parts[3], parts[4]],
            COMPACT_HELPER,
            SM80_SOURCE,
        )
        .unwrap_err();
        assert!(error.contains("unexpected.cuh"));
    }

    #[test]
    fn composed_source_emits_the_same_helper_it_validates() {
        let chosen = format!("// chosen helper seam\n{COMPACT_HELPER}");
        let composed = super::compose_from_parts_for_test(
            super::source_parts_for_test(),
            &chosen,
            SM80_SOURCE,
        )
        .unwrap();
        assert_eq!(composed.matches("// chosen helper seam").count(), 1);
        assert!(composed.contains(&chosen));
    }

    #[test]
    fn finalist_uses_a_only_ldmatrix_for_all_cells_and_slices_only_proven_shapes() {
        let body = finalist_body_from(COMPACT_HELPER, SM80_SOURCE).unwrap();
        let expected_gate = concat!(
            "bool use_stage_sliced =\n",
            "                (params.m == 2048 && params.k == 1536 && params.n == 768\n",
            "                    && params.lda == 768 && params.ldb == 768 && params.ldc == 1536)\n",
            "                || (params.m == 4096 && params.k == 3072 && params.n == 1536\n",
            "                    && params.lda == 1536 && params.ldb == 1536 && params.ldc == 3072);"
        );
        assert!(body.contains(expected_gate));
        assert!(body.contains(A_LDMATRIX_LOAD));
        assert!(!body.contains("params.m == 2048 && params.k == 768 && params.n == 3072"));
        assert!(!body.contains("params.m == 4621 && params.k == 384 && params.n == 1928"));
        assert!(body.contains(
            "static_assert(sizeof(Sm80Tf32KernelParams) == 32, \"TF32 parameter ABI drift\");"
        ));
        assert!(body.contains(concat!(
            "using SgbTf32KernelSignature = void (*)(\n",
            "    float*, const float*, const float*, const float*, Sm80Tf32KernelParams);"
        )));
        assert_eq!(
            body.matches("gemm_bi_tf32_nt_compact_sliced_mainloop(")
                .count(),
            2
        );
        assert_eq!(body.matches(SM89_FINALIST_SYMBOL).count(), 2);
        assert!(body.contains(concat!(
            "if (use_stage_sliced) {\n",
            "                gemm_bi_tf32_nt_compact_sliced_mainloop(\n",
            "                    storage, problem, tile_count, thread_plan, accumulators);\n",
            "            } else {\n",
            "                gemm_bi_tf32_async_mainloop<\n",
            "                    Op, BM, BN, Stages, MAtoms, NAtoms, false, false>(\n",
            "                    storage, problem, tile_count, thread_plan, accumulators);\n",
            "            }"
        )));
    }
}

#[cfg(test)]
fn reverse_sm80_transformation_for_test(source: &str) -> Result<String, String> {
    let mut restored = source.to_owned();
    replace_exact(
        &mut restored,
        SLICED_BRANCH,
        WIDE_BRANCH,
        "stage-sliced runtime gate",
    )?;
    replace_exact(
        &mut restored,
        &format!("{SLICED_HELPERS}{ASYNC_MAINLOOP_MARKER}"),
        ASYNC_MAINLOOP_MARKER,
        "stage-sliced helper insertion",
    )?;
    for transformation in TRANSFORMATIONS.into_iter().rev() {
        replace_exact(
            &mut restored,
            transformation.to,
            transformation.from,
            transformation.label,
        )?;
    }
    Ok(restored)
}

#[cfg(test)]
fn source_parts_for_test() -> [&'static str; 5] {
    PREAMBLES
}

#[cfg(test)]
fn compose_from_parts_for_test(
    preambles: [&str; 5],
    helper: &str,
    source: &str,
) -> Result<String, String> {
    compose_from_parts(preambles, helper, source)
}
