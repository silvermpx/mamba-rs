#[path = "triad_tf32_nt_compact_a_ldmatrix_sliced_source.rs"]
mod composed_parent;

pub const SYMBOL: &str =
    "gemm_bi_nt_test_a_ldmatrix_prism_full_domain_sm80_mma_tf32_v1_m128n64_bk32_s2";
pub const RETAINED_SYMBOL: &str = composed_parent::RETAINED_SYMBOL;
pub const BLOCK_THREADS: u32 = 256;
pub const DYNAMIC_SHARED_BYTES: u32 = 49_152;
pub const REGISTER_CAP: i32 = 128;
pub const REQUIRED_OCCUPANCY: u32 = 2;
pub const TARGET: (usize, usize, usize) = (4_621, 384, 1_928);
pub const TARGET_GRID: (u32, u32, u32) = (6, 37, 1);
pub const FULL_M_CTAS: usize = 36 * 6;

const ASYNC_MAINLOOP_MARKER: &str = r#"template <SgbTf32Op Op, int BM, int BN, int Stages,
          int MAtoms, int NAtoms, bool NarrowA, bool NarrowB>
__device__ __forceinline__ void gemm_bi_tf32_async_mainloop("#;

const ORIGINAL_GEOMETRY: &str = r#"    int columns = gemm_bi_tf32_columns<Op>(params);
    int column_tiles = (columns + BN - 1) / BN;
    SgbTf32Problem problem = {
        output, a, b, bias, params,
        (int)blockIdx.x / column_tiles * BM,
        (int)blockIdx.x % column_tiles * BN,
    };"#;

const PRISM_GEOMETRY: &str = r#"    int columns = gemm_bi_tf32_columns<Op>(params);
    int column_tiles = (columns + BN - 1) / BN;
    constexpr bool prism_specialization =
        Op == SgbTf32Nt && BM == 128 && BN == 64 && Stages == 2;
    bool canonical_prism = prism_specialization
        && params.m == 4621 && params.k == 384 && params.n == 1928
        && params.lda == 1928 && params.ldb == 1928 && params.ldc == 384
        && params.alpha == 1.0f && params.beta == 0.0f && bias == nullptr;
    SgbTf32Problem problem = {
        output, a, b, bias, params,
        canonical_prism ? (int)blockIdx.y * BM
                        : (int)blockIdx.x / column_tiles * BM,
        canonical_prism ? (int)blockIdx.x * BN
                        : (int)blockIdx.x % column_tiles * BN,
    };
    bool canonical_prism_full_output = canonical_prism
        && problem.tile_row + BM <= params.m
        && problem.tile_column + BN <= columns;"#;

const ORIGINAL_WIDE_BRANCH: &str = r#"    if (wide_a && wide_b) {
        gemm_bi_tf32_async_mainloop<
            Op, BM, BN, Stages, MAtoms, NAtoms, false, false>(
            storage, problem, tile_count, thread_plan, accumulators);
    } else if (gemm_bi_tf32_can_stage_async_4(a, b, params)) {"#;

const PRISM_WIDE_BRANCH: &str = r#"    if (wide_a && wide_b) {
        if constexpr (prism_specialization) {
            if (canonical_prism) {
                gemm_bi_tf32_nt_prism_full_domain_mainloop(
                    storage, problem, tile_count, thread_plan,
                    canonical_prism_full_output, accumulators);
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

const ORIGINAL_STORE: &str = r#"                    gemm_bi_tf32_store<Op>(output, row, column,
                        accumulators[m_atom][n_atom][element], bias, params);"#;

const PRISM_STORE: &str = r#"                    if (canonical_prism_full_output) {
                        float value = gemm_bi_tf32_epilogue<Op>(
                            accumulators[m_atom][n_atom][element], 0.0f,
                            bias, column, params);
                        output[(long long)row * params.ldc + column] = value;
                    } else {
                        gemm_bi_tf32_store<Op>(output, row, column,
                            accumulators[m_atom][n_atom][element], bias, params);
                    }"#;

const FULL_DOMAIN_HELPERS: &str = r#"__device__ __forceinline__ void gemm_bi_tf32_nt_prism_copy_full(
    unsigned shared_dst, const float* global_src) {
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
        :: "r"(shared_dst), "l"(global_src));
}

__device__ __forceinline__ void gemm_bi_tf32_nt_prism_stage_full(
    SgbTf32Storage<SgbTf32Nt, 128, 64, 2>* storage, int stage,
    const SgbTf32Problem& problem, int reduction_base) {
    for (int linear = (int)threadIdx.x; linear < 128 * 8; linear += 256) {
        int row = linear >> 3;
        int reduction = (linear & 7) * 4;
        const float* source = problem.a
            + (long long)(problem.tile_row + row) * problem.params.lda
            + reduction_base + reduction;
        unsigned destination = (unsigned)__cvta_generic_to_shared(
            &gemm_bi_tf32_a_slot<SgbTf32Nt>(storage, stage, row, reduction));
        gemm_bi_tf32_nt_prism_copy_full(destination, source);
    }
    for (int linear = (int)threadIdx.x; linear < 64 * 8; linear += 256) {
        int column = linear >> 3;
        int reduction = (linear & 7) * 4;
        const float* source = problem.b
            + (long long)(problem.tile_column + column) * problem.params.ldb
            + reduction_base + reduction;
        unsigned destination = (unsigned)__cvta_generic_to_shared(
            &gemm_bi_tf32_b_slot<SgbTf32Nt>(storage, stage, reduction, column));
        gemm_bi_tf32_nt_prism_copy_full(destination, source);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void gemm_bi_tf32_nt_prism_full_domain_mainloop(
    SgbTf32Storage<SgbTf32Nt, 128, 64, 2>* storage,
    const SgbTf32Problem& problem, unsigned tile_count,
    const SgbTf32ThreadPlan& thread_plan, bool full_output,
    float (&accumulators)[2][4][4]) {
    if (tile_count != 0) {
        if (full_output && tile_count > 1) {
            gemm_bi_tf32_nt_prism_stage_full(storage, 0, problem, 0);
        } else {
            gemm_bi_tf32_stage_async<SgbTf32Nt, 128, 64, 2, false, false>(
                storage, 0, problem, 0);
        }
    } else {
        asm volatile("cp.async.commit_group;\n" ::);
    }
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        unsigned next = tile + 1;
        if (next < tile_count) {
            if (full_output && next + 1 < tile_count) {
                gemm_bi_tf32_nt_prism_stage_full(
                    storage, static_cast<int>(next & 1U), problem,
                    static_cast<int>(next * 32U));
            } else {
                gemm_bi_tf32_stage_async<
                    SgbTf32Nt, 128, 64, 2, false, false>(
                    storage, static_cast<int>(next & 1U), problem,
                    static_cast<int>(next * 32U));
            }
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
        if (thread_plan.compute) {
            gemm_bi_tf32_compute_stage<SgbTf32Nt, 128, 64, 2, 2, 4>(
                storage, static_cast<int>(tile & 1U),
                thread_plan.warp_m, thread_plan.warp_n,
                thread_plan.group, thread_plan.thread, accumulators);
        }
        __syncthreads();
    }
}

"#;

pub const fn full_m_cta(tile_y: usize) -> bool {
    tile_y < 36
}

pub const fn full_k_stage(tile: usize) -> bool {
    tile < 60
}

pub fn retained_source(production: &str, compact_layout: &str) -> Result<String, String> {
    composed_parent::retained_source(production, compact_layout)
}

pub fn candidate_source(production: &str, compact_layout: &str) -> Result<String, String> {
    let mut source = retained_source(production, compact_layout)?;
    replace_exact(
        &mut source,
        ASYNC_MAINLOOP_MARKER,
        &format!("{FULL_DOMAIN_HELPERS}{ASYNC_MAINLOOP_MARKER}"),
        "helper insertion",
    )?;
    replace_exact(
        &mut source,
        ORIGINAL_GEOMETRY,
        PRISM_GEOMETRY,
        "2D geometry",
    )?;
    replace_exact(
        &mut source,
        ORIGINAL_WIDE_BRANCH,
        PRISM_WIDE_BRANCH,
        "wide branch",
    )?;
    replace_exact(
        &mut source,
        ORIGINAL_STORE,
        PRISM_STORE,
        "full output store",
    )?;
    replace_exact_twice(&mut source, RETAINED_SYMBOL, SYMBOL, "candidate export")?;
    Ok(source)
}

pub fn restore_retained_source(candidate: &str) -> Result<String, String> {
    let mut source = candidate.to_owned();
    replace_exact_twice(&mut source, SYMBOL, RETAINED_SYMBOL, "restore export")?;
    replace_exact(&mut source, PRISM_STORE, ORIGINAL_STORE, "restore store")?;
    replace_exact(
        &mut source,
        PRISM_WIDE_BRANCH,
        ORIGINAL_WIDE_BRANCH,
        "restore wide branch",
    )?;
    replace_exact(
        &mut source,
        PRISM_GEOMETRY,
        ORIGINAL_GEOMETRY,
        "restore geometry",
    )?;
    replace_exact(
        &mut source,
        &format!("{FULL_DOMAIN_HELPERS}{ASYNC_MAINLOOP_MARKER}"),
        ASYNC_MAINLOOP_MARKER,
        "remove helpers",
    )?;
    Ok(source)
}

fn replace_exact(source: &mut String, from: &str, to: &str, label: &str) -> Result<(), String> {
    let count = source.matches(from).count();
    if count != 1 {
        return Err(format!(
            "TF32 NT canonical Prism full-domain {label} seam changed: expected 1, observed {count}"
        ));
    }
    *source = source.replacen(from, to, 1);
    Ok(())
}

fn replace_exact_twice(
    source: &mut String,
    from: &str,
    to: &str,
    label: &str,
) -> Result<(), String> {
    let count = source.matches(from).count();
    if count != 2 {
        return Err(format!(
            "TF32 NT canonical Prism full-domain {label} seam changed: expected 2, observed {count}"
        ));
    }
    *source = source.replace(from, to);
    Ok(())
}
