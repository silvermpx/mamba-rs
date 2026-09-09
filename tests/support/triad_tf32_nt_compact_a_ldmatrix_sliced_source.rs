#[path = "triad_tf32_nt_compact_a_ldmatrix_source.rs"]
mod retained_a;

pub const SYMBOL: &str =
    "gemm_bi_nt_test_compact_a_ldmatrix_sliced_sm80_mma_tf32_v1_m128n64_bk32_s2";
pub const RETAINED_SYMBOL: &str = retained_a::SYMBOL;
pub const BLOCK_THREADS: u32 = 256;
pub const DYNAMIC_SHARED_BYTES: u32 = 49_152;
pub const REGISTER_CAP: i32 = 128;
pub const REQUIRED_OCCUPANCY: u32 = 2;
pub const TARGET: (usize, usize, usize) = (2_048, 1_536, 768);
pub const TARGET_GRID: u32 = 384;

const COMPACT_PARENT_SYMBOL: &str =
    "gemm_bi_nt_test_compact_eight_warp_sm80_mma_tf32_v1_m128n64_bk32_s2";
const PRODUCTION_SYMBOL: &str = "gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s2";

const ASYNC_MAINLOOP_MARKER: &str = r#"template <SgbTf32Op Op, int BM, int BN, int Stages,
          int MAtoms, int NAtoms, bool NarrowA, bool NarrowB>
__device__ __forceinline__ void gemm_bi_tf32_async_mainloop("#;

const WIDE_BRANCH: &str = r#"    if (wide_a && wide_b) {
        gemm_bi_tf32_async_mainloop<
            Op, BM, BN, Stages, MAtoms, NAtoms, false, false>(
            storage, problem, tile_count, thread_plan, accumulators);
    } else if (gemm_bi_tf32_can_stage_async_4(a, b, params)) {"#;

const SLICED_BRANCH: &str = r#"    if (wide_a && wide_b) {
        if constexpr (Op == SgbTf32Nt && BM == 128 && BN == 64 && Stages == 2) {
            gemm_bi_tf32_nt_compact_sliced_mainloop(
                storage, problem, tile_count, thread_plan, accumulators);
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

pub const fn a_copy_coordinate(thread: usize, slice: usize) -> (usize, usize) {
    let linear = thread + slice * 256;
    (linear >> 3, (linear & 7) * 4)
}

pub const fn b_copy_coordinate(thread: usize, slice: usize) -> Option<(usize, usize)> {
    if slice >= 2 {
        return None;
    }
    let linear = thread + slice * 256;
    Some((linear >> 3, (linear & 7) * 4))
}

pub fn retained_source(production: &str, compact_layout: &str) -> Result<String, String> {
    let compact = compact_parent_source(production, compact_layout)?;
    retained_a::candidate_source(&compact)
}

pub fn candidate_source(production: &str, compact_layout: &str) -> Result<String, String> {
    let mut source = retained_source(production, compact_layout)?;
    replace_exact(
        &mut source,
        ASYNC_MAINLOOP_MARKER,
        &format!("{SLICED_HELPERS}{ASYNC_MAINLOOP_MARKER}"),
        1,
        "helper insertion",
    )?;
    replace_exact(
        &mut source,
        WIDE_BRANCH,
        SLICED_BRANCH,
        1,
        "wide mainloop branch",
    )?;
    replace_exact(&mut source, RETAINED_SYMBOL, SYMBOL, 2, "candidate export")?;
    Ok(source)
}

pub fn restore_retained_source(candidate: &str) -> Result<String, String> {
    let mut source = candidate.to_owned();
    replace_exact(&mut source, SYMBOL, RETAINED_SYMBOL, 2, "restore export")?;
    replace_exact(
        &mut source,
        SLICED_BRANCH,
        WIDE_BRANCH,
        1,
        "restore mainloop branch",
    )?;
    replace_exact(
        &mut source,
        &format!("{SLICED_HELPERS}{ASYNC_MAINLOOP_MARKER}"),
        ASYNC_MAINLOOP_MARKER,
        1,
        "restore helper insertion",
    )?;
    Ok(source)
}

fn compact_parent_source(production: &str, compact_layout: &str) -> Result<String, String> {
    let mut source = production.to_owned();
    replace_exact(
        &mut source,
        concat!(
            "    static constexpr int bk32 = 32;\n",
            "    static constexpr int ARows = Op == SgbTf32Tn ? bk32 : BM;\n",
            "    static constexpr int AStride = Op == SgbTf32Tn ? BM + 8 : 36;\n",
            "    static constexpr int BRows = Op == SgbTf32Nt ? BN : bk32;\n",
            "    static constexpr int BStride = Op == SgbTf32Nt ? 36\n",
            "        : (BN == 64 ? 72 : (BN == 32 ? 40 : 24));"
        ),
        concat!(
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
        1,
        "compact storage",
    )?;
    replace_exact(
        &mut source,
        concat!(
            "    if constexpr (Op == SgbTf32Tn) {\n",
            "        return storage->a[stage][reduction][row];\n",
            "    }\n",
            "    return storage->a[stage][row][reduction];"
        ),
        concat!(
            "    if constexpr (Op == SgbTf32Tn) {\n",
            "        return storage->a[stage][reduction][row];\n",
            "    }\n",
            "    if constexpr (Op == SgbTf32Nt && BM == 128 && BN == 64 && Stages == 2) {\n",
            "        return storage->a[stage][row][gemm_bi_nt_test_compact_xor_k(row, reduction)];\n",
            "    }\n",
            "    return storage->a[stage][row][reduction];"
        ),
        1,
        "compact A slot",
    )?;
    replace_exact(
        &mut source,
        concat!(
            "    if constexpr (Op == SgbTf32Nt) {\n",
            "        return storage->b[stage][column][reduction];\n",
            "    }\n",
            "    return storage->b[stage][reduction][column];"
        ),
        concat!(
            "    if constexpr (Op == SgbTf32Nt) {\n",
            "        if constexpr (BM == 128 && BN == 64 && Stages == 2) {\n",
            "            return storage->b[stage][column][gemm_bi_nt_test_compact_xor_k(column, reduction)];\n",
            "        }\n",
            "        return storage->b[stage][column][reduction];\n",
            "    }\n",
            "    return storage->b[stage][reduction][column];"
        ),
        1,
        "compact B slot",
    )?;
    replace_exact(
        &mut source,
        "== 55296, \"NT M128N64 s2 storage\"",
        "== 49152, \"NT compact-eight-warp M128N64 s2 storage\"",
        1,
        "compact extent",
    )?;
    replace_exact(
        &mut source,
        concat!(
            "__device__ __forceinline__ void gemm_bi_tf32_kernel(\n",
            "    float* output, const float* a, const float* b, const float* bias,\n",
            "    Sm80Tf32KernelParams params) {\n",
            "    constexpr int MAtoms = BM == 128 ? 4 : (BM == 64 ? 2 : 1);"
        ),
        concat!(
            "__device__ __forceinline__ void gemm_bi_tf32_kernel(\n",
            "    float* output, const float* a, const float* b, const float* bias,\n",
            "    Sm80Tf32KernelParams params) {\n",
            "    constexpr bool compact_eight_warp_s2 =\n",
            "        Op == SgbTf32Nt && BM == 128 && BN == 64 && Stages == 2;\n",
            "    constexpr int MAtoms = compact_eight_warp_s2 ? 2\n",
            "        : (BM == 128 ? 4 : (BM == 64 ? 2 : 1));"
        ),
        1,
        "compact accumulator ownership",
    )?;
    replace_exact(
        &mut source,
        concat!(
            "    bool compute = BM != 128 || warp < 4;\n",
            "    int warp_m = BM == 128 ? (warp >> 1) * 64\n",
            "        : (BM == 64 ? (warp >> 1) * 32 : 0);"
        ),
        concat!(
            "    bool compute = compact_eight_warp_s2 || BM != 128 || warp < 4;\n",
            "    int warp_m = compact_eight_warp_s2 ? (warp >> 1) * 32\n",
            "        : (BM == 128 ? (warp >> 1) * 64\n",
            "        : (BM == 64 ? (warp >> 1) * 32 : 0));"
        ),
        1,
        "compact compute ownership",
    )?;
    replace_exact(
        &mut source,
        &format!("GEMM_BI_TF32_DEFINE_KERNEL({PRODUCTION_SYMBOL}, SgbTf32Nt, 128, 64, 2, 256, 1)"),
        &format!(
            "GEMM_BI_TF32_DEFINE_KERNEL({COMPACT_PARENT_SYMBOL}, SgbTf32Nt, 128, 64, 2, 256, 1)"
        ),
        1,
        "compact export",
    )?;
    replace_exact(
        &mut source,
        &format!("TF32_ASSERT_KERNEL_SIGNATURE({PRODUCTION_SYMBOL});"),
        &format!("TF32_ASSERT_KERNEL_SIGNATURE({COMPACT_PARENT_SYMBOL});"),
        1,
        "compact signature",
    )?;
    Ok(format!("{compact_layout}\n{source}"))
}

fn replace_exact(
    source: &mut String,
    from: &str,
    to: &str,
    expected: usize,
    label: &str,
) -> Result<(), String> {
    let count = source.matches(from).count();
    if count != expected {
        return Err(format!(
            "TF32 NT sliced {label} seam changed: expected {expected}, observed {count}"
        ));
    }
    *source = source.replace(from, to);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_triad/sm80.cu");
    const COMPACT_LAYOUT: &str = include_str!("../gemm_bi_tf32_nt_compact_xor.cu");

    #[test]
    fn four_slices_cover_the_compact_a_and_b_stage_once_in_aligned_chunks() {
        let mut a_seen = vec![false; 128 * 8];
        let mut b_seen = vec![false; 64 * 8];
        for thread in 0..256 {
            for slice in 0..4 {
                let (row, reduction) = a_copy_coordinate(thread, slice);
                assert!(row < 128 && reduction + 4 <= 32);
                let shared = row * 32 + (reduction ^ ((row & 7) << 2));
                assert_eq!(shared * 4 % 16, 0);
                let slot = row * 8 + reduction / 4;
                assert!(!a_seen[slot], "duplicate A slot {slot}");
                a_seen[slot] = true;
                if let Some((column, reduction)) = b_copy_coordinate(thread, slice) {
                    assert!(column < 64 && reduction + 4 <= 32);
                    let shared = column * 32 + (reduction ^ ((column & 7) << 2));
                    assert_eq!(shared * 4 % 16, 0);
                    let slot = column * 8 + reduction / 4;
                    assert!(!b_seen[slot], "duplicate B slot {slot}");
                    b_seen[slot] = true;
                }
            }
        }
        assert!(a_seen.into_iter().all(|seen| seen));
        assert!(b_seen.into_iter().all(|seen| seen));
    }

    #[test]
    fn source_interleaves_slice_before_each_ascending_k8_issue() {
        let source = candidate_source(PRODUCTION, COMPACT_LAYOUT).unwrap();
        assert!(source.contains("gemm_bi_tf32_nt_compact_stage_slice"));
        assert!(source.contains("gemm_bi_tf32_nt_compact_sliced_mainloop"));
        let compute_start = source
            .find("void gemm_bi_tf32_nt_compact_compute_sliced")
            .expect("candidate compute helper");
        let compute_end = source[compute_start..]
            .find("void gemm_bi_tf32_nt_compact_sliced_mainloop")
            .map(|offset| compute_start + offset)
            .expect("candidate mainloop helper");
        let compute = &source[compute_start..compute_end];
        let loop_start = compute
            .find("for (int issue = 0; issue < 4; ++issue)")
            .expect("candidate issue loop");
        let body = &compute[loop_start..];
        let copy = body.find("gemm_bi_tf32_nt_compact_stage_slice").unwrap();
        let load = body
            .find("ldmatrix.sync.aligned.m8n8.x4.shared.b16")
            .unwrap();
        let mma = body.find("gemm_bi_tf32_mma_m16n8k8").unwrap();
        assert!(copy < load && load < mma);
        assert!(compute.contains("const int k_offsets[4] = {0, 8, 16, 24};"));
        assert!(
            source
                .contains("if constexpr (Op == SgbTf32Nt && BM == 128 && BN == 64 && Stages == 2)")
        );
    }

    #[test]
    fn source_transform_is_exactly_reversible_to_the_frozen_a_only_parent() {
        let retained = retained_source(PRODUCTION, COMPACT_LAYOUT).unwrap();
        let candidate = candidate_source(PRODUCTION, COMPACT_LAYOUT).unwrap();
        assert_eq!(restore_retained_source(&candidate).unwrap(), retained);
        assert_eq!(candidate.matches(SYMBOL).count(), 2);
        assert!(!candidate.contains(RETAINED_SYMBOL));
        assert_eq!(
            candidate
                .matches("gemm_bi_tf32_nt_compact_stage_slice(")
                .count(),
            3
        );
        assert_eq!(
            candidate.matches("cp.async.commit_group;").count(),
            retained.matches("cp.async.commit_group;").count() + 3
        );
    }

    #[test]
    fn d768_out_geometry_and_resource_contract_are_frozen() {
        assert_eq!(TARGET, (2_048, 1_536, 768));
        assert_eq!(2_048usize.div_ceil(128) * 1_536usize.div_ceil(64), 384);
        assert_eq!(TARGET_GRID, 384);
        assert_eq!(BLOCK_THREADS, 256);
        assert_eq!(DYNAMIC_SHARED_BYTES, 49_152);
        assert_eq!(REGISTER_CAP, 128);
        assert_eq!(REQUIRED_OCCUPANCY, 2);
    }
}
