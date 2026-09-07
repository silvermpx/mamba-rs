const BM: usize = 128;
const BN: usize = 64;
const BK: usize = 32;
const STAGES: usize = 3;
const CANDIDATE_SYMBOL: &str = "gemm_bi_nt_test_compact_xor_sm80_mma_tf32_v1_m128n64_bk32_s3";
const PADDED_STRIDE: usize = 36;
const PADDED_COPY_PLAN_SYMBOL: &str =
    "gemm_bi_nt_test_padded_copy_plan_sm80_mma_tf32_v1_m128n64_bk32_s3";
const PADDED_LDMATRIX_SYMBOL: &str =
    "gemm_bi_nt_test_padded_ldmatrix_sm80_mma_tf32_v1_m128n64_bk32_s3";
const PRODUCTION_CUDA: &str = include_str!("../kernels/gemm_bi_triad/sm80.cu");
const CANDIDATE_CUDA: &str = include_str!("gemm_bi_tf32_nt_compact_xor.cu");
const PADDED_COPY_PLAN_CUDA: &str = include_str!("gemm_bi_tf32_nt_padded_copy_plan.cuh");
const PADDED_LDMATRIX_CUDA: &str = include_str!("gemm_bi_tf32_nt_padded_ldmatrix.cuh");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CandidateVariant {
    CompactXor,
    PaddedCopyPlan,
    PaddedLdmatrix,
}

impl CandidateVariant {
    const fn name(self) -> &'static str {
        match self {
            Self::CompactXor => "compact_xor",
            Self::PaddedCopyPlan => "padded_copy_plan",
            Self::PaddedLdmatrix => "padded_ldmatrix",
        }
    }

    const fn symbol(self) -> &'static str {
        match self {
            Self::CompactXor => CANDIDATE_SYMBOL,
            Self::PaddedCopyPlan => PADDED_COPY_PLAN_SYMBOL,
            Self::PaddedLdmatrix => PADDED_LDMATRIX_SYMBOL,
        }
    }

    const fn shared_bytes(self) -> u32 {
        match self {
            Self::CompactXor => 73_728,
            Self::PaddedCopyPlan | Self::PaddedLdmatrix => 82_944,
        }
    }

    fn source(self) -> Result<String, String> {
        match self {
            Self::CompactXor => compact_candidate_source(),
            Self::PaddedCopyPlan => padded_copy_plan_candidate_source(),
            Self::PaddedLdmatrix => padded_ldmatrix_candidate_source(),
        }
    }
}

#[cfg(feature = "cuda")]
mod common;

#[cfg(feature = "cuda")]
#[path = "support/fixed_full_mantissa.rs"]
mod fixed_full_mantissa;

const fn compact_k(row_or_column: usize, logical_k: usize) -> usize {
    logical_k ^ ((row_or_column & 7) << 2)
}

const fn compact_index(row_or_column: usize, logical_k: usize) -> usize {
    row_or_column * BK + compact_k(row_or_column, logical_k)
}

fn fragment_banks(row_base: usize, k8: usize, half: usize) -> [usize; 32] {
    std::array::from_fn(|lane| {
        let row = row_base + lane / 4;
        let logical_k = k8 + half + lane % 4;
        compact_index(row, logical_k) % 32
    })
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
            "{label} source boundary count changed: expected {expected}, observed {count}"
        ));
    }
    *source = source.replace(from, to);
    Ok(())
}

fn compact_candidate_source() -> Result<String, String> {
    let mut source = PRODUCTION_CUDA.to_owned();
    replace_exact(
        &mut source,
        "static constexpr int AStride = Op == SgbTf32Tn ? BM + 8 : 36;",
        "static constexpr int AStride = Op == SgbTf32Tn ? BM + 8 : (Op == SgbTf32Nt ? 32 : 36);",
        1,
        "NT A stride",
    )?;
    replace_exact(
        &mut source,
        "static constexpr int BStride = Op == SgbTf32Nt ? 36\n        : (BN == 64 ? 72 : (BN == 32 ? 40 : 24));",
        "static constexpr int BStride = Op == SgbTf32Nt ? 32\n        : (BN == 64 ? 72 : (BN == 32 ? 40 : 24));",
        1,
        "NT B stride",
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
            "    if constexpr (Op == SgbTf32Nt) {\n",
            "        return storage->a[stage][row][gemm_bi_nt_test_compact_xor_k(row, reduction)];\n",
            "    }\n",
            "    return storage->a[stage][row][reduction];"
        ),
        1,
        "NT A slot",
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
            "        return storage->b[stage][column][gemm_bi_nt_test_compact_xor_k(column, reduction)];\n",
            "    }\n",
            "    return storage->b[stage][reduction][column];"
        ),
        1,
        "NT B slot",
    )?;
    for (old, new, label) in [
        (
            "== 55296, \"NT M128N64 s2 storage\"",
            "== 49152, \"NT compact M128N64 s2 storage\"",
            "NT M128N64 S2 extent",
        ),
        (
            "== 82944, \"NT M128N64 s3 storage\"",
            "== 73728, \"NT compact M128N64 s3 storage\"",
            "NT M128N64 S3 extent",
        ),
        (
            "== 36864, \"NT M64N64 s2 storage\"",
            "== 32768, \"NT compact M64N64 s2 storage\"",
            "NT M64N64 S2 extent",
        ),
        (
            "== 55296, \"NT M64N64 s3 storage\"",
            "== 49152, \"NT compact M64N64 s3 storage\"",
            "NT M64N64 S3 extent",
        ),
        (
            "== 27648, \"NT M16N32 s4 storage\"",
            "== 24576, \"NT compact M16N32 s4 storage\"",
            "NT M16N32 S4 extent",
        ),
        (
            "== 20736, \"NT M16N32 s3 storage\"",
            "== 18432, \"NT compact M16N32 s3 storage\"",
            "NT M16N32 S3 extent",
        ),
        (
            "== 27648, \"NT M32N32 s3 storage\"",
            "== 24576, \"NT compact M32N32 s3 storage\"",
            "NT M32N32 S3 extent",
        ),
        (
            "== 36864, \"NT M32N32 s4 storage\"",
            "== 32768, \"NT compact M32N32 s4 storage\"",
            "NT M32N32 S4 extent",
        ),
        (
            "== 18432, \"NT M16N16 s4 storage\"",
            "== 16384, \"NT compact M16N16 s4 storage\"",
            "NT M16N16 S4 extent",
        ),
    ] {
        replace_exact(&mut source, old, new, 1, label)?;
    }
    replace_exact(
        &mut source,
        "gemm_bi_nt_sm80_mma_tf32_v1_",
        "gemm_bi_nt_test_compact_xor_sm80_mma_tf32_v1_",
        12,
        "NT direct symbol isolation",
    )?;
    replace_exact(
        &mut source,
        "gemm_bi_nt_sm80_mma_tf32_splitk",
        "gemm_bi_nt_test_compact_xor_sm80_mma_tf32_splitk",
        8,
        "NT split-K symbol isolation",
    )?;
    Ok(format!("{CANDIDATE_CUDA}\n{source}"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PaddedCopyAssignment {
    axis: usize,
    reduction: usize,
    destination: usize,
}

fn padded_copy_assignment(thread: usize, slice: usize) -> PaddedCopyAssignment {
    let linear = thread + slice * 256;
    let axis = linear >> 3;
    let reduction = (linear & 7) * 4;
    PaddedCopyAssignment {
        axis,
        reduction,
        destination: axis * PADDED_STRIDE + reduction,
    }
}

fn padded_copy_bytes(
    reduction_extent: usize,
    reduction_base: usize,
    reduction_offset: usize,
    axis_valid: bool,
) -> usize {
    if !axis_valid {
        return 0;
    }
    reduction_extent
        .saturating_sub(reduction_base + reduction_offset)
        .min(4)
        * size_of::<f32>()
}

fn padded_copy_plan_candidate_source() -> Result<String, String> {
    let mut source = PRODUCTION_CUDA.to_owned();
    let thread_plan = concat!(
        "struct SgbTf32ThreadPlan {\n",
        "    bool compute;\n",
        "    int warp_m;\n",
        "    int warp_n;\n",
        "    int group;\n",
        "    int thread;\n",
        "};"
    );
    replace_exact(
        &mut source,
        thread_plan,
        &format!("{thread_plan}\n\n{PADDED_COPY_PLAN_CUDA}"),
        1,
        "padded copy-plan helper insertion",
    )?;
    let wide_branch = concat!(
        "    if (wide_a && wide_b) {\n",
        "        gemm_bi_tf32_async_mainloop<\n",
        "            Op, BM, BN, Stages, MAtoms, NAtoms, false, false>(\n",
        "            storage, problem, tile_count, thread_plan, accumulators);\n",
        "    } else if (gemm_bi_tf32_can_stage_async_4(a, b, params)) {"
    );
    let candidate_branch = concat!(
        "    if (wide_a && wide_b) {\n",
        "        if constexpr (Op == SgbTf32Nt && BM == 128 && BN == 64 && Stages == 3) {\n",
        "            gemm_bi_tf32_nt_test_padded_copy_plan_mainloop<MAtoms, NAtoms>(\n",
        "                storage, problem, tile_count, thread_plan, accumulators);\n",
        "        } else {\n",
        "            gemm_bi_tf32_async_mainloop<\n",
        "                Op, BM, BN, Stages, MAtoms, NAtoms, false, false>(\n",
        "                storage, problem, tile_count, thread_plan, accumulators);\n",
        "        }\n",
        "    } else if (gemm_bi_tf32_can_stage_async_4(a, b, params)) {"
    );
    replace_exact(
        &mut source,
        wide_branch,
        candidate_branch,
        1,
        "padded copy-plan target branch",
    )?;
    replace_exact(
        &mut source,
        concat!(
            "GEMM_BI_TF32_DEFINE_KERNEL(",
            "gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s3, ",
            "SgbTf32Nt, 128, 64, 3, 256, 1)"
        ),
        concat!(
            "GEMM_BI_TF32_DEFINE_KERNEL(",
            "gemm_bi_nt_test_padded_copy_plan_sm80_mma_tf32_v1_m128n64_bk32_s3, ",
            "SgbTf32Nt, 128, 64, 3, 256, 1)"
        ),
        1,
        "padded copy-plan target symbol",
    )?;
    Ok(source)
}

fn padded_ldmatrix_candidate_source() -> Result<String, String> {
    let mut source = PRODUCTION_CUDA.to_owned();
    let rna = concat!(
        "__device__ __forceinline__ unsigned gemm_bi_tf32_rna(float value) {\n",
        "    unsigned result;\n",
        "    asm(\"cvt.rna.tf32.f32 %0, %1;\" : \"=r\"(result) : \"f\"(value));\n",
        "    return result;\n",
        "}"
    );
    replace_exact(
        &mut source,
        rna,
        &format!("{rna}\n\n{PADDED_LDMATRIX_CUDA}"),
        1,
        "padded ldmatrix helper insertion",
    )?;
    let scalar_fragment_loads = concat!(
        "        unsigned a_fragments[MAtoms][4];\n",
        "        unsigned b_fragments[NAtoms][2];\n",
        "#pragma unroll\n",
        "        for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {\n",
        "            int row = warp_m + m_atom * 16 + group;\n",
        "            a_fragments[m_atom][0] =\n",
        "                gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row, k8 + thread));\n",
        "            a_fragments[m_atom][1] =\n",
        "                gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row + 8, k8 + thread));\n",
        "            a_fragments[m_atom][2] =\n",
        "                gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row, k8 + thread + 4));\n",
        "            a_fragments[m_atom][3] =\n",
        "                gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row + 8, k8 + thread + 4));\n",
        "        }\n",
        "#pragma unroll\n",
        "        for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {\n",
        "            int column = warp_n + n_atom * 8 + group;\n",
        "            b_fragments[n_atom][0] =\n",
        "                gemm_bi_tf32_rna(gemm_bi_tf32_b_slot<Op>(storage, stage, k8 + thread, column));\n",
        "            b_fragments[n_atom][1] =\n",
        "                gemm_bi_tf32_rna(gemm_bi_tf32_b_slot<Op>(storage, stage, k8 + thread + 4, column));\n",
        "        }"
    );
    let candidate_fragment_loads = format!(
        concat!(
            "        unsigned a_fragments[MAtoms][4];\n",
            "        unsigned b_fragments[NAtoms][2];\n",
            "        if constexpr (Op == SgbTf32Nt && BM == 128 && BN == 64 && Stages == 3) {{\n",
            "            gemm_bi_tf32_nt_padded_ldmatrix_fragments(\n",
            "                storage, stage, warp_m, warp_n, k8, a_fragments, b_fragments);\n",
            "        }} else {{\n",
            "{}\n",
            "        }}"
        ),
        scalar_fragment_loads
            .strip_prefix(
                "        unsigned a_fragments[MAtoms][4];\n        unsigned b_fragments[NAtoms][2];\n"
            )
            .expect("scalar fragment prefix")
            .lines()
            .map(|line| format!("    {line}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    replace_exact(
        &mut source,
        scalar_fragment_loads,
        &candidate_fragment_loads,
        1,
        "padded ldmatrix target fragment loads",
    )?;
    replace_exact(
        &mut source,
        concat!(
            "GEMM_BI_TF32_DEFINE_KERNEL(",
            "gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s3, ",
            "SgbTf32Nt, 128, 64, 3, 256, 1)"
        ),
        concat!(
            "GEMM_BI_TF32_DEFINE_KERNEL(",
            "gemm_bi_nt_test_padded_ldmatrix_sm80_mma_tf32_v1_m128n64_bk32_s3, ",
            "SgbTf32Nt, 128, 64, 3, 256, 1)"
        ),
        1,
        "padded ldmatrix target symbol",
    )?;
    Ok(source)
}

#[test]
fn padded_copy_plan_preserves_six_production_assignments_per_thread() {
    let mut a_seen = vec![false; BM * (BK / 4)];
    let mut b_seen = vec![false; BN * (BK / 4)];
    for thread in 0..256 {
        for slice in 0..4 {
            let got = padded_copy_assignment(thread, slice);
            let linear = thread + slice * 256;
            let expected = PaddedCopyAssignment {
                axis: linear >> 3,
                reduction: (linear & 7) * 4,
                destination: (linear >> 3) * PADDED_STRIDE + (linear & 7) * 4,
            };
            assert_eq!(got, expected, "A thread {thread} slice {slice}");
            assert_eq!(got.destination * size_of::<f32>() % 16, 0);
            assert!(got.reduction + 4 <= BK);
            let slot = got.axis * (BK / 4) + got.reduction / 4;
            assert!(!a_seen[slot], "duplicate A slot {slot}");
            a_seen[slot] = true;
        }
        for slice in 0..2 {
            let got = padded_copy_assignment(thread, slice);
            let linear = thread + slice * 256;
            let expected = PaddedCopyAssignment {
                axis: linear >> 3,
                reduction: (linear & 7) * 4,
                destination: (linear >> 3) * PADDED_STRIDE + (linear & 7) * 4,
            };
            assert_eq!(got, expected, "B thread {thread} slice {slice}");
            assert_eq!(got.destination * size_of::<f32>() % 16, 0);
            assert!(got.reduction + 4 <= BK);
            let slot = got.axis * (BK / 4) + got.reduction / 4;
            assert!(!b_seen[slot], "duplicate B slot {slot}");
            b_seen[slot] = true;
        }
    }
    assert!(a_seen.into_iter().all(|seen| seen));
    assert!(b_seen.into_iter().all(|seen| seen));
}

#[test]
fn padded_copy_plan_clamps_tail_bytes_and_uses_zero_for_invalid_axes() {
    assert_eq!(padded_copy_bytes(35, 32, 0, true), 12);
    assert_eq!(padded_copy_bytes(35, 32, 4, true), 0);
    assert_eq!(padded_copy_bytes(36, 32, 0, true), 16);
    assert_eq!(padded_copy_bytes(36, 32, 4, true), 0);
    assert_eq!(padded_copy_bytes(36, 32, 0, false), 0);
    assert_eq!(padded_copy_bytes(32, 32, 0, true), 0);
}

#[test]
fn padded_copy_plan_source_isolates_one_symbol_without_layout_or_k_order_changes() {
    let source = padded_copy_plan_candidate_source().unwrap();
    assert!(source.contains(PADDED_COPY_PLAN_SYMBOL));
    assert!(!source.contains(CANDIDATE_SYMBOL));
    assert!(!source.contains(PADDED_LDMATRIX_CUDA));
    assert!(source.contains("static constexpr int AStride = Op == SgbTf32Tn ? BM + 8 : 36;"));
    assert!(source.contains("static constexpr int BStride = Op == SgbTf32Nt ? 36"));
    assert!(source.contains("const int k_offsets[4] = {0, 8, 16, 24};"));
    assert!(source.contains("gemm_bi_tf32_nt_test_padded_copy_plan_mainloop"));
    assert!(source.contains(
        "gemm_bi_tf32_async_mainloop<\n                Op, BM, BN, Stages, MAtoms, NAtoms, false, false>"
    ));
}

#[test]
fn padded_ldmatrix_source_isolates_one_symbol_and_only_target_fragment_loads() {
    let source = padded_ldmatrix_candidate_source().unwrap();
    assert!(source.contains(PADDED_LDMATRIX_SYMBOL));
    assert!(source.contains(PADDED_LDMATRIX_CUDA));
    assert!(source.contains("gemm_bi_tf32_nt_padded_ldmatrix_fragments"));
    assert!(source.contains("static constexpr int AStride = Op == SgbTf32Tn ? BM + 8 : 36;"));
    assert!(source.contains("static constexpr int BStride = Op == SgbTf32Nt ? 36"));
    assert!(source.contains("const int k_offsets[4] = {0, 8, 16, 24};"));
    assert!(!source.contains(PADDED_COPY_PLAN_CUDA));
    assert!(!source.contains(CANDIDATE_SYMBOL));
}

#[test]
fn candidate_variants_have_independent_sources_symbols_and_shared_extents() {
    let compact = CandidateVariant::CompactXor;
    let copy_plan = CandidateVariant::PaddedCopyPlan;
    let ldmatrix = CandidateVariant::PaddedLdmatrix;
    assert_eq!(compact.shared_bytes(), 73_728);
    assert_eq!(copy_plan.shared_bytes(), 82_944);
    assert_eq!(ldmatrix.shared_bytes(), 82_944);
    assert_ne!(compact.symbol(), copy_plan.symbol());
    assert_ne!(compact.symbol(), ldmatrix.symbol());
    assert_ne!(copy_plan.symbol(), ldmatrix.symbol());
    for variant in [copy_plan, ldmatrix] {
        let source = variant.source().unwrap();
        assert!(source.contains(variant.symbol()));
        assert!(source.contains("NT M128N64 s3 storage"));
        assert_eq!(variant.name().starts_with("padded_"), true);
    }
}

#[test]
fn compact_xor_maps_each_row_by_its_low_three_bits() {
    assert_eq!(compact_k(0, 0), 0);
    assert_eq!(compact_k(1, 0), 4);
    assert_eq!(compact_k(7, 0), 28);
    assert_eq!(compact_k(9, 12), 8);
}

#[test]
fn compact_xor_is_bounded_bijective_and_preserves_four_float_groups() {
    for row in 0..BM {
        let mut seen = [false; BK];
        for logical_k in 0..BK {
            let physical_k = compact_k(row, logical_k);
            assert!(physical_k < BK, "row={row} k={logical_k}");
            assert!(!seen[physical_k], "row={row} physical_k={physical_k}");
            seen[physical_k] = true;
        }
        assert!(seen.into_iter().all(|value| value));
        for logical_k in (0..BK).step_by(4) {
            let physical_k = compact_k(row, logical_k);
            assert_eq!(physical_k % 4, 0, "row={row} k={logical_k}");
            assert!(physical_k + 3 < BK, "row={row} k={logical_k}");
            for element in 0..4 {
                assert_eq!(
                    compact_k(row, logical_k + element),
                    physical_k + element,
                    "row={row} k={logical_k} element={element}"
                );
            }
        }
    }
}

#[test]
fn actual_nt_a_and_b_fragment_formulas_are_bank_permutations() {
    for warp in 0..4 {
        let warp_m = (warp >> 1) * 64;
        let warp_n = (warp & 1) * 32;
        for k8 in [0, 8, 16, 24] {
            for half in [0, 4] {
                for atom in 0..4 {
                    for (operand, row_base) in [("A", warp_m + atom * 16), ("B", warp_n + atom * 8)]
                    {
                        let banks = fragment_banks(row_base, k8, half);
                        let mut seen = [false; 32];
                        for bank in banks {
                            assert!(
                                !seen[bank],
                                "operand={operand} warp={warp} atom={atom} k8={k8} half={half}"
                            );
                            seen[bank] = true;
                        }
                        assert!(seen.into_iter().all(|value| value));
                    }
                }
            }
        }
    }
}

#[test]
fn compact_candidate_shared_extent_is_exact() {
    assert_eq!(STAGES * (BM * BK + BN * BK) * size_of::<f32>(), 73_728);
}

#[test]
fn transformed_source_binds_only_distinct_nt_symbols_and_compact_slots() {
    let source = compact_candidate_source().unwrap();
    assert!(source.contains(CANDIDATE_SYMBOL));
    assert!(!source.contains("gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s3"));
    assert!(source.contains("logical_k ^ ((row_or_column & 7) << 2)"));
    assert!(source.contains("NT compact M128N64 s3 storage"));
}

#[cfg(feature = "cuda")]
mod cuda_suite {
    use std::ffi::CStr;

    use common::gpu_quiet::QuietGpu;
    use cudarc::driver::{CudaFunction, CudaGraph, DeviceRepr, LaunchConfig, PushKernelArg, sys};
    use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
    use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
        PhysicalQualificationF32Epilogue, PhysicalQualificationRequest, PhysicalQualificationRoute,
        QualifiedPhysicalLaunch, Tf32PhysicalRoute, Tf32PortableRoute, Tf32PortableStages,
        Tf32PortableTile, presize_physical_qualification_suite, qualify_physical_launch,
    };
    use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
    use mamba_rs::mamba_ssm::gpu::kernel_identity::{
        ModuleKind, PhysicalLaunchKind, PolicyDtype, ResolvedGemmOp,
    };
    use sha2::{Digest as _, Sha256};

    use super::*;

    const TARGET: (usize, usize, usize) = (2_048, 768, 3_072);
    const TAIL: (usize, usize, usize) = (129, 65, 36);
    const AUTO_SYMBOL: &str = "gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s3";
    const AUTO_SHARED_BYTES: u32 = 82_944;
    const GUARD: usize = 32;
    const GUARD_BITS: u32 = 0x7fc1_4e54;
    const WINDOWS: usize = 7;
    const WARMUPS: usize = 64;
    const PILOT: usize = 16;
    const TARGET_WINDOW_US: f64 = 5_000.0;
    const MAX_ITERATIONS: usize = 4_096;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Params {
        alpha: f32,
        beta: f32,
        m: i32,
        k: i32,
        n: i32,
        lda: i32,
        ldb: i32,
        ldc: i32,
    }

    unsafe impl DeviceRepr for Params {}

    struct GuardedBuffer {
        buffer: GpuBuffer,
        active: Vec<u32>,
    }

    impl GuardedBuffer {
        fn new(ctx: &GpuCtx, active: Vec<u32>) -> Result<Self, String> {
            let mut words = vec![GUARD_BITS; active.len() + 2 * GUARD];
            words[GUARD..GUARD + active.len()].copy_from_slice(&active);
            let values = words.into_iter().map(f32::from_bits).collect::<Vec<_>>();
            let buffer = GpuBuffer::from_cpu(&ctx.stream, &values)?;
            ctx.stream
                .synchronize()
                .map_err(|error| format!("synchronize guarded candidate allocation: {error:?}"))?;
            Ok(Self { buffer, active })
        }

        fn reset(&mut self, ctx: &GpuCtx) -> Result<(), String> {
            let mut words = vec![GUARD_BITS; self.active.len() + 2 * GUARD];
            words[GUARD..GUARD + self.active.len()].copy_from_slice(&self.active);
            let values = words.into_iter().map(f32::from_bits).collect::<Vec<_>>();
            self.buffer.upload(&ctx.stream, &values)?;
            ctx.stream
                .synchronize()
                .map_err(|error| format!("synchronize guarded candidate reset: {error:?}"))
        }

        fn pointer(&self, ctx: &GpuCtx) -> Result<u64, String> {
            let pointer = self.buffer.raw_ptr_at(&ctx.stream, GUARD);
            if pointer == 0 || !pointer.is_multiple_of(16) {
                return Err(format!(
                    "guarded candidate pointer is not 16-byte aligned: {pointer:#x}"
                ));
            }
            Ok(pointer)
        }

        fn snapshot(&self, ctx: &GpuCtx, immutable: bool) -> Result<Vec<u32>, String> {
            let values = self.buffer.to_cpu(&ctx.stream)?;
            ctx.stream
                .synchronize()
                .map_err(|error| format!("synchronize guarded candidate readback: {error:?}"))?;
            let words = values.into_iter().map(f32::to_bits).collect::<Vec<_>>();
            if words.len() != self.active.len() + 2 * GUARD
                || words[..GUARD].iter().any(|word| *word != GUARD_BITS)
                || words[GUARD + self.active.len()..]
                    .iter()
                    .any(|word| *word != GUARD_BITS)
            {
                return Err("candidate guarded allocation changed".into());
            }
            let active = words[GUARD..GUARD + self.active.len()].to_vec();
            if immutable && active != self.active {
                return Err("candidate input allocation changed".into());
            }
            Ok(active)
        }
    }

    struct Candidate {
        variant: CandidateVariant,
        graph: CudaGraph,
        function: CudaFunction,
        output: GuardedBuffer,
        a: GuardedBuffer,
        b: GuardedBuffer,
        config: LaunchConfig,
        params: Params,
        ctx: GpuCtx,
    }

    #[derive(Clone, Copy)]
    enum Path {
        Eager,
        Graph,
    }

    impl Path {
        const fn name(self) -> &'static str {
            match self {
                Self::Eager => "eager",
                Self::Graph => "graph",
            }
        }
    }

    #[derive(Clone, Copy)]
    enum Order {
        Abba,
        Baab,
    }

    impl Order {
        const fn name(self) -> &'static str {
            match self {
                Self::Abba => "ABBA",
                Self::Baab => "BAAB",
            }
        }
    }

    fn checked_len(left: usize, right: usize, label: &str) -> Result<usize, String> {
        left.checked_mul(right)
            .ok_or_else(|| format!("{label} extent overflow"))
    }

    fn fixture_words(
        dims: (usize, usize, usize),
    ) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>), String> {
        let (m, k, n) = dims;
        let output = fixed_full_mantissa::finite_full_mantissa_values(
            checked_len(m, k, "output")?,
            0xc040_4e54,
        );
        let a =
            fixed_full_mantissa::finite_full_mantissa_values(checked_len(m, n, "A")?, 0xa040_4e54);
        let b =
            fixed_full_mantissa::finite_full_mantissa_values(checked_len(k, n, "B")?, 0xb040_4e54);
        Ok((
            output.into_iter().map(f32::to_bits).collect(),
            a.into_iter().map(f32::to_bits).collect(),
            b.into_iter().map(f32::to_bits).collect(),
        ))
    }

    fn compose_source(variant: CandidateVariant) -> Result<String, String> {
        let transformed = variant.source()?;
        Ok([
            include_str!("../kernels/_typed_prelude.cuh"),
            include_str!("../kernels/gemm_bi_triad/contract.cuh"),
            include_str!("../kernels/gemm_bi_triad/common.cuh"),
            include_str!("../kernels/gemm_bi_triad/epilogue.cuh"),
            include_str!("../kernels/gemm_bi_triad/mma16.cuh"),
            &transformed,
        ]
        .iter()
        .map(|part| {
            part.lines()
                .filter(|line| !line.trim().starts_with("#include \"_typed_prelude.cuh\""))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .collect::<Vec<_>>()
        .join("\n"))
    }

    fn compile_candidate(
        ctx: &GpuCtx,
        variant: CandidateVariant,
    ) -> Result<(CudaFunction, String), String> {
        let source = compose_source(variant)?;
        let source_sha = format!("{:x}", Sha256::digest(source.as_bytes()));
        let options = cudarc::nvrtc::CompileOptions {
            arch: Some("compute_89"),
            options: vec![
                "--fmad=true".into(),
                "--extra-device-vectorization".into(),
                "-DNDEBUG".into(),
                "-DGEMM_BI_GROUP_M=16".into(),
                "-DMAMBA_RS_STATE_CAP=256".into(),
                "--frandom-seed=1295072049".into(),
            ],
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(source, options)
            .map_err(|error| format!("compile {} NT candidate: {error:?}", variant.name()))?;
        let module = ctx
            .stream
            .context()
            .load_module(ptx)
            .map_err(|error| format!("load {} NT candidate: {error:?}", variant.name()))?;
        let function = module
            .load_function(variant.symbol())
            .map_err(|error| format!("load {} NT symbol: {error:?}", variant.name()))?;
        function
            .set_attribute(
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                variant.shared_bytes() as i32,
            )
            .map_err(|error| format!("set {} NT dynamic shared: {error:?}", variant.name()))?;
        Ok((function, source_sha))
    }

    fn launch_candidate(
        ctx: &GpuCtx,
        function: &CudaFunction,
        output: &GuardedBuffer,
        a: &GuardedBuffer,
        b: &GuardedBuffer,
        config: LaunchConfig,
        params: Params,
        variant: CandidateVariant,
    ) -> Result<(), String> {
        let output = output.pointer(ctx)?;
        let a = a.pointer(ctx)?;
        let b = b.pointer(ctx)?;
        let bias = 0_u64;
        let mut builder = ctx.stream.launch_builder(function);
        builder.arg(&output);
        builder.arg(&a);
        builder.arg(&b);
        builder.arg(&bias);
        builder.arg(&params);
        unsafe { builder.launch(config) }
            .map(|_| ())
            .map_err(|error| format!("launch {} NT candidate: {error:?}", variant.name()))
    }

    impl Candidate {
        fn new(
            device: &GpuDevice,
            variant: CandidateVariant,
            dims: (usize, usize, usize),
            words: &(Vec<u32>, Vec<u32>, Vec<u32>),
        ) -> Result<(Self, String), String> {
            if size_of::<Params>() != 32 || align_of::<Params>() != 4 {
                return Err(format!(
                    "{} NT parameter ABI changed: size={} align={}",
                    variant.name(),
                    size_of::<Params>(),
                    align_of::<Params>()
                ));
            }
            let ctx = GpuCtx::new(device)?;
            let (function, source_sha) = compile_candidate(&ctx, variant)?;
            let output = GuardedBuffer::new(&ctx, words.0.clone())?;
            let a = GuardedBuffer::new(&ctx, words.1.clone())?;
            let b = GuardedBuffer::new(&ctx, words.2.clone())?;
            let config = LaunchConfig {
                grid_dim: (
                    (dims.0 as u32).div_ceil(BM as u32) * (dims.1 as u32).div_ceil(BN as u32),
                    1,
                    1,
                ),
                block_dim: (256, 1, 1),
                shared_mem_bytes: variant.shared_bytes(),
            };
            let params = Params {
                alpha: 1.0,
                beta: 0.0,
                m: i32::try_from(dims.0).map_err(|_| "M exceeds i32")?,
                k: i32::try_from(dims.1).map_err(|_| "K exceeds i32")?,
                n: i32::try_from(dims.2).map_err(|_| "N exceeds i32")?,
                lda: i32::try_from(dims.2).map_err(|_| "lda exceeds i32")?,
                ldb: i32::try_from(dims.2).map_err(|_| "ldb exceeds i32")?,
                ldc: i32::try_from(dims.1).map_err(|_| "ldc exceeds i32")?,
            };
            let graph = unsafe {
                capture_into_graph(&ctx.stream, || {
                    launch_candidate(&ctx, &function, &output, &a, &b, config, params, variant)
                })
            }?;
            validate_candidate_graph(&graph, config, variant)?;
            Ok((
                Self {
                    variant,
                    graph,
                    function,
                    output,
                    a,
                    b,
                    config,
                    params,
                    ctx,
                },
                source_sha,
            ))
        }

        fn reset(&mut self) -> Result<(), String> {
            self.output.reset(&self.ctx)?;
            self.a.reset(&self.ctx)?;
            self.b.reset(&self.ctx)
        }

        fn launch(&self, path: Path) -> Result<(), String> {
            match path {
                Path::Eager => launch_candidate(
                    &self.ctx,
                    &self.function,
                    &self.output,
                    &self.a,
                    &self.b,
                    self.config,
                    self.params,
                    self.variant,
                ),
                Path::Graph => self
                    .graph
                    .launch()
                    .map_err(|error| format!("launch {} NT graph: {error:?}", self.variant.name())),
            }
        }

        fn output_bits(&self) -> Result<Vec<u32>, String> {
            let bits = self.output.snapshot(&self.ctx, false)?;
            self.a.snapshot(&self.ctx, true)?;
            self.b.snapshot(&self.ctx, true)?;
            Ok(bits)
        }

        fn measure(&self, path: Path, iterations: usize) -> Result<f64, String> {
            let start = self
                .ctx
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|error| format!("record candidate start: {error:?}"))?;
            for _ in 0..iterations {
                self.launch(path)?;
            }
            let end = self
                .ctx
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|error| format!("record candidate end: {error:?}"))?;
            let us = f64::from(
                start
                    .elapsed_ms(&end)
                    .map_err(|error| format!("measure candidate: {error:?}"))?,
            ) * 1_000.0
                / iterations as f64;
            positive(us, "candidate")
        }
    }

    fn validate_candidate_graph(
        graph: &CudaGraph,
        expected: LaunchConfig,
        variant: CandidateVariant,
    ) -> Result<(), String> {
        let mut count = 0usize;
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count) },
            "count candidate graph nodes",
        )?;
        if count != 1 {
            return Err(format!("{} NT graph has {count} nodes", variant.name()));
        }
        let mut node = std::ptr::null_mut();
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(graph.cu_graph(), &mut node, &mut count) },
            "read candidate graph node",
        )?;
        let mut params = unsafe { std::mem::zeroed() };
        cuda_ok(
            unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) },
            "read candidate graph params",
        )?;
        let mut name = std::ptr::null();
        cuda_ok(
            unsafe { sys::cuFuncGetName(&mut name, params.func) },
            "read candidate graph symbol",
        )?;
        if name.is_null() {
            return Err(format!(
                "{} NT graph returned a null symbol name",
                variant.name()
            ));
        }
        let name = unsafe { CStr::from_ptr(name) }.to_string_lossy();
        if name != variant.symbol()
            || (params.gridDimX, params.gridDimY, params.gridDimZ) != expected.grid_dim
            || (params.blockDimX, params.blockDimY, params.blockDimZ) != expected.block_dim
            || params.sharedMemBytes != expected.shared_mem_bytes
        {
            return Err(format!(
                "{} NT graph identity changed: symbol={name} grid={:?} block={:?} shared={}",
                variant.name(),
                (params.gridDimX, params.gridDimY, params.gridDimZ),
                (params.blockDimX, params.blockDimY, params.blockDimZ),
                params.sharedMemBytes
            ));
        }
        for (index, expected) in [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)]
            .into_iter()
            .enumerate()
        {
            let mut offset = 0;
            let mut size = 0;
            cuda_ok(
                unsafe { sys::cuFuncGetParamInfo(params.func, index, &mut offset, &mut size) },
                &format!("read {} NT ABI parameter {index}", variant.name()),
            )?;
            if (offset, size) != expected {
                return Err(format!(
                    "{} NT ABI parameter {index} changed: {:?}",
                    variant.name(),
                    (offset, size)
                ));
            }
        }
        let mut offset = 0;
        let mut size = 0;
        if unsafe { sys::cuFuncGetParamInfo(params.func, 5, &mut offset, &mut size) }
            != sys::CUresult::CUDA_ERROR_INVALID_VALUE
        {
            return Err(format!(
                "{} NT ABI accepted a sixth argument",
                variant.name()
            ));
        }
        Ok(())
    }

    fn cuda_ok(result: sys::CUresult, label: &str) -> Result<(), String> {
        if result == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{label}: {result:?}"))
        }
    }

    fn validate_resources(candidate: &Candidate, source_sha: &str) -> Result<(), String> {
        let function = &candidate.function;
        let variant = candidate.variant;
        let registers = function
            .num_regs()
            .map_err(|error| format!("candidate registers: {error:?}"))?;
        let local = function
            .local_size_bytes()
            .map_err(|error| format!("candidate local bytes: {error:?}"))?;
        let static_shared = function
            .shared_size_bytes()
            .map_err(|error| format!("candidate static shared: {error:?}"))?;
        let max_threads = function
            .max_threads_per_block()
            .map_err(|error| format!("candidate max threads: {error:?}"))?;
        let max_dynamic = function
            .get_attribute(
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            )
            .map_err(|error| format!("candidate max dynamic shared: {error:?}"))?;
        let occupancy = function
            .occupancy_max_active_blocks_per_multiprocessor(
                256,
                variant.shared_bytes() as usize,
                None,
            )
            .map_err(|error| format!("candidate occupancy: {error:?}"))?;
        if registers <= 0
            || local != 0
            || static_shared != 0
            || max_threads < 256
            || max_dynamic < variant.shared_bytes() as i32
            || occupancy < 1
        {
            return Err(format!(
                "{} NT resource gate failed: regs={registers} local={local} static={static_shared} max_threads={max_threads} max_dynamic={max_dynamic} occupancy={occupancy}",
                variant.name()
            ));
        }
        println!(
            "{{\"schema\":\"MambaBiTf32NtDiscoveryResourceV1\",\"variant\":\"{}\",\"source_sha256\":\"{source_sha}\",\"symbol\":\"{}\",\"registers\":{registers},\"local_bytes\":{local},\"static_shared_bytes\":{static_shared},\"dynamic_shared_bytes\":{},\"max_dynamic_shared_bytes\":{max_dynamic},\"max_threads\":{max_threads},\"occupancy\":{occupancy}}}",
            variant.name(),
            variant.symbol(),
            variant.shared_bytes()
        );
        Ok(())
    }

    fn configure(device: &GpuDevice) -> Result<GpuCtx, String> {
        let ctx = GpuCtx::new(device)?;
        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_fast_gemm(false);
        ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
        Ok(ctx)
    }

    fn request(dims: (usize, usize, usize), actual_auto: bool) -> PhysicalQualificationRequest {
        let route = if actual_auto {
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::AllowDeterministicTf32V1)
        } else {
            PhysicalQualificationRoute::Tf32Forced(Tf32PhysicalRoute::MmaTf32RnaV1(
                Tf32PortableRoute {
                    tile: Tf32PortableTile::M128N64,
                    stages: Tf32PortableStages::S3,
                },
            ))
        };
        PhysicalQualificationRequest::contiguous_f32(
            ResolvedGemmOp::Nt,
            dims,
            route,
            PhysicalQualificationF32Epilogue::new(1.0, 0.0, false),
        )
    }

    fn validate_reference(
        dims: (usize, usize, usize),
        actual_auto: bool,
        launch: &QualifiedPhysicalLaunch<'_>,
    ) -> Result<(), String> {
        let evidence = launch.evidence();
        let [node] = evidence.nodes() else {
            return Err(format!(
                "reference has {} physical nodes",
                evidence.launch_count()
            ));
        };
        let grid = (
            (dims.0 as u32).div_ceil(BM as u32) * (dims.1 as u32).div_ceil(BN as u32),
            1,
            1,
        );
        if !evidence.eager_graph_equal()
            || evidence.single_launch_symbol() != Some(AUTO_SYMBOL)
            || evidence.uniform_module_kind() != Some(ModuleKind::TriadSm80)
            || evidence.uniform_execution_dtype() != Some(PolicyDtype::F32)
            || node.kind != PhysicalLaunchKind::Gemm
            || node.logical_op != ResolvedGemmOp::Nt
            || node.shape != dims
            || node.strides != (dims.2, dims.2, dims.1)
            || node.launch.grid_dim != grid
            || node.launch.block_dim != (256, 1, 1)
            || node.launch.shared_mem_bytes != AUTO_SHARED_BYTES
        {
            return Err(format!(
                "{} reference identity changed: {evidence:?}",
                if actual_auto { "AUTO" } else { "forced" }
            ));
        }
        Ok(())
    }

    fn upload_reference(
        launch: &mut QualifiedPhysicalLaunch<'_>,
        ctx: &GpuCtx,
        words: &(Vec<u32>, Vec<u32>, Vec<u32>),
    ) -> Result<(), String> {
        launch.upload_exact_unbiased_f32_words(ctx, &words.0, &words.1, &words.2)
    }

    fn check_bits(
        reference: &mut QualifiedPhysicalLaunch<'_>,
        reference_ctx: &GpuCtx,
        candidate: &mut Candidate,
        words: &(Vec<u32>, Vec<u32>, Vec<u32>),
        label: &str,
    ) -> Result<Vec<u32>, String> {
        let mut expected = None;
        for path in [Path::Eager, Path::Graph] {
            for repeat in 0..2 {
                upload_reference(reference, reference_ctx, words)?;
                match path {
                    Path::Eager => reference.measure_eager_window_ms(reference_ctx, 1)?,
                    Path::Graph => reference.measure_graph_window_ms(reference_ctx, 1)?,
                };
                let actual = reference.f32_output_bits(reference_ctx)?;
                let operands = reference.f32_operand_bits(reference_ctx)?;
                if operands.0 != words.1 || operands.1 != words.2 {
                    return Err(format!(
                        "{label} reference {} repeat {repeat} changed an input",
                        path.name()
                    ));
                }
                reference.validate_red_zones(reference_ctx)?;
                if expected
                    .as_ref()
                    .is_some_and(|expected| expected != &actual)
                {
                    return Err(format!(
                        "{label} reference {path_name} repeat {repeat} changed bits",
                        path_name = path.name()
                    ));
                }
                expected.get_or_insert(actual);
                candidate.reset()?;
                candidate.launch(path)?;
                candidate
                    .ctx
                    .stream
                    .synchronize()
                    .map_err(|error| format!("synchronize candidate correctness: {error:?}"))?;
                let actual = candidate.output_bits()?;
                if expected.as_ref() != Some(&actual) {
                    let mismatch = expected
                        .as_ref()
                        .unwrap()
                        .iter()
                        .zip(&actual)
                        .position(|(left, right)| left != right)
                        .unwrap_or(actual.len());
                    return Err(format!(
                        "{label} {} candidate differs from reference at {mismatch} on {} repeat {repeat}",
                        candidate.variant.name(),
                        path.name(),
                    ));
                }
            }
        }
        Ok(expected.unwrap())
    }

    fn positive(value: f64, label: &str) -> Result<f64, String> {
        if value.is_finite() && value > 0.0 {
            Ok(value)
        } else {
            Err(format!("{label} timing is invalid: {value}"))
        }
    }

    fn measure_reference(
        launch: &mut QualifiedPhysicalLaunch<'_>,
        ctx: &GpuCtx,
        path: Path,
        iterations: usize,
    ) -> Result<f64, String> {
        let ms = match path {
            Path::Eager => launch.measure_eager_window_ms(ctx, iterations)?,
            Path::Graph => launch.measure_graph_window_ms(ctx, iterations)?,
        };
        positive(ms * 1_000.0 / iterations as f64, "AUTO")
    }

    fn percentile(values: &[f64], fraction: f64) -> f64 {
        let mut values = values.to_vec();
        values.sort_by(f64::total_cmp);
        values[((values.len() - 1) as f64 * fraction).round() as usize]
    }

    fn json_f64s(values: &[f64]) -> String {
        format!(
            "[{}]",
            values
                .iter()
                .map(|value| format!("{value:.9}"))
                .collect::<Vec<_>>()
                .join(",")
        )
    }

    fn json_pairs(values: &[[f64; 2]]) -> String {
        format!(
            "[{}]",
            values
                .iter()
                .map(|[left, right]| format!("[{left:.9},{right:.9}]"))
                .collect::<Vec<_>>()
                .join(",")
        )
    }

    fn screen(
        reference: &mut QualifiedPhysicalLaunch<'_>,
        reference_ctx: &GpuCtx,
        candidate: &Candidate,
        path: Path,
        order: Order,
    ) -> Result<(f64, f64), String> {
        measure_reference(reference, reference_ctx, path, WARMUPS)?;
        candidate.measure(path, WARMUPS)?;
        let auto_pilot = measure_reference(reference, reference_ctx, path, PILOT)?;
        let candidate_pilot = candidate.measure(path, PILOT)?;
        let calibrated =
            |us: f64| ((TARGET_WINDOW_US / us).round() as usize).clamp(1, MAX_ITERATIONS);
        let iterations = calibrated(auto_pilot).max(calibrated(candidate_pilot));
        let mut ratios = Vec::with_capacity(WINDOWS);
        let mut auto_samples = Vec::with_capacity(WINDOWS);
        let mut candidate_samples = Vec::with_capacity(WINDOWS);
        let mut brackets = Vec::with_capacity(WINDOWS);
        for _ in 0..WINDOWS {
            let (a0, c0, c1, a1) = match order {
                Order::Abba => (
                    measure_reference(reference, reference_ctx, path, iterations)?,
                    candidate.measure(path, iterations)?,
                    candidate.measure(path, iterations)?,
                    measure_reference(reference, reference_ctx, path, iterations)?,
                ),
                Order::Baab => {
                    let c0 = candidate.measure(path, iterations)?;
                    let a0 = measure_reference(reference, reference_ctx, path, iterations)?;
                    let a1 = measure_reference(reference, reference_ctx, path, iterations)?;
                    let c1 = candidate.measure(path, iterations)?;
                    (a0, c0, c1, a1)
                }
            };
            let auto = (a0 + a1) * 0.5;
            let candidate = (c0 + c1) * 0.5;
            auto_samples.push(auto);
            candidate_samples.push(candidate);
            ratios.push(candidate / auto);
            brackets.push([a0, c0, c1, a1]);
        }
        let p50 = percentile(&ratios, 0.50);
        let p95 = percentile(&ratios, 0.95);
        let variant = candidate.variant;
        let brackets = format!(
            "[{}]",
            brackets
                .iter()
                .map(|[a0, c0, c1, a1]| { format!("[{a0:.9},{c0:.9},{c1:.9},{a1:.9}]") })
                .collect::<Vec<_>>()
                .join(",")
        );
        println!(
            "{{\"schema\":\"MambaBiTf32NtDiscoveryScreenV1\",\"variant\":\"{}\",\"symbol\":\"{}\",\"dynamic_shared_bytes\":{},\"path\":\"{}\",\"order\":\"{}\",\"windows\":{WINDOWS},\"iterations\":{iterations},\"bracket_fields\":[\"auto0_us\",\"candidate0_us\",\"candidate1_us\",\"auto1_us\"],\"brackets\":{brackets},\"ratio_direction\":\"candidate_over_actual_auto\",\"ratio_p50\":{p50:.9},\"ratio_p95\":{p95:.9},\"auto_samples_us\":{},\"candidate_samples_us\":{},\"ratios\":{}}}",
            variant.name(),
            variant.symbol(),
            variant.shared_bytes(),
            path.name(),
            order.name(),
            json_f64s(&auto_samples),
            json_f64s(&candidate_samples),
            json_f64s(&ratios)
        );
        Ok((p50, p95))
    }

    fn run(variant: CandidateVariant) {
        assert!(
            !cfg!(debug_assertions),
            "{} NT discovery requires --release",
            variant.name()
        );
        let quiet = QuietGpu::for_cuda_ordinal(0).unwrap();
        let cohort = format!("tf32-nt-{}/", variant.name().replace('_', "-"));
        let pre = quiet.require_pre_context(&format!("{cohort}pre")).unwrap();
        let device = GpuDevice::new(0).unwrap();
        assert_eq!(device.compute_capability, (8, 9));

        let target_words = fixture_words(TARGET).unwrap();
        let auto_ctx = configure(&device).unwrap();
        let auto_request = request(TARGET, true);
        presize_physical_qualification_suite(&auto_ctx, &[auto_request]).unwrap();
        let mut actual_auto = qualify_physical_launch(&auto_ctx, auto_request).unwrap();
        validate_reference(TARGET, true, &actual_auto).unwrap();
        let compiler = actual_auto.evidence().route_identity().compiler;
        assert_eq!(compiler.nvrtc_version, (13, 2));
        assert_eq!(compiler.target.as_str(), "sm_89");
        assert!(compiler.nvrtc_library_known);
        let (mut candidate, source_sha) =
            Candidate::new(&device, variant, TARGET, &target_words).unwrap();
        validate_resources(&candidate, &source_sha).unwrap();
        let golden = check_bits(
            &mut actual_auto,
            &auto_ctx,
            &mut candidate,
            &target_words,
            "target",
        )
        .unwrap();

        let tail_words = fixture_words(TAIL).unwrap();
        let tail_ctx = configure(&device).unwrap();
        let tail_request = request(TAIL, false);
        presize_physical_qualification_suite(&tail_ctx, &[tail_request]).unwrap();
        let mut tail_reference = qualify_physical_launch(&tail_ctx, tail_request).unwrap();
        validate_reference(TAIL, false, &tail_reference).unwrap();
        let (mut tail_candidate, tail_source_sha) =
            Candidate::new(&device, variant, TAIL, &tail_words).unwrap();
        assert_eq!(tail_source_sha, source_sha);
        check_bits(
            &mut tail_reference,
            &tail_ctx,
            &mut tail_candidate,
            &tail_words,
            "tail",
        )
        .unwrap();
        drop(tail_reference);
        drop(tail_candidate);

        upload_reference(&mut actual_auto, &auto_ctx, &target_words).unwrap();
        candidate.reset().unwrap();
        let timed_pre = quiet.require_cohort(&format!("{cohort}timed")).unwrap();
        let mut strata = Vec::new();
        for path in [Path::Eager, Path::Graph] {
            for order in [Order::Abba, Order::Baab] {
                strata.push(screen(&mut actual_auto, &auto_ctx, &candidate, path, order).unwrap());
            }
        }
        upload_reference(&mut actual_auto, &auto_ctx, &target_words).unwrap();
        actual_auto.measure_graph_window_ms(&auto_ctx, 1).unwrap();
        assert_eq!(actual_auto.f32_output_bits(&auto_ctx).unwrap(), golden);
        assert_eq!(
            actual_auto.f32_operand_bits(&auto_ctx).unwrap(),
            (target_words.1.clone(), target_words.2.clone())
        );
        actual_auto.validate_red_zones(&auto_ctx).unwrap();
        candidate.reset().unwrap();
        candidate.launch(Path::Graph).unwrap();
        candidate.ctx.stream.synchronize().unwrap();
        assert_eq!(candidate.output_bits().unwrap(), golden);
        let post = quiet.verify_post_cohort(&format!("{cohort}post")).unwrap();
        let retain = strata.iter().all(|(p50, p95)| *p50 < 0.99 && *p95 < 0.99);
        let strata = strata
            .into_iter()
            .map(|(p50, p95)| [p50, p95])
            .collect::<Vec<_>>();
        println!(
            "{{\"schema\":\"MambaBiTf32NtDiscoveryDecisionV1\",\"variant\":\"{}\",\"symbol\":\"{}\",\"dynamic_shared_bytes\":{},\"shape\":[2048,768,3072],\"source_sha256\":\"{source_sha}\",\"pre\":{pre:?},\"timed_pre\":{timed_pre:?},\"post\":{post:?},\"strata_fields\":[\"ratio_p50\",\"ratio_p95\"],\"strata\":{},\"retain\":{retain},\"decision\":\"{}\",\"promotion\":false}}",
            variant.name(),
            variant.symbol(),
            variant.shared_bytes(),
            json_pairs(&strata),
            if retain {
                "advance_to_full_qualification"
            } else {
                "stop_no_retry"
            }
        );
    }

    #[test]
    #[ignore = "requires exclusive Ada CC8.9 CUDA13.2; archived compact NT discovery"]
    fn ada_tf32_nt_compact_xor_discovery_once7() {
        run(CandidateVariant::CompactXor);
    }

    #[test]
    #[ignore = "requires exclusive Ada CC8.9 CUDA13.2; padded copy-plan NT discovery"]
    fn ada_tf32_nt_padded_copy_plan_discovery_once7() {
        run(CandidateVariant::PaddedCopyPlan);
    }

    #[test]
    #[ignore = "requires exclusive Ada CC8.9 CUDA13.2; padded ldmatrix NT discovery"]
    fn ada_tf32_nt_padded_ldmatrix_discovery_once7() {
        run(CandidateVariant::PaddedLdmatrix);
    }
}
