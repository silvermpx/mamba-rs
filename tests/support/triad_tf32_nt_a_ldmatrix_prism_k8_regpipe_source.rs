#[path = "triad_tf32_nt_compact_a_ldmatrix_sliced_source.rs"]
mod composed_parent;

pub const SYMBOL: &str =
    "gemm_bi_nt_test_a_ldmatrix_prism_k8_regpipe_sm80_mma_tf32_v1_m128n64_bk32_s2";
pub const RETAINED_SYMBOL: &str = composed_parent::RETAINED_SYMBOL;
pub const BLOCK_THREADS: u32 = 256;
pub const DYNAMIC_SHARED_BYTES: u32 = 49_152;
pub const REGISTER_CAP: i32 = 128;
pub const REQUIRED_OCCUPANCY: u32 = 2;
pub const TARGET: (usize, usize, usize) = (4_621, 384, 1_928);
pub const TARGET_GRID: (u32, u32, u32) = (222, 1, 1);

const COMPUTE_MARKER: &str = r#"template <SgbTf32Op Op, int BM, int BN, int Stages, int MAtoms, int NAtoms>
__device__ __forceinline__ void gemm_bi_tf32_compute_stage("#;
const COMPUTE_OPEN_END: &str = "    float (&accumulators)[MAtoms][NAtoms][4]) {\n";
const COMPUTE_BODY_START: &str = r#"    const int k_offsets[4] = {0, 8, 16, 24};"#;
const COMPUTE_END: &str = "\n}\n\nstruct SgbTf32ThreadPlan {";

const K8_LOAD_HELPER: &str = r#"__device__ __forceinline__ void gemm_bi_tf32_nt_a_ldmatrix_prism_k8_load_a(
    SgbTf32Storage<SgbTf32Nt, 128, 64, 2>* storage, int stage,
    int warp_m, int k8, unsigned (&a_fragments)[2][4]) {
    int lane = (int)threadIdx.x & 31;
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
        int row = warp_m + m_atom * 16 + (lane & 15);
        int reduction = k8 + ((lane >> 4) << 2);
        unsigned address = (unsigned)__cvta_generic_to_shared(
            &gemm_bi_tf32_a_slot<SgbTf32Nt>(storage, stage, row, reduction));
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
}

"#;

const SPECIALIZED_BRANCH: &str = r#"    if constexpr (Op == SgbTf32Nt && BM == 128 && BN == 64
                  && Stages == 2 && MAtoms == 2 && NAtoms == 4) {
        unsigned a_fragments[2][2][4];
        gemm_bi_tf32_nt_a_ldmatrix_prism_k8_load_a(
            storage, stage, warp_m, 0, a_fragments[0]);
#pragma unroll
        for (int issue = 0; issue < 4; ++issue) {
            int k8 = issue * 8;
            int current_slot = issue & 1;
            if (issue + 1 < 4) {
                int next_slot = (issue + 1) & 1;
                gemm_bi_tf32_nt_a_ldmatrix_prism_k8_load_a(
                    storage, stage, warp_m, k8 + 8, a_fragments[next_slot]);
            }
#pragma unroll
            for (int n_atom = 0; n_atom < 4; ++n_atom) {
                int column = warp_n + n_atom * 8 + group;
                unsigned b_fragment[2];
                b_fragment[0] = gemm_bi_tf32_rna(
                    gemm_bi_tf32_b_slot<SgbTf32Nt>(
                        storage, stage, k8 + thread, column));
                b_fragment[1] = gemm_bi_tf32_rna(
                    gemm_bi_tf32_b_slot<SgbTf32Nt>(
                        storage, stage, k8 + thread + 4, column));
#pragma unroll
                for (int m_atom = 0; m_atom < 2; ++m_atom) {
                    gemm_bi_tf32_mma_m16n8k8(
                        accumulators[m_atom][n_atom],
                        a_fragments[current_slot][m_atom], b_fragment);
                }
            }
        }
    } else {
"#;

pub fn retained_source(production: &str, compact_layout: &str) -> Result<String, String> {
    composed_parent::retained_source(production, compact_layout)
}

pub fn candidate_source(production: &str, compact_layout: &str) -> Result<String, String> {
    let mut source = retained_source(production, compact_layout)?;
    insert_helper(&mut source)?;
    wrap_compute_body(&mut source)?;
    replace_exact_twice(&mut source, RETAINED_SYMBOL, SYMBOL, "candidate export")?;
    Ok(source)
}

pub fn restore_retained_source(candidate: &str) -> Result<String, String> {
    let mut source = candidate.to_owned();
    replace_exact_twice(&mut source, SYMBOL, RETAINED_SYMBOL, "restore export")?;
    unwrap_compute_body(&mut source)?;
    replace_exact(
        &mut source,
        &format!("{K8_LOAD_HELPER}{COMPUTE_MARKER}"),
        COMPUTE_MARKER,
        "remove helper",
    )?;
    Ok(source)
}

fn insert_helper(source: &mut String) -> Result<(), String> {
    replace_exact(
        source,
        COMPUTE_MARKER,
        &format!("{K8_LOAD_HELPER}{COMPUTE_MARKER}"),
        "helper insertion",
    )
}

fn compute_bounds(source: &str) -> Result<(usize, usize), String> {
    let marker = source
        .find(COMPUTE_MARKER)
        .ok_or("TF32 NT K8 compute marker missing")?;
    if source[marker + COMPUTE_MARKER.len()..].contains(COMPUTE_MARKER) {
        return Err("TF32 NT K8 compute marker duplicated".into());
    }
    let body = source[marker..]
        .find(COMPUTE_OPEN_END)
        .map(|offset| marker + offset + COMPUTE_OPEN_END.len())
        .ok_or("TF32 NT K8 compute signature end missing")?;
    let end = source[body..]
        .find(COMPUTE_END)
        .map(|offset| body + offset)
        .ok_or("TF32 NT K8 compute end missing")?;
    Ok((body, end))
}

fn wrap_compute_body(source: &mut String) -> Result<(), String> {
    let (body, end) = compute_bounds(source)?;
    let retained_body = &source[body..end];
    if !retained_body.starts_with(COMPUTE_BODY_START) {
        return Err("TF32 NT K8 retained compute body anchor changed".into());
    }
    if retained_body.contains(SPECIALIZED_BRANCH) {
        return Err("TF32 NT K8 compute body already wrapped".into());
    }
    let replacement = format!("{SPECIALIZED_BRANCH}{retained_body}\n    }}");
    source.replace_range(body..end, &replacement);
    Ok(())
}

fn unwrap_compute_body(source: &mut String) -> Result<(), String> {
    let (body, end) = compute_bounds(source)?;
    let wrapped = &source[body..end];
    if !wrapped.starts_with(SPECIALIZED_BRANCH) || !wrapped.ends_with("\n    }") {
        return Err("TF32 NT K8 wrapped compute seam changed".into());
    }
    let retained = &wrapped[SPECIALIZED_BRANCH.len()..wrapped.len() - "\n    }".len()];
    let retained = retained.to_owned();
    source.replace_range(body..end, &retained);
    Ok(())
}

fn replace_exact(source: &mut String, from: &str, to: &str, label: &str) -> Result<(), String> {
    let count = source.matches(from).count();
    if count != 1 {
        return Err(format!(
            "TF32 NT canonical Prism K8 {label} seam changed: expected 1, observed {count}"
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
            "TF32 NT canonical Prism K8 {label} seam changed: expected 2, observed {count}"
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
    fn k8_schedule_primes_and_consumes_the_original_order() {
        let mut slots = [None; 2];
        slots[0] = Some(0usize);
        let mut consumed = Vec::new();
        for issue in 0..4 {
            if issue + 1 < 4 {
                slots[(issue + 1) & 1] = Some(issue + 1);
            }
            assert_eq!(slots[issue & 1], Some(issue));
            consumed.push(slots[issue & 1].take().unwrap() * 8);
        }
        assert_eq!(consumed, [0, 8, 16, 24]);
    }

    #[test]
    fn candidate_is_reversible_and_avoids_the_full_domain_mechanism() {
        let retained = retained_source(PRODUCTION, COMPACT_LAYOUT).unwrap();
        let candidate = candidate_source(PRODUCTION, COMPACT_LAYOUT).unwrap();
        assert_eq!(restore_retained_source(&candidate).unwrap(), retained);
        assert_eq!(candidate.matches(SYMBOL).count(), 2);
        assert!(!candidate.contains(RETAINED_SYMBOL));
        assert_eq!(
            candidate
                .matches("gemm_bi_tf32_nt_a_ldmatrix_prism_k8_load_a(")
                .count(),
            3
        );
        assert!(candidate.contains("unsigned a_fragments[2][2][4];"));
        assert!(candidate.contains("unsigned b_fragment[2];"));
        assert!(!candidate.contains("canonical_prism"));
        assert!(!candidate.contains("gemm_bi_tf32_nt_prism_stage_full"));
    }

    #[test]
    fn missing_or_duplicated_compute_seams_are_rejected() {
        let retained = retained_source(PRODUCTION, COMPACT_LAYOUT).unwrap();
        assert!(
            candidate_source(
                &PRODUCTION.replacen(COMPUTE_BODY_START, "missing", 1),
                COMPACT_LAYOUT
            )
            .is_err()
        );
        let mut missing = retained.clone();
        missing = missing.replacen(COMPUTE_MARKER, "missing", 1);
        assert!(insert_helper(&mut missing).is_err());
        let mut duplicated = format!("{retained}\n{COMPUTE_MARKER}");
        assert!(insert_helper(&mut duplicated).is_err());
    }
}
