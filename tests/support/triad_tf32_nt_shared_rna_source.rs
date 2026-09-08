pub const SYMBOL: &str = "gemm_bi_nt_test_shared_rna_sm80_mma_tf32_v1_m128n64_bk32_s2";

const PARENT_SYMBOL: &str = "gemm_bi_nt_test_compact_eight_warp_sm80_mma_tf32_v1_m128n64_bk32_s2";
pub const EXPECTED_PARENT_FNV64: u64 = 0x2ded_0f1a_133e_5949;

const ASYNC_MAINLOOP_MARKER: &str = concat!(
    "template <SgbTf32Op Op, int BM, int BN, int Stages,\n",
    "          int MAtoms, int NAtoms, bool NarrowA, bool NarrowB>\n",
    "__device__ __forceinline__ void gemm_bi_tf32_async_mainloop("
);
const KERNEL_MARKER: &str = concat!(
    "template <SgbTf32Op Op, int BM, int BN, int Stages>\n",
    "__device__ __forceinline__ void gemm_bi_tf32_kernel("
);
const WAIT_AND_PUBLISH: &str = concat!(
    "        asm volatile(\"cp.async.wait_group %0;\\n\" :: \"n\"(Stages - 2));\n",
    "        __syncthreads();"
);
const COMPUTE_CALL: &str = concat!(
    "            gemm_bi_tf32_compute_stage<Op, BM, BN, Stages, MAtoms, NAtoms>(\n",
    "                storage, static_cast<int>(tile % Stages),\n",
    "                thread_plan.warp_m, thread_plan.warp_n,\n",
    "                thread_plan.group, thread_plan.thread, accumulators);"
);

const SHARED_RNA_HELPERS: &str = r#"
template <int Stages>
__device__ __forceinline__ void gemm_bi_tf32_nt_test_shared_rna_stage(
    SgbTf32Storage<SgbTf32Nt, 128, 64, Stages>* storage, int stage) {
    static_assert(Stages == 2, "shared-stage RNA is scoped to S2");
    constexpr int Threads = 256;
#pragma unroll
    for (int linear = (int)threadIdx.x; linear < 128 * 8; linear += Threads) {
        int row = linear >> 3;
        int reduction = (linear & 7) * 4;
        float* slot = &gemm_bi_tf32_a_slot<SgbTf32Nt>(storage, stage, row, reduction);
        uint4 raw = *reinterpret_cast<uint4*>(slot);
        raw.x = gemm_bi_tf32_rna(__uint_as_float(raw.x));
        raw.y = gemm_bi_tf32_rna(__uint_as_float(raw.y));
        raw.z = gemm_bi_tf32_rna(__uint_as_float(raw.z));
        raw.w = gemm_bi_tf32_rna(__uint_as_float(raw.w));
        *reinterpret_cast<uint4*>(slot) = raw;
    }
#pragma unroll
    for (int linear = (int)threadIdx.x; linear < 64 * 8; linear += Threads) {
        int column = linear >> 3;
        int reduction = (linear & 7) * 4;
        float* slot = &gemm_bi_tf32_b_slot<SgbTf32Nt>(storage, stage, reduction, column);
        uint4 raw = *reinterpret_cast<uint4*>(slot);
        raw.x = gemm_bi_tf32_rna(__uint_as_float(raw.x));
        raw.y = gemm_bi_tf32_rna(__uint_as_float(raw.y));
        raw.z = gemm_bi_tf32_rna(__uint_as_float(raw.z));
        raw.w = gemm_bi_tf32_rna(__uint_as_float(raw.w));
        *reinterpret_cast<uint4*>(slot) = raw;
    }
}

template <int Stages, int MAtoms, int NAtoms>
__device__ __forceinline__ void gemm_bi_tf32_nt_test_shared_rna_compute_stage(
    SgbTf32Storage<SgbTf32Nt, 128, 64, Stages>* storage, int stage,
    int warp_m, int warp_n, int group, int thread,
    float (&accumulators)[MAtoms][NAtoms][4]) {
    static_assert(Stages == 2 && MAtoms == 2 && NAtoms == 4,
                  "shared-stage RNA is scoped to compact8 S2");
    const int k_offsets[4] = {0, 8, 16, 24};
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        int k8 = k_offsets[issue];
        unsigned a_fragments[MAtoms][4];
        unsigned b_fragments[NAtoms][2];
#pragma unroll
        for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
            int row = warp_m + m_atom * 16 + group;
            a_fragments[m_atom][0] = __float_as_uint(
                gemm_bi_tf32_a_slot<SgbTf32Nt>(storage, stage, row, k8 + thread));
            a_fragments[m_atom][1] = __float_as_uint(
                gemm_bi_tf32_a_slot<SgbTf32Nt>(storage, stage, row + 8, k8 + thread));
            a_fragments[m_atom][2] = __float_as_uint(
                gemm_bi_tf32_a_slot<SgbTf32Nt>(storage, stage, row, k8 + thread + 4));
            a_fragments[m_atom][3] = __float_as_uint(
                gemm_bi_tf32_a_slot<SgbTf32Nt>(storage, stage, row + 8, k8 + thread + 4));
        }
#pragma unroll
        for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
            int column = warp_n + n_atom * 8 + group;
            b_fragments[n_atom][0] = __float_as_uint(
                gemm_bi_tf32_b_slot<SgbTf32Nt>(storage, stage, k8 + thread, column));
            b_fragments[n_atom][1] = __float_as_uint(
                gemm_bi_tf32_b_slot<SgbTf32Nt>(storage, stage, k8 + thread + 4, column));
        }
#pragma unroll
        for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
                gemm_bi_tf32_mma_m16n8k8(
                    accumulators[m_atom][n_atom],
                    a_fragments[m_atom], b_fragments[n_atom]);
            }
        }
    }
}
"#;

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
            "{label}: expected {expected} source anchors, found {count}"
        ));
    }
    *source = source.replacen(from, to, expected);
    Ok(())
}

fn insert_before_exact(
    source: &mut String,
    marker: &str,
    insertion: &str,
    label: &str,
) -> Result<(), String> {
    replace_exact(source, marker, &format!("{insertion}\n{marker}"), 1, label)
}

fn transform_async_mainloop(source: &mut String) -> Result<(), String> {
    let start = source
        .find(ASYNC_MAINLOOP_MARKER)
        .ok_or("shared-stage RNA async mainloop marker is missing")?;
    if source[start + ASYNC_MAINLOOP_MARKER.len()..].contains(ASYNC_MAINLOOP_MARKER) {
        return Err("shared-stage RNA async mainloop marker is duplicated".into());
    }
    let relative_end = source[start..]
        .find(KERNEL_MARKER)
        .ok_or("shared-stage RNA kernel marker is missing after async mainloop")?;
    let end = start + relative_end;
    let mut mainloop = source[start..end].to_owned();
    replace_exact(
        &mut mainloop,
        WAIT_AND_PUBLISH,
        concat!(
            "        if constexpr (Op == SgbTf32Nt && BM == 128 && BN == 64 && Stages == 2) {\n",
            "            asm volatile(\"cp.async.wait_group %0;\\n\" :: \"n\"(Stages - 2) : \"memory\");\n",
            "            gemm_bi_tf32_nt_test_shared_rna_stage<Stages>(\n",
            "                reinterpret_cast<SgbTf32Storage<SgbTf32Nt, 128, 64, Stages>*>(storage),\n",
            "                static_cast<int>(tile % Stages));\n",
            "        } else {\n",
            "            asm volatile(\"cp.async.wait_group %0;\\n\" :: \"n\"(Stages - 2));\n",
            "        }\n",
            "        __syncthreads();"
        ),
        1,
        "shared-stage RNA wait and publication boundary",
    )?;
    replace_exact(
        &mut mainloop,
        COMPUTE_CALL,
        concat!(
            "            if constexpr (Op == SgbTf32Nt && BM == 128 && BN == 64 && Stages == 2) {\n",
            "                gemm_bi_tf32_nt_test_shared_rna_compute_stage<Stages, MAtoms, NAtoms>(\n",
            "                    reinterpret_cast<SgbTf32Storage<SgbTf32Nt, 128, 64, Stages>*>(storage),\n",
            "                    static_cast<int>(tile % Stages),\n",
            "                    thread_plan.warp_m, thread_plan.warp_n,\n",
            "                    thread_plan.group, thread_plan.thread, accumulators);\n",
            "            } else {\n",
            "                gemm_bi_tf32_compute_stage<Op, BM, BN, Stages, MAtoms, NAtoms>(\n",
            "                    storage, static_cast<int>(tile % Stages),\n",
            "                    thread_plan.warp_m, thread_plan.warp_n,\n",
            "                    thread_plan.group, thread_plan.thread, accumulators);\n",
            "            }"
        ),
        1,
        "shared-stage RNA async compute selection",
    )?;
    source.replace_range(start..end, &mainloop);
    Ok(())
}

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

fn candidate_source_with_digest(
    compact_parent: &str,
    expected_parent_fnv64: u64,
) -> Result<String, String> {
    let observed = fnv1a64(compact_parent.as_bytes());
    if observed != expected_parent_fnv64 {
        return Err(format!(
            "compact parent digest changed: expected {expected_parent_fnv64:#018x}, observed {observed:#018x}"
        ));
    }
    let mut source = compact_parent.to_owned();
    transform_async_mainloop(&mut source)?;
    insert_before_exact(
        &mut source,
        ASYNC_MAINLOOP_MARKER,
        SHARED_RNA_HELPERS,
        "shared-stage RNA helper insertion",
    )?;
    replace_exact(
        &mut source,
        PARENT_SYMBOL,
        SYMBOL,
        2,
        "shared-stage RNA target symbol and signature",
    )?;
    Ok(source)
}

pub fn candidate_source(compact_parent: &str) -> Result<String, String> {
    candidate_source_with_digest(compact_parent, EXPECTED_PARENT_FNV64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn compact_index(row: usize, reduction: usize) -> usize {
        row * 32 + (reduction ^ ((row & 7) << 2))
    }

    fn producer_words(rows: usize) -> BTreeSet<usize> {
        let mut words = BTreeSet::new();
        for thread in 0..256 {
            let mut linear = thread;
            while linear < rows * 8 {
                let row = linear >> 3;
                let reduction = (linear & 7) * 4;
                assert_eq!(compact_index(row, reduction) % 4, 0);
                for word in 0..4 {
                    assert!(words.insert(compact_index(row, reduction + word)));
                }
                linear += 256;
            }
        }
        words
    }

    fn synthetic_parent() -> String {
        format!(
            "prefix\n{ASYNC_MAINLOOP_MARKER} body {{\n{WAIT_AND_PUBLISH}\nif (thread_plan.compute) {{\n{COMPUTE_CALL}\n}}\n}}\n{KERNEL_MARKER} kernel body\nGEMM_BI_TF32_DEFINE_KERNEL({PARENT_SYMBOL}, SgbTf32Nt, 128, 64, 2, 256, 1)\nTF32_ASSERT_KERNEL_SIGNATURE({PARENT_SYMBOL});\nscalar fallback gemm_bi_tf32_compute_stage<Op, BM, BN, Stages, MAtoms, NAtoms>(storage);\n"
        )
    }

    #[test]
    fn async_copy_owners_cover_each_compact_stage_word_once() {
        let a = producer_words(128);
        let b = producer_words(64);
        assert_eq!(a.len(), 128 * 32);
        assert_eq!(b.len(), 64 * 32);
        assert_eq!(a, (0..128 * 32).collect());
        assert_eq!(b, (0..64 * 32).collect());
    }

    #[test]
    fn all_eight_warp_consumers_read_producer_owned_words() {
        let a = producer_words(128);
        let b = producer_words(64);
        for warp in 0..8 {
            let warp_m = (warp >> 1) * 32;
            let warp_n = (warp & 1) * 32;
            for lane in 0..32 {
                let group = lane >> 2;
                let thread = lane & 3;
                for k8 in [0, 8, 16, 24] {
                    for m_atom in 0..2 {
                        let row = warp_m + m_atom * 16 + group;
                        for (r, k) in [
                            (row, k8 + thread),
                            (row + 8, k8 + thread),
                            (row, k8 + thread + 4),
                            (row + 8, k8 + thread + 4),
                        ] {
                            assert!(a.contains(&compact_index(r, k)));
                        }
                    }
                    for n_atom in 0..4 {
                        let column = warp_n + n_atom * 8 + group;
                        assert!(b.contains(&compact_index(column, k8 + thread)));
                        assert!(b.contains(&compact_index(column, k8 + thread + 4)));
                    }
                }
            }
        }
    }

    #[test]
    fn s2_ring_wait_convert_publish_compute_reuse_is_ordered() {
        let tile_count = 7;
        let mut stage_tile = [None; 2];
        for tile in 0..1 {
            stage_tile[tile] = Some(tile);
        }
        let mut trace = Vec::new();
        for tile in 0..tile_count {
            let stage = tile % 2;
            assert_eq!(stage_tile[stage], Some(tile));
            trace.push((tile, stage, "wait"));
            trace.push((tile, stage, "convert"));
            trace.push((tile, stage, "publish"));
            trace.push((tile, stage, "compute"));
            let next = tile + 1;
            if next < tile_count {
                let next_stage = next % 2;
                if next_stage == stage {
                    panic!("S2 attempted to refill the stage being computed");
                }
                stage_tile[next_stage] = Some(next);
                trace.push((next, next_stage, "refill"));
            }
            stage_tile[stage] = None;
        }
        for tile in 0..tile_count {
            assert_eq!(
                trace
                    .iter()
                    .filter(|(trace_tile, _, action)| {
                        *trace_tile == tile && *action == "convert"
                    })
                    .count(),
                1
            );
        }
    }

    #[test]
    fn source_adapter_targets_only_async_compact8_s2() {
        let parent = synthetic_parent();
        let candidate = candidate_source_with_digest(&parent, fnv1a64(parent.as_bytes())).unwrap();
        assert_eq!(candidate.matches(SYMBOL).count(), 2);
        assert!(!candidate.contains(PARENT_SYMBOL));
        assert!(candidate.contains(": \"memory\");"));
        let wait = candidate.find("cp.async.wait_group %0;").unwrap();
        let conversion = candidate
            .find("gemm_bi_tf32_nt_test_shared_rna_stage<Stages>")
            .unwrap();
        let publish = candidate[conversion..].find("__syncthreads();").unwrap() + conversion;
        assert!(wait < conversion && conversion < publish);
        assert!(candidate.contains("const int k_offsets[4] = {0, 8, 16, 24};"));
        assert!(candidate.contains("__float_as_uint("));
        assert!(candidate.contains("gemm_bi_tf32_mma_m16n8k8("));
        assert!(candidate.contains(
            "scalar fallback gemm_bi_tf32_compute_stage<Op, BM, BN, Stages, MAtoms, NAtoms>"
        ));
    }

    #[test]
    fn source_adapter_fails_closed_on_changed_seams() {
        let parent = synthetic_parent();
        for changed in [
            parent.replacen(ASYNC_MAINLOOP_MARKER, "changed async mainloop", 1),
            parent.replacen(WAIT_AND_PUBLISH, "changed wait", 1),
            parent.replacen(COMPUTE_CALL, "changed compute", 1),
            parent.replacen(PARENT_SYMBOL, "changed target", 1),
        ] {
            assert!(candidate_source_with_digest(&changed, fnv1a64(changed.as_bytes())).is_err());
        }
    }
}
