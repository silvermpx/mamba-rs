pub const SYMBOL: &str = "gemm_bi_tn_test_single_barrier_sm80_mma_tf32_v1_m128n64_bk32_s3";

const OLD_SYMBOL: &str = "gemm_bi_tn_sm80_mma_tf32_v1_m128n64_bk32_s3";
const OLD_BOTTOM: &str = concat!(
    "        }\n",
    "        __syncthreads();\n",
    "    }\n",
    "}\n\n",
    "template <SgbTf32Op Op, int BM, int BN, int Stages>\n",
    "__device__ __forceinline__ void gemm_bi_tf32_kernel("
);
const NEW_BOTTOM: &str = concat!(
    "        }\n",
    "        if constexpr (!(Op == SgbTf32Tn && BM == 128 && BN == 64 && Stages == 3)) {\n",
    "            __syncthreads();\n",
    "        }\n",
    "    }\n",
    "}\n\n",
    "template <SgbTf32Op Op, int BM, int BN, int Stages>\n",
    "__device__ __forceinline__ void gemm_bi_tf32_kernel("
);

pub fn candidate_source(production: &str) -> Result<String, String> {
    let mut source = production.to_owned();
    replace_exact(
        &mut source,
        OLD_BOTTOM,
        NEW_BOTTOM,
        1,
        "async-mainloop bottom barrier",
    )?;
    replace_exact(&mut source, OLD_SYMBOL, SYMBOL, 2, "TN target symbol")?;
    Ok(source)
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

#[cfg(test)]
mod tests {
    use super::*;

    const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_triad/sm80.cu");
    #[test]
    fn next_iteration_top_barrier_fences_every_reused_s3_slot() {
        let stages = 3usize;
        for tile_count in 0..65 {
            let mut last_compute_iteration = [None; 3];
            for tile in 0..tile_count {
                // All threads reach this barrier only after the prior
                // iteration's compute. A refill after it may therefore reuse
                // any prior slot.
                let next = tile + stages - 1;
                if next < tile_count {
                    let write_slot = next % stages;
                    assert_ne!(write_slot, tile % stages);
                    if let Some(last_compute) = last_compute_iteration[write_slot] {
                        assert_eq!(last_compute + 1, tile);
                    }
                }
                last_compute_iteration[tile % stages] = Some(tile);
            }
        }
    }

    #[test]
    fn source_targets_only_tn_128x64_s3_bottom_barrier_and_export() {
        let source = candidate_source(PRODUCTION).unwrap();
        assert_eq!(source.matches(NEW_BOTTOM).count(), 1);
        assert_eq!(source.matches(SYMBOL).count(), 2);
        assert_eq!(source.matches(OLD_SYMBOL).count(), 0);
        assert!(source.contains(
            "cp.async.wait_group %0;\\n\" :: \"n\"(Stages - 2));\n        __syncthreads();"
        ));

        let restored = source
            .replacen(NEW_BOTTOM, OLD_BOTTOM, 1)
            .replace(SYMBOL, OLD_SYMBOL);
        assert_eq!(restored, PRODUCTION);
    }

    #[test]
    fn changed_or_duplicated_schedule_and_symbol_anchors_are_rejected() {
        assert!(candidate_source(&PRODUCTION.replacen(OLD_BOTTOM, "", 1)).is_err());
        assert!(candidate_source(&format!("{PRODUCTION}\n{OLD_BOTTOM}")).is_err());
        assert!(candidate_source(&PRODUCTION.replacen(OLD_SYMBOL, "missing", 1)).is_err());
        assert!(candidate_source(&format!("{PRODUCTION}\n{OLD_SYMBOL}")).is_err());
    }
}
