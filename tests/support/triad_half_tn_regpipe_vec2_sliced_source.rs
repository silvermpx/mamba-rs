#[path = "triad_half_tn_vec2_epilogue_source.rs"]
mod retained;

pub const SYMBOL_PREFIX: &str = "gemm_bi_tn_test_tc64_bk64_s2_regpipe_vec2_sliced_";
pub const RETAINED_SYMBOL_PREFIX: &str = retained::SYMBOL_PREFIX;
pub const BLOCK_THREADS: u32 = 128;
pub const STATIC_SHARED_BYTES: i32 = 32_768;
pub const REQUIRED_OCCUPANCY: u32 = 3;
pub const REGISTER_CAP: i32 = 128;
pub const TARGET: (usize, usize, usize) = (2_048, 768, 3_072);
pub const TARGET_GRID: u32 = 576;

pub fn retained_source(production: &str) -> Result<String, String> {
    retained::candidate_source(production)
}

pub fn candidate_source(production: &str) -> Result<String, String> {
    let mut source = retained_source(production)?;
    let sliced_state = encode_macro_block(SLICED_STATE);
    let sliced_regpipe_loop = encode_macro_block(SLICED_REGPIPE_LOOP);
    let sliced_commit = encode_macro_block(SLICED_COMMIT);
    replace_exact(
        &mut source,
        SCALAR_STAGE_MARKER,
        &format!("{SLICED_STAGE_MACRO}{SCALAR_STAGE_MARKER}"),
        1,
        "sliced-stage helper insertion",
    )?;
    replace_exact(
        &mut source,
        NEXT_TILE_MARKER,
        &format!("{sliced_state}{NEXT_TILE_MARKER}"),
        1,
        "next-tile state",
    )?;
    replace_exact(
        &mut source,
        RETAINED_REFILL,
        SLICED_REFILL_NOTE,
        1,
        "retained refill",
    )?;
    replace_exact(
        &mut source,
        REGPIPE_LOOP_LINE,
        &sliced_regpipe_loop,
        1,
        "regpipe slice issue",
    )?;
    replace_exact(
        &mut source,
        READ_BUFFER_ADVANCE,
        &format!("{sliced_commit}{READ_BUFFER_ADVANCE}"),
        1,
        "sliced commit",
    )?;
    replace_exact(
        &mut source,
        RETAINED_SYMBOL_PREFIX,
        SYMBOL_PREFIX,
        1,
        "candidate export",
    )?;
    Ok(source)
}

pub fn restore_retained_source(candidate: &str) -> Result<String, String> {
    let mut source = candidate.to_owned();
    let sliced_state = encode_macro_block(SLICED_STATE);
    let sliced_regpipe_loop = encode_macro_block(SLICED_REGPIPE_LOOP);
    let sliced_commit = encode_macro_block(SLICED_COMMIT);
    replace_exact(
        &mut source,
        SYMBOL_PREFIX,
        RETAINED_SYMBOL_PREFIX,
        1,
        "restore export",
    )?;
    replace_exact(
        &mut source,
        &format!("{sliced_commit}{READ_BUFFER_ADVANCE}"),
        READ_BUFFER_ADVANCE,
        1,
        "restore advance",
    )?;
    replace_exact(
        &mut source,
        &sliced_regpipe_loop,
        REGPIPE_LOOP_LINE,
        1,
        "restore regpipe loop",
    )?;
    replace_exact(
        &mut source,
        SLICED_REFILL_NOTE,
        RETAINED_REFILL,
        1,
        "restore refill",
    )?;
    replace_exact(
        &mut source,
        &format!("{sliced_state}{NEXT_TILE_MARKER}"),
        NEXT_TILE_MARKER,
        1,
        "restore next-tile state",
    )?;
    replace_exact(
        &mut source,
        &format!("{SLICED_STAGE_MACRO}{SCALAR_STAGE_MARKER}"),
        SCALAR_STAGE_MARKER,
        1,
        "restore sliced-stage helper",
    )?;
    Ok(source)
}

const SCALAR_STAGE_MARKER: &str = "#define GEMM_BI_TC64_STAGE_TN_SCALAR";
const NEXT_TILE_MARKER: &str = "        if (mt + 1 < num_m_tiles) {";
const RETAINED_REFILL: &str =
    "                GEMM_BI_TC64_STAGE_TN_ASYNC(read_buf ^ 1, (mt + 1) * GEMM_BI_TC64_BK);";
const SLICED_REFILL_NOTE: &str =
    "                /* the fast refill is issued in four slices beside the K16 groups */";
const REGPIPE_LOOP_LINE: &str = "        for (int ks = 0; ks < 4; ++ks) { \\\n";
const SLICED_REGPIPE_LOOP: &str = r#"        for (int ks = 0; ks < 4; ++ks) {
            if (sliced_refill) {
                GEMM_BI_TC64_STAGE_TN_ASYNC_SLICE(
                    sliced_write_buf, sliced_m_index, ks);
            }"#;
const READ_BUFFER_ADVANCE: &str = "        read_buf ^= 1;";
const SLICED_COMMIT: &str = r#"        if (sliced_refill) {
            asm volatile("cp.async.commit_group;\n");
        }
"#;
const SLICED_STATE: &str = r#"        bool sliced_refill = fast_stage && mt + 1 < num_m_tiles;
        int sliced_write_buf = read_buf ^ 1;
        int sliced_m_index = (mt + 1) * GEMM_BI_TC64_BK;
"#;

const SLICED_STAGE_MACRO: &str = r#"#define GEMM_BI_TC64_STAGE_TN_ASYNC_SLICE(buf, mIdx, slice)                    \
    do {                                                                      \
        int _i = threadIdx.x + (slice) * GEMM_BI_TC64_THREADS;                \
        int _r = _i / (GEMM_BI_TC64_BM / 8);                                 \
        int _c = (_i % (GEMM_BI_TC64_BM / 8)) * 8;                           \
        int _gm = (mIdx) + _r;                                                \
        {                                                                     \
            unsigned _xs = Xs_sbase +                                         \
                (unsigned)((buf) * GEMM_BI_TC64_BK * GEMM_BI_TC64_LDB * 2);   \
            int _gk = pid_m * GEMM_BI_TC64_BM + _c;                           \
            int _elems =                                                      \
                gemm_bi_cp_async_valid_elems(_gm < M_red, K_out, _gk);        \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _xs +                                             \
                (unsigned)(GEMM_BI_HALF_TN_INDEX(_r, _c) * 2);                \
            long long _offset =                                               \
                _bytes == 0 ? 0 : (long long)_gm * K_out + _gk;               \
            const void* _src = gemm_bi_cp_async_source(A, _offset, _bytes);   \
            gemm_bi_cp_async_16_zfill(_dst, _src, _bytes);                    \
        }                                                                     \
        {                                                                     \
            unsigned _ys = Ys_sbase +                                         \
                (unsigned)((buf) * GEMM_BI_TC64_BK * GEMM_BI_TC64_LDB * 2);   \
            int _gn = pid_n * GEMM_BI_TC64_BN + _c;                           \
            int _elems =                                                      \
                gemm_bi_cp_async_valid_elems(_gm < M_red, N, _gn);            \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _ys +                                             \
                (unsigned)(GEMM_BI_HALF_TN_INDEX(_r, _c) * 2);                \
            long long _offset =                                               \
                _bytes == 0 ? 0 : (long long)_gm * N + _gn;                   \
            const void* _src = gemm_bi_cp_async_source(B, _offset, _bytes);   \
            gemm_bi_cp_async_16_zfill(_dst, _src, _bytes);                    \
        }                                                                     \
    } while (0)

"#;

fn replace_exact(
    source: &mut String,
    before: &str,
    after: &str,
    expected: usize,
    label: &str,
) -> Result<(), String> {
    let actual = source.matches(before).count();
    if actual != expected {
        return Err(format!(
            "half TN sliced {label} seam expected {expected}, observed {actual}"
        ));
    }
    *source = source.replacen(before, after, expected);
    Ok(())
}

fn encode_macro_block(decoded: &str) -> String {
    decoded.lines().map(|line| format!("{line} \\\n")).collect()
}

pub const fn copy_coordinate(thread: usize, slice: usize) -> (usize, usize) {
    let linear = thread + slice * BLOCK_THREADS as usize;
    (linear / 8, (linear % 8) * 8)
}

pub fn all_strata_below(strata: &[[f64; 2]], threshold: f64) -> bool {
    strata.len() == 4
        && threshold.is_finite()
        && threshold > 0.0
        && strata.iter().all(|[p50, p95]| {
            p50.is_finite()
                && p95.is_finite()
                && *p50 > 0.0
                && *p95 > 0.0
                && *p50 < threshold
                && *p95 < threshold
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_triad/sm80.cu");

    #[test]
    fn four_slices_cover_each_retained_vector_copy_once() {
        let mut owners = [0_u8; 64 * 8];
        for slice in 0..4 {
            for thread in 0..128 {
                let (row, vector) = copy_coordinate(thread, slice);
                assert!(row < 64);
                assert!(vector < 64);
                assert_eq!(vector % 8, 0);
                owners[row * 8 + vector / 8] += 1;
            }
        }
        assert!(owners.into_iter().all(|count| count == 1));
    }

    #[test]
    fn candidate_interleaves_four_slices_without_changing_mma_or_epilogue() {
        let retained = retained_source(PRODUCTION).unwrap();
        let candidate = candidate_source(PRODUCTION).unwrap();
        assert_eq!(
            candidate
                .matches("GEMM_BI_TC64_STAGE_TN_ASYNC_SLICE(")
                .count(),
            2
        );
        assert_eq!(candidate.matches("if (sliced_refill)").count(), 2);
        assert!(candidate.contains("int _i = threadIdx.x + (slice) * GEMM_BI_TC64_THREADS;"));
        for line in [
            "        bool sliced_refill = fast_stage && mt + 1 < num_m_tiles;",
            "        for (int ks = 0; ks < 4; ++ks) {",
            "            if (sliced_refill) {",
            "                GEMM_BI_TC64_STAGE_TN_ASYNC_SLICE(",
            "            asm volatile(\"cp.async.commit_group;\\n\");",
        ] {
            assert!(
                candidate.contains(&format!("{line} \\\n")),
                "macro line lost continuation: {line}"
            );
        }
        assert!(candidate.contains("for (int ks = 0; ks < 4; ++ks)"));
        assert!(candidate.contains("a_frag[(ks + 1) & 1][fm]"));
        assert!(candidate.contains("a_frag[ks & 1][fm]"));
        assert!(candidate.contains("gemm_bi_accumulate_float2_or_scalar("));
        assert_eq!(
            candidate
                .matches("mma.sync.aligned.m16n8k16.row.col.f32.")
                .count(),
            1
        );
        assert_eq!(
            candidate.matches("cp.async.wait_group 0;").count(),
            retained.matches("cp.async.wait_group 0;").count()
        );
        assert_eq!(
            candidate.matches("__syncthreads();").count(),
            retained.matches("__syncthreads();").count()
        );
        assert_eq!(candidate.matches(SYMBOL_PREFIX).count(), 1);
        assert!(!candidate.contains("void gemm_bi_tn_test_tc64_bk64_s2_regpipe_vec2_##SUFFIX"));
        assert_eq!(restore_retained_source(&candidate).unwrap(), retained);
    }

    #[test]
    fn d768_in_geometry_resources_and_screen_policy_are_frozen() {
        assert_eq!(TARGET, (2_048, 768, 3_072));
        assert_eq!(TARGET_GRID, 576);
        assert_eq!(BLOCK_THREADS, 128);
        assert_eq!(STATIC_SHARED_BYTES, 32_768);
        assert_eq!(REQUIRED_OCCUPANCY, 3);
        assert_eq!(REGISTER_CAP, 128);
        assert!(all_strata_below(&[[0.970, 0.980]; 4], 0.985));
        assert!(!all_strata_below(&[[0.970, 0.985]; 4], 0.985));
    }
}
