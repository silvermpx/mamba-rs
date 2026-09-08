#[path = "triad_half_tn_vec2_epilogue_source.rs"]
mod retained;

pub const SYMBOL_PREFIX: &str = "gemm_bi_tn_test_tc64_bk64_s2_regpipe_vec2_ws5_";
pub const THREADS: u32 = 160;
pub const EXPECTED_STATIC_SHARED: i32 = 32_800;

pub fn logical_consumer(thread: u32) -> Option<(u32, u32)> {
    if !(32..THREADS).contains(&thread) {
        return None;
    }
    Some((thread / 32 - 1, thread % 32))
}

pub fn candidate_source(production: &str) -> Result<String, String> {
    let mut source = retained::candidate_source(production)?;
    replace_exact(
        &mut source,
        "#define GEMM_BI_TC64_THREADS 128",
        "#define GEMM_BI_TC64_THREADS 160",
    )?;
    replace_exact(&mut source, retained::SYMBOL_PREFIX, SYMBOL_PREFIX)?;

    replace_section(
        &mut source,
        "#define GEMM_BI_TC64_STAGE_TN_ASYNC",
        "#define GEMM_BI_TC64_STAGE_TN_SCALAR",
        PRODUCER_STAGE,
    )?;
    replace_exact(&mut source, SHARED_ARRAYS, SHARED_ARRAYS_WITH_PIPELINE)?;
    replace_exact(&mut source, WARP_MAPPING, PRODUCER_CONSUMER_MAPPING)?;

    let loop_start = unique_offset(&source, NUM_TILES)?;
    let epilogue_start = unique_offset(&source, EPILOGUE)?;
    if loop_start >= epilogue_start {
        return Err("half TN warp-specialized loop boundaries reversed".into());
    }
    let old_loop = &source[loop_start..epilogue_start];
    let compute_start = unique_offset(old_loop, COMPUTE_START)?;
    let compute_end = unique_offset(old_loop, COMPUTE_END)?;
    if compute_start >= compute_end {
        return Err("half TN warp-specialized compute boundaries reversed".into());
    }
    let compute = &old_loop[compute_start..compute_end];
    let new_loop = format!("{PIPELINE_PROLOGUE}{compute}{PIPELINE_EPILOGUE}");
    source.replace_range(loop_start..epilogue_start, &new_loop);

    Ok(format!("#include <cuda_awbarrier_primitives.h>\n{source}"))
}

const PRODUCER_STAGE: &str = r#"#define GEMM_BI_TC64_STAGE_TN_ASYNC(buf, mIdx, TT)                            \
    do {                                                                      \
        TT* _xs = &Xs[(buf)][0][0];                                           \
        TT* _ys = &Ys[(buf)][0][0];                                           \
        for (int _i = lane;                                                   \
             _i < GEMM_BI_TC64_BK * (GEMM_BI_TC64_BM / 8); _i += 32) {       \
            int _r = _i / (GEMM_BI_TC64_BM / 8);                              \
            int _c = (_i % (GEMM_BI_TC64_BM / 8)) * 8;                        \
            int _gm = (mIdx) + _r;                                            \
            int _gk = pid_m * GEMM_BI_TC64_BM + _c;                           \
            int _elems = gemm_bi_cp_async_valid_elems(_gm < M_red, K_out, _gk); \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = (unsigned)__cvta_generic_to_shared(              \
                _xs + GEMM_BI_HALF_TN_INDEX(_r, _c));                         \
            long long _offset =                                               \
                _bytes == 0 ? 0 : (long long)_gm * K_out + _gk;               \
            const void* _src = gemm_bi_cp_async_source(A, _offset, _bytes);    \
            gemm_bi_cp_async_16_zfill_l2(_dst, _src, _bytes);                 \
        }                                                                     \
        for (int _i = lane;                                                   \
             _i < GEMM_BI_TC64_BK * (GEMM_BI_TC64_BN / 8); _i += 32) {       \
            int _r = _i / (GEMM_BI_TC64_BN / 8);                              \
            int _c = (_i % (GEMM_BI_TC64_BN / 8)) * 8;                        \
            int _gm = (mIdx) + _r;                                            \
            int _gn = pid_n * GEMM_BI_TC64_BN + _c;                           \
            int _elems = gemm_bi_cp_async_valid_elems(_gm < M_red, N, _gn);   \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = (unsigned)__cvta_generic_to_shared(              \
                _ys + GEMM_BI_HALF_TN_INDEX(_r, _c));                         \
            long long _offset = _bytes == 0 ? 0 : (long long)_gm * N + _gn;   \
            const void* _src = gemm_bi_cp_async_source(B, _offset, _bytes);    \
            gemm_bi_cp_async_16_zfill_l2(_dst, _src, _bytes);                 \
        }                                                                     \
        unsigned _filled = (unsigned)__cvta_generic_to_shared(&filled[buf]);  \
        asm volatile("cp.async.mbarrier.arrive.shared.b64 [%0];\n"            \
                     :: "r"(_filled) : "memory");                            \
        __mbarrier_arrive(&filled[buf]);                                      \
    } while (0)

"#;

const SHARED_ARRAYS: &str = r#"    __shared__ __align__(16) T_ACT Xs[2][GEMM_BI_TC64_BK][GEMM_BI_TC64_LDB];           \
    __shared__ __align__(16) T_ACT Ys[2][GEMM_BI_TC64_BK][GEMM_BI_TC64_LDB];           \"#;

const SHARED_ARRAYS_WITH_PIPELINE: &str = r#"    __shared__ __align__(16) T_ACT Xs[2][GEMM_BI_TC64_BK][GEMM_BI_TC64_LDB];           \
    __shared__ __align__(16) T_ACT Ys[2][GEMM_BI_TC64_BK][GEMM_BI_TC64_LDB];           \
    __shared__ __align__(8) __mbarrier_t pipeline_barriers[4];                 \
    __mbarrier_t* ready = &pipeline_barriers[0];                               \
    __mbarrier_t* filled = &pipeline_barriers[2];                              \"#;

const WARP_MAPPING: &str = r#"    int warp = threadIdx.x / 32;                                               \
    int lane = threadIdx.x % 32;                                               \
    int warpM = (warp / 2) * 32;                                               \
    int warpN = (warp % 2) * 32;                                               \"#;

const PRODUCER_CONSUMER_MAPPING: &str = r#"    int physical_warp = threadIdx.x / 32;                                      \
    int lane = threadIdx.x % 32;                                               \
    bool producer = physical_warp == 0;                                        \
    int warp = physical_warp - 1;                                              \
    int warpM = (warp / 2) * 32;                                               \
    int warpN = (warp % 2) * 32;                                               \"#;

const NUM_TILES: &str = "    int num_m_tiles = (M_red + GEMM_BI_TC64_BK - 1) / GEMM_BI_TC64_BK;";
const COMPUTE_START: &str = "        unsigned Xs_rd =";
const COMPUTE_END: &str = "        read_buf ^= 1;";
const EPILOGUE: &str = "    /* epilogue: paired f32 accumulate into dW, scalar at the N tail */";

const PIPELINE_PROLOGUE: &str = r#"    int num_m_tiles = (M_red + GEMM_BI_TC64_BK - 1) / GEMM_BI_TC64_BK;     \
    if (!fast_stage || num_m_tiles == 0) return;                               \
    if (threadIdx.x < 4)                                                       \
        __mbarrier_init(&pipeline_barriers[threadIdx.x], GEMM_BI_TC64_THREADS); \
    __syncthreads();                                                           \
    if (!producer) {                                                           \
        __mbarrier_arrive(&ready[0]);                                          \
        __mbarrier_arrive(&ready[1]);                                          \
    }                                                                          \
    if (producer) {                                                            \
        for (int mt = 0; mt < num_m_tiles; ++mt) {                             \
            int write_buf = mt & 1;                                            \
            __mbarrier_token_t token = __mbarrier_arrive(&ready[write_buf]);   \
            while (!__mbarrier_test_wait(&ready[write_buf], token)) {}         \
            GEMM_BI_TC64_STAGE_TN_ASYNC(                                      \
                write_buf, mt * GEMM_BI_TC64_BK, T_ACT);                      \
        }                                                                      \
    } else {                                                                   \
        for (int mt = 0; mt < num_m_tiles; ++mt) {                             \
            int read_buf = mt & 1;                                             \
            __mbarrier_token_t token = __mbarrier_arrive(&filled[read_buf]);   \
            while (!__mbarrier_test_wait(&filled[read_buf], token)) {}         \
"#;

const PIPELINE_EPILOGUE: &str = r#"            __mbarrier_arrive(&ready[read_buf]);                              \
        }                                                                      \
    }                                                                          \
    if (producer) return;                                                      \
"#;

fn replace_exact(source: &mut String, before: &str, after: &str) -> Result<(), String> {
    let actual = source.matches(before).count();
    if actual != 1 {
        return Err(format!(
            "half TN warp-specialized anchor {before:?}: expected 1, observed {actual}"
        ));
    }
    *source = source.replacen(before, after, 1);
    Ok(())
}

fn replace_section(
    source: &mut String,
    start: &str,
    end: &str,
    replacement: &str,
) -> Result<(), String> {
    let start = unique_offset(source, start)?;
    let end = unique_offset(source, end)?;
    if start >= end {
        return Err("half TN warp-specialized section boundaries reversed".into());
    }
    source.replace_range(start..end, replacement);
    Ok(())
}

fn unique_offset(source: &str, anchor: &str) -> Result<usize, String> {
    let mut matches = source.match_indices(anchor);
    let offset = matches
        .next()
        .map(|(offset, _)| offset)
        .ok_or_else(|| format!("half TN warp-specialized anchor missing: {anchor:?}"))?;
    if matches.next().is_some() {
        return Err(format!(
            "half TN warp-specialized anchor duplicated: {anchor:?}"
        ));
    }
    Ok(offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_triad/sm80.cu");

    #[test]
    fn first_warp_is_producer_and_remaining_warps_keep_original_lane_ownership() {
        for thread in 0..32 {
            assert_eq!(logical_consumer(thread), None);
        }
        let mut owners = [[0u8; 32]; 4];
        for thread in 32..THREADS {
            let (warp, lane) = logical_consumer(thread).expect("consumer thread");
            owners[warp as usize][lane as usize] += 1;
        }
        assert!(owners.into_iter().flatten().all(|count| count == 1));
        assert_eq!(logical_consumer(THREADS), None);
    }

    #[test]
    fn producer_lane_stride_covers_every_vector_once_for_each_operand() {
        const VECTORS_PER_OPERAND: usize = 64 * (64 / 8);
        let mut owners = [0u8; VECTORS_PER_OPERAND];
        for lane in 0..32 {
            for vector in (lane..VECTORS_PER_OPERAND).step_by(32) {
                owners[vector] += 1;
            }
        }
        assert!(owners.into_iter().all(|count| count == 1));
        assert_eq!(VECTORS_PER_OPERAND / 32, 16);
    }

    #[test]
    fn source_uses_a_two_stage_partitioned_pipeline_and_five_warp_launch() {
        let source = candidate_source(PRODUCTION).unwrap();
        assert!(source.contains("#include <cuda_awbarrier_primitives.h>"));
        assert!(source.contains("#define GEMM_BI_TC64_THREADS 160"));
        assert!(source.contains("__shared__ __align__(8) __mbarrier_t pipeline_barriers[4]"));
        assert!(source.contains("bool producer = physical_warp == 0;"));
        assert!(source.contains("int warp = physical_warp - 1;"));
        assert!(source.contains("for (int _i = lane;"));
        assert!(source.contains("_i += 32"));
        assert_eq!(source.matches("GEMM_BI_TC64_STAGE_TN_ASYNC(").count(), 2);
        assert_eq!(source.matches("gemm_bi_cp_async_16_zfill_l2(").count(), 2);
        assert!(source.contains("cp.async.mbarrier.arrive.shared.b64"));
        assert!(source.contains("__mbarrier_arrive(&filled[buf]);"));
        assert!(source.contains("__mbarrier_arrive(&ready[0]);"));
        assert!(source.contains("__mbarrier_test_wait(&ready[write_buf]"));
        assert!(source.contains("__mbarrier_test_wait(&filled[read_buf]"));
        assert!(!source.contains("__mbarrier_try_wait("));
        assert!(source.contains("if (producer) return;"));
    }

    #[test]
    fn source_preserves_retained_consumer_math_and_epilogue_exactly() {
        let parent = retained::candidate_source(PRODUCTION).unwrap();
        let source = candidate_source(PRODUCTION).unwrap();
        let mma = "mma.sync.aligned.m16n8k16.row.col.f32.";
        assert_eq!(source.matches(mma).count(), parent.matches(mma).count());
        for anchor in [
            "unsigned a_frag[2][2][4];",
            "unsigned b_frag[2][4][2];",
            "a_frag[(ks + 1) & 1][fm]",
            "a_frag[ks & 1][fm]",
            "gemm_bi_accumulate_float2_or_scalar(",
            "dst, alpha * acc[fm][fn][2 * half]",
            "dst[0] += alpha * acc[fm][fn][2 * half];",
        ] {
            assert_eq!(
                source.matches(anchor).count(),
                parent.matches(anchor).count()
            );
        }
    }

    #[test]
    fn malformed_parent_fails_closed() {
        assert!(candidate_source("").is_err());
        let duplicate = format!("{PRODUCTION}\n#define GEMM_BI_TC64_STAGE_TN_ASYNC");
        assert!(candidate_source(&duplicate).is_err());
    }
}
