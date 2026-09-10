#[path = "triad_half_nt_m64n128_s3_source.rs"]
mod measured_s3;

pub const SYMBOL_PREFIX: &str = "gemm_bi_nt_test_fixed_s2_m64n128_";
pub const D768_OUT: (usize, usize, usize) = (2_048, 1_536, 768);
pub const BLOCK_THREADS: u32 = 256;
pub const DYNAMIC_SHARED_BYTES: usize = 49_152;
pub const REQUIRED_OCCUPANCY: u32 = 2;

const MEASURED_S3_FNV1A64: u64 = 0x5856_5c11_3487_2c4d;

pub fn measured_s3_source(swizzle: &str, s3: &str) -> Result<String, String> {
    let source = measured_s3::candidate_source(swizzle, s3)?;
    let observed = fnv1a64(source.as_bytes());
    if observed != MEASURED_S3_FNV1A64 {
        return Err(format!(
            "measured M64N128/S3 source drifted: expected {MEASURED_S3_FNV1A64:#018x}, observed {observed:#018x}"
        ));
    }
    Ok(source)
}

pub fn candidate_source(swizzle: &str, s3: &str) -> Result<String, String> {
    let mut source = measured_s3_source(swizzle, s3)?;
    s3_to_s2(&mut source)?;
    Ok(source)
}

pub fn restore_measured_s3_source(candidate: &str) -> Result<String, String> {
    let mut source = candidate.to_owned();
    s2_to_s3(&mut source)?;
    let observed = fnv1a64(source.as_bytes());
    if observed != MEASURED_S3_FNV1A64 {
        return Err(format!(
            "reverse transform did not restore measured M64N128/S3: expected {MEASURED_S3_FNV1A64:#018x}, observed {observed:#018x}"
        ));
    }
    Ok(source)
}

pub fn all_retained_strata_pass(strata: &[[f64; 2]]) -> bool {
    strata.len() == 4
        && strata
            .iter()
            .all(|[p50, p95]| p50.is_finite() && p95.is_finite() && *p50 < 0.99 && *p95 < 0.99)
}

fn s3_to_s2(source: &mut String) -> Result<(), String> {
    replace_exact(
        source,
        "static constexpr int kSharedBytes = 73728;\nstatic_assert(3 * (64 * 64 + 128 * 64) * 2 == kSharedBytes, \"M64N128 S3 shared ABI\");",
        "static constexpr int kSharedBytes = 49152;\nstatic_assert(2 * (64 * 64 + 128 * 64) * 2 == kSharedBytes, \"M64N128 S2 shared ABI\");",
    )?;
    replace_exact(
        source,
        "        sm89_fhs_shared + 3 * 64 * 64 * (int)sizeof(T));",
        "        sm89_fhs_shared + 2 * 64 * 64 * (int)sizeof(T));",
    )?;
    replace_exact(source, S3_SECOND_PROLOGUE, "")?;
    replace_exact(source, S3_FAST_PIPELINE, S2_FAST_PIPELINE)?;
    replace_all_exact(
        source,
        "sm89_test_half_nt_m64n128_s3",
        "sm89_test_half_nt_m64n128_s2",
        3,
    )?;
    replace_all_exact(
        source,
        "gemm_bi_nt_test_fixed_s3_m64n128_",
        SYMBOL_PREFIX,
        1,
    )
}

fn s2_to_s3(source: &mut String) -> Result<(), String> {
    replace_all_exact(
        source,
        "gemm_bi_nt_test_fixed_s2_m64n128_",
        measured_s3::SYMBOL_PREFIX,
        1,
    )?;
    replace_all_exact(
        source,
        "sm89_test_half_nt_m64n128_s2",
        "sm89_test_half_nt_m64n128_s3",
        3,
    )?;
    replace_exact(source, S2_FAST_PIPELINE, S3_FAST_PIPELINE)?;
    replace_exact(
        source,
        S3_PROLOGUE_INSERTION_POINT,
        &format!("{S3_PROLOGUE_INSERTION_POINT}{S3_SECOND_PROLOGUE}"),
    )?;
    replace_exact(
        source,
        "        sm89_fhs_shared + 2 * 64 * 64 * (int)sizeof(T));",
        "        sm89_fhs_shared + 3 * 64 * 64 * (int)sizeof(T));",
    )?;
    replace_exact(
        source,
        "static constexpr int kSharedBytes = 49152;\nstatic_assert(2 * (64 * 64 + 128 * 64) * 2 == kSharedBytes, \"M64N128 S2 shared ABI\");",
        "static constexpr int kSharedBytes = 73728;\nstatic_assert(3 * (64 * 64 + 128 * 64) * 2 == kSharedBytes, \"M64N128 S3 shared ABI\");",
    )
}

const S3_PROLOGUE_INSERTION_POINT: &str = r#"            asm volatile("cp.async.commit_group;\n" ::);
"#;

const S3_SECOND_PROLOGUE: &str = r#"            if (num_k_tiles > 1) {
#pragma unroll
                for (int slice = 0; slice < 4; ++slice)
                    sm89_fixed_half_swizzle::nt_m64n128_copy_slice(plan, A, B, 1, 64, N, slice);
                asm volatile("cp.async.commit_group;\n" ::);
            }
"#;

const S3_FAST_PIPELINE: &str = r#"    if (fast_stage) {
        sm89_fixed_half_swizzle::Fragments fragments[2];
        if (num_k_tiles > 0) {
            if (num_k_tiles > 1) asm volatile("cp.async.wait_group 1;\n" ::);
            else asm volatile("cp.async.wait_group 0;\n" ::);
            __syncthreads();
            sm89_fixed_half_swizzle::nt_m64n128_load_fragments(As_sbase, Bs_sbase, 0, offsets, fragments[0]);
        }
        int write_buf = 2;
        for (int kt = 0; kt < num_k_tiles; ++kt) {
            bool next = kt + 1 < num_k_tiles;
            bool refill = kt + 2 < num_k_tiles;
            int next_k = (kt + 2) * 64;
            unsigned a_read = As_sbase + (unsigned)(read_buf * 64 * 64 * 2);
            unsigned b_read = Bs_sbase + (unsigned)(read_buf * 64 * 128 * 2);

            if (refill) sm89_fixed_half_swizzle::nt_m64n128_copy_slice(plan, A, B, write_buf, next_k, N, 0);
            sm89_fixed_half_swizzle::nt_m64n128_load_fragments(a_read, b_read, 1, offsets, fragments[1]);
            sm89_fixed_half_swizzle::nt_m64n128_consume_fragments<T>(fragments[0], acc);

            if (refill) sm89_fixed_half_swizzle::nt_m64n128_copy_slice(plan, A, B, write_buf, next_k, N, 1);
            sm89_fixed_half_swizzle::nt_m64n128_load_fragments(a_read, b_read, 2, offsets, fragments[0]);
            sm89_fixed_half_swizzle::nt_m64n128_consume_fragments<T>(fragments[1], acc);

            if (refill) {
                sm89_fixed_half_swizzle::nt_m64n128_copy_slice(plan, A, B, write_buf, next_k, N, 2);
                sm89_fixed_half_swizzle::nt_m64n128_copy_slice(plan, A, B, write_buf, next_k, N, 3);
            }
            sm89_fixed_half_swizzle::nt_m64n128_load_fragments(a_read, b_read, 3, offsets, fragments[1]);
            sm89_fixed_half_swizzle::nt_m64n128_consume_fragments<T>(fragments[0], acc);

            if (refill) asm volatile("cp.async.commit_group;\n" ::);
            if (next) {
                if (refill) asm volatile("cp.async.wait_group 1;\n" ::);
                else asm volatile("cp.async.wait_group 0;\n" ::);
                __syncthreads();
                read_buf = read_buf == 2 ? 0 : read_buf + 1;
                write_buf = write_buf == 2 ? 0 : write_buf + 1;
                unsigned next_a_read = As_sbase + (unsigned)(read_buf * 64 * 64 * 2);
                unsigned next_b_read = Bs_sbase + (unsigned)(read_buf * 64 * 128 * 2);
                sm89_fixed_half_swizzle::nt_m64n128_load_fragments(next_a_read, next_b_read, 0, offsets, fragments[0]);
            }
            sm89_fixed_half_swizzle::nt_m64n128_consume_fragments<T>(fragments[1], acc);
        }
    }"#;

const S2_FAST_PIPELINE: &str = r#"    if (fast_stage) {
        sm89_fixed_half_swizzle::Fragments fragments[2];
        for (int kt = 0; kt < num_k_tiles; ++kt) {
            asm volatile("cp.async.wait_group 0;\n" ::);
            __syncthreads();
            bool next = kt + 1 < num_k_tiles;
            int next_k = (kt + 1) * 64;
            int write_buf = read_buf ^ 1;
            unsigned a_read = As_sbase + (unsigned)(read_buf * 64 * 64 * 2);
            unsigned b_read = Bs_sbase + (unsigned)(read_buf * 64 * 128 * 2);

            sm89_fixed_half_swizzle::nt_m64n128_load_fragments(a_read, b_read, 0, offsets, fragments[0]);
            if (next) sm89_fixed_half_swizzle::nt_m64n128_copy_slice(plan, A, B, write_buf, next_k, N, 0);
            sm89_fixed_half_swizzle::nt_m64n128_load_fragments(a_read, b_read, 1, offsets, fragments[1]);
            sm89_fixed_half_swizzle::nt_m64n128_consume_fragments<T>(fragments[0], acc);

            if (next) sm89_fixed_half_swizzle::nt_m64n128_copy_slice(plan, A, B, write_buf, next_k, N, 1);
            sm89_fixed_half_swizzle::nt_m64n128_load_fragments(a_read, b_read, 2, offsets, fragments[0]);
            sm89_fixed_half_swizzle::nt_m64n128_consume_fragments<T>(fragments[1], acc);

            if (next) {
                sm89_fixed_half_swizzle::nt_m64n128_copy_slice(plan, A, B, write_buf, next_k, N, 2);
                sm89_fixed_half_swizzle::nt_m64n128_copy_slice(plan, A, B, write_buf, next_k, N, 3);
            }
            sm89_fixed_half_swizzle::nt_m64n128_load_fragments(a_read, b_read, 3, offsets, fragments[1]);
            sm89_fixed_half_swizzle::nt_m64n128_consume_fragments<T>(fragments[0], acc);

            if (next) asm volatile("cp.async.commit_group;\n" ::);
            sm89_fixed_half_swizzle::nt_m64n128_consume_fragments<T>(fragments[1], acc);
            if (next) read_buf = write_buf;
        }
    }"#;

fn replace_exact(source: &mut String, before: &str, after: &str) -> Result<(), String> {
    require_count(source, before, 1)?;
    *source = source.replacen(before, after, 1);
    Ok(())
}

fn replace_all_exact(
    source: &mut String,
    before: &str,
    after: &str,
    expected: usize,
) -> Result<(), String> {
    require_count(source, before, expected)?;
    *source = source.replace(before, after);
    Ok(())
}

fn require_count(source: &str, anchor: &str, expected: usize) -> Result<(), String> {
    let actual = source.matches(anchor).count();
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "half NT M64N128/S2 anchor {anchor:?}: expected {expected}, observed {actual}"
        ))
    }
}

const fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    let mut index = 0;
    while index < bytes.len() {
        hash ^= bytes[index] as u64;
        hash = hash.wrapping_mul(0x100000001b3);
        index += 1;
    }
    hash
}
