pub const SYMBOL: &str =
    "gemm_bi_nn_triad_sm89_add_half_tf32_direct_epilogue_full_domain_cpasync_exp_m128n96_bk32_s3";
pub const DIRECT_RETAINED_SYMBOL: &str =
    "gemm_bi_nn_triad_sm89_add_half_tf32_direct_epilogue_exp_m128n96_bk32_s3";
const EXPECTED_DIRECT_RETAINED_FNV64: u64 = 0x8144_785a_a923_8460;

const COPY_PLAN_ENTRY: &str = "struct Tf32n96CopyPlan {";
const FULL_COPY_AND_PLAN_ENTRY: &str = r#"__device__ __forceinline__ void tf32n96_copy_cg_full(
    unsigned shared_dst, const void* global_src) {
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
        :: "r"(shared_dst), "l"(global_src));
}

struct Tf32n96CopyPlan {"#;

const RETAINED_STAGE_SLICE: &str = r#"__device__ __forceinline__ void tf32n96_stage_slice(
    const Tf32n96CopyPlan& plan, unsigned a_stage_bytes,
    unsigned b_stage_bytes, int k_base, int reduction, int issue) {
    {
        int remaining = reduction - k_base - plan.a_k_offset;
        remaining = remaining < 0 ? 0 : (remaining > 4 ? 4 : remaining);
        int bytes = plan.a_row_valid[issue] ? remaining * 4 : 0;
        gbf_tf32_copy_cg(
            plan.a_destination[issue] + a_stage_bytes,
            plan.a_source[issue], bytes);
    }
    if (issue < 3) {
        int bytes = k_base + plan.b_k_row[issue] < reduction
            ? plan.b_column_bytes[issue]
            : 0;
        gbf_tf32_copy_cg(
            plan.b_destination[issue] + b_stage_bytes,
            plan.b_source[issue], bytes);
    }
}"#;

const CANDIDATE_STAGE_SLICE: &str = r#"__device__ __forceinline__ void tf32n96_stage_slice(
    const Tf32n96CopyPlan& plan, unsigned a_stage_bytes,
    unsigned b_stage_bytes, int k_base, int reduction,
    bool full_copy_domain, int issue) {
    if (full_copy_domain) {
        tf32n96_copy_cg_full(
            plan.a_destination[issue] + a_stage_bytes,
            plan.a_source[issue]);
    } else {
        int remaining = reduction - k_base - plan.a_k_offset;
        remaining = remaining < 0 ? 0 : (remaining > 4 ? 4 : remaining);
        int bytes = plan.a_row_valid[issue] ? remaining * 4 : 0;
        gbf_tf32_copy_cg(
            plan.a_destination[issue] + a_stage_bytes,
            plan.a_source[issue], bytes);
    }
    if (issue < 3) {
        if (full_copy_domain) {
            tf32n96_copy_cg_full(
                plan.b_destination[issue] + b_stage_bytes,
                plan.b_source[issue]);
        } else {
            int bytes = k_base + plan.b_k_row[issue] < reduction
                ? plan.b_column_bytes[issue]
                : 0;
            gbf_tf32_copy_cg(
                plan.b_destination[issue] + b_stage_bytes,
                plan.b_source[issue], bytes);
        }
    }
}"#;

const RETAINED_STAGE_ASYNC: &str = r#"__device__ __forceinline__ void tf32n96_stage_async(
    const Tf32n96CopyPlan& plan, unsigned a_stage_bytes,
    unsigned b_stage_bytes, int k_base, int reduction) {
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        tf32n96_stage_slice(
            plan, a_stage_bytes, b_stage_bytes, k_base, reduction, issue);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}"#;

const CANDIDATE_STAGE_ASYNC: &str = r#"__device__ __forceinline__ void tf32n96_stage_async(
    const Tf32n96CopyPlan& plan, unsigned a_stage_bytes,
    unsigned b_stage_bytes, int k_base, int reduction,
    bool full_copy_domain) {
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        tf32n96_stage_slice(
            plan, a_stage_bytes, b_stage_bytes, k_base, reduction,
            full_copy_domain, issue);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}"#;

const FULL_DOMAIN_GUARD: &str = r#"    bool full_copy_domain = params.k > 0 && (params.k & 31) == 0
        && tile_row + 128 <= params.m && tile_column + 96 <= params.n
        && (params.lda & 3) == 0 && (params.ldb & 3) == 0
        && (reinterpret_cast<unsigned long long>(a) & 15ULL) == 0ULL
        && (reinterpret_cast<unsigned long long>(b) & 15ULL) == 0ULL;"#;

const RETAINED_INITIAL_STAGE_CALL: &str = r#"            tf32n96_stage_async(
                plan, tile * 128U * 32U * 4U, tile * 32U * 96U * 4U,
                (int)(tile * 32U), params.k);"#;
const CANDIDATE_INITIAL_STAGE_CALL: &str = r#"            tf32n96_stage_async(
                plan, tile * 128U * 32U * 4U, tile * 32U * 96U * 4U,
                (int)(tile * 32U), params.k, full_copy_domain);"#;

const RETAINED_NEXT_STAGE_CALL: &str = r#"                tf32n96_stage_slice(
                    plan, write_a_bytes, write_b_bytes,
                    (int)(next * 32U), params.k, issue);"#;
const CANDIDATE_NEXT_STAGE_CALL: &str = r#"                tf32n96_stage_slice(
                    plan, write_a_bytes, write_b_bytes,
                    (int)(next * 32U), params.k, full_copy_domain, issue);"#;

const RETAINED_MAINLOOP: &str = r#"    Tf32n96CopyPlan plan;
    tf32n96_copy_plan(
        a_stages, b_stages, a, b, params, tile_row, tile_column, plan);
    long long b_slab_rows = 32LL * params.ldb;
    unsigned tile_count = (static_cast<unsigned>(params.k) + 31U) / 32U;
#pragma unroll
    for (unsigned tile = 0; tile < 2; ++tile) {
        if (tile < tile_count) {
            tf32n96_stage_async(
                plan, tile * 128U * 32U * 4U, tile * 32U * 96U * 4U,
                (int)(tile * 32U), params.k);
            tf32n96_advance_plan(plan, b_slab_rows);
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
    }
    int read_stage = 0;
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();
        unsigned next = tile + 2;
        bool has_next = next < tile_count;
        int write_stage = read_stage == 0 ? 2 : read_stage - 1;
        unsigned write_a_bytes = (unsigned)write_stage * 128U * 32U * 4U;
        unsigned write_b_bytes = (unsigned)write_stage * 32U * 96U * 4U;
        const float* a_read = a_stages + read_stage * 128 * 32;
        const float* b_read = b_stages + read_stage * 32 * 96;
        Tf32n96Fragments fragments[2];
        tf32n96_load_fragments(a_read, b_read, 0, offsets, fragments[0]);
#pragma unroll
        for (int issue = 0; issue < 4; ++issue) {
            if (has_next) {
                tf32n96_stage_slice(
                    plan, write_a_bytes, write_b_bytes,
                    (int)(next * 32U), params.k, issue);
            }
            if (issue < 3) {
                tf32n96_load_fragments(
                    a_read, b_read, issue + 1, offsets, fragments[(issue + 1) & 1]);
            }
            tf32n96_mma(fragments[issue & 1], acc);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        if (has_next) tf32n96_advance_plan(plan, b_slab_rows);
        if (++read_stage == 3) read_stage = 0;
    }"#;

pub const fn full_domain(
    m: usize,
    k: usize,
    n: usize,
    tile_row: usize,
    tile_column: usize,
    lda: usize,
    ldb: usize,
    a_address: usize,
    b_address: usize,
) -> bool {
    k > 0
        && k.is_multiple_of(32)
        && tile_row + 128 <= m
        && tile_column + 96 <= n
        && lda.is_multiple_of(4)
        && ldb.is_multiple_of(4)
        && a_address.is_multiple_of(16)
        && b_address.is_multiple_of(16)
}

pub fn compose_candidate_source(direct_retained_source: &str) -> Result<String, String> {
    let observed = fnv64(direct_retained_source.as_bytes());
    if observed != EXPECTED_DIRECT_RETAINED_FNV64 {
        return Err(format!(
            "direct-retained N96 source changed: expected {EXPECTED_DIRECT_RETAINED_FNV64:#018x}, observed {observed:#018x}"
        ));
    }
    let mut source = direct_retained_source.to_owned();
    replace_exact(
        &mut source,
        COPY_PLAN_ENTRY,
        FULL_COPY_AND_PLAN_ENTRY,
        "full copy helper",
    )?;
    replace_exact(
        &mut source,
        RETAINED_STAGE_SLICE,
        CANDIDATE_STAGE_SLICE,
        "stage slice template",
    )?;
    replace_exact(
        &mut source,
        RETAINED_STAGE_ASYNC,
        CANDIDATE_STAGE_ASYNC,
        "stage async template",
    )?;
    let candidate_mainloop = candidate_mainloop()?;
    replace_exact(
        &mut source,
        RETAINED_MAINLOOP,
        &candidate_mainloop,
        "full/edge mainloop branch",
    )?;
    replace_exact(&mut source, DIRECT_RETAINED_SYMBOL, SYMBOL, "export symbol")?;
    Ok(source)
}

pub fn restore_direct_retained_source(candidate: &str) -> Result<String, String> {
    let mut source = candidate.to_owned();
    replace_exact(
        &mut source,
        SYMBOL,
        DIRECT_RETAINED_SYMBOL,
        "restored export symbol",
    )?;
    let candidate_mainloop = candidate_mainloop()?;
    replace_exact(
        &mut source,
        &candidate_mainloop,
        RETAINED_MAINLOOP,
        "restored mainloop",
    )?;
    replace_exact(
        &mut source,
        CANDIDATE_STAGE_ASYNC,
        RETAINED_STAGE_ASYNC,
        "restored stage async",
    )?;
    replace_exact(
        &mut source,
        CANDIDATE_STAGE_SLICE,
        RETAINED_STAGE_SLICE,
        "restored stage slice",
    )?;
    replace_exact(
        &mut source,
        FULL_COPY_AND_PLAN_ENTRY,
        COPY_PLAN_ENTRY,
        "removed full copy helper",
    )?;
    Ok(source)
}

fn candidate_mainloop() -> Result<String, String> {
    let mut mainloop = RETAINED_MAINLOOP.to_owned();
    replace_exact(
        &mut mainloop,
        RETAINED_INITIAL_STAGE_CALL,
        CANDIDATE_INITIAL_STAGE_CALL,
        "initial stage call",
    )?;
    replace_exact(
        &mut mainloop,
        RETAINED_NEXT_STAGE_CALL,
        CANDIDATE_NEXT_STAGE_CALL,
        "next stage call",
    )?;
    Ok(format!("{FULL_DOMAIN_GUARD}\n{mainloop}"))
}

fn replace_exact(source: &mut String, old: &str, new: &str, label: &str) -> Result<(), String> {
    let count = source.matches(old).count();
    if count != 1 {
        return Err(format!(
            "N96 full-domain cp.async {label} seam changed: expected 1, observed {count}"
        ));
    }
    *source = source.replacen(old, new, 1);
    Ok(())
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
