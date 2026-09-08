pub const SYMBOL: &str = "gemm_bi_nn_triad_sm89_add_half_tf32_b_ldmatrix_exp_m128n96_bk32_s3";
pub const RETAINED_SYMBOL: &str = "gemm_bi_nn_triad_sm89_add_half_tf32_exp_m128n96_bk32_s3";
const EXPECTED_RETAINED_FNV64: u64 = 0xe8a2_1385_db2b_b352;

pub const fn n96_grid(m: usize, n: usize) -> usize {
    m.div_ceil(128) * n.div_ceil(96)
}

const OLD_SIGNATURE: &str = r#"__device__ __forceinline__ void tf32n96_load_fragments(
    const float* a_stage, const float* b_stage, int step,
    const Tf32n96FragmentOffsets& offsets, Tf32n96Fragments& fragments) {"#;

const NEW_SIGNATURE: &str = r#"__device__ __forceinline__ void tf32n96_load_fragments(
    const float* a_stage, const float* b_stage, int step, int warp_n,
    const Tf32n96FragmentOffsets& offsets, Tf32n96Fragments& fragments) {"#;

const OLD_B_LOAD: &str = r#"    const float* b_step = b_stage + step * 8 * 96;
#pragma unroll
    for (int n_atom = 0; n_atom < 3; ++n_atom) {
        fragments.b[n_atom][0] =
            tf32n96_round(__float_as_uint(b_step[offsets.b[n_atom][0]]));
        fragments.b[n_atom][1] =
            tf32n96_round(__float_as_uint(b_step[offsets.b[n_atom][1]]));
    }
"#;

const NEW_B_LOAD: &str = r#"    unsigned b_base = (unsigned)__cvta_generic_to_shared(b_stage);
    int lane = (int)threadIdx.x & 31;
    int address_matrix = (lane >> 3) & 1;
    int address_row = lane & 7;
    int source_lane = ((lane & 3) << 2) + ((lane >> 2) & 3)
        + (((lane >> 2) >= 4) ? 16 : 0);
#pragma unroll
    for (int n_atom = 0; n_atom < 3; ++n_atom) {
        int reduction = step * 8 + address_matrix * 4 + (address_row & 3);
        int column = warp_n + n_atom * 8 + (address_row >> 2) * 4;
        unsigned address = b_base
            + (unsigned)tf32n96_b_index(reduction, column) * 4U;
        unsigned raw0, raw1;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
            : "=r"(raw0), "=r"(raw1)
            : "r"(address));
        raw0 = __shfl_sync(0xffffffffU, raw0, source_lane);
        raw1 = __shfl_sync(0xffffffffU, raw1, source_lane);
        fragments.b[n_atom][0] = tf32n96_round(raw0);
        fragments.b[n_atom][1] = tf32n96_round(raw1);
    }
"#;

const OLD_FIRST_CALL: &str = "tf32n96_load_fragments(a_read, b_read, 0, offsets, fragments[0]);";
const NEW_FIRST_CALL: &str =
    "tf32n96_load_fragments(a_read, b_read, 0, warp_n, offsets, fragments[0]);";
const OLD_NEXT_CALL: &str = r#"tf32n96_load_fragments(
                    a_read, b_read, issue + 1, offsets, fragments[(issue + 1) & 1]);"#;
const NEW_NEXT_CALL: &str = r#"tf32n96_load_fragments(
                    a_read, b_read, issue + 1, warp_n, offsets, fragments[(issue + 1) & 1]);"#;

pub const fn b_index(k: usize, column: usize) -> usize {
    let chunk = (column >> 2) ^ ((k & 3) << 1);
    k * 96 + chunk * 4 + (column & 3)
}

pub const fn scalar_coordinate(
    k8: usize,
    column_base: usize,
    lane: usize,
    register: usize,
) -> (usize, usize) {
    (k8 + (lane & 3) + register * 4, column_base + (lane >> 2))
}

pub const fn source_lane_for_consumer(lane: usize) -> usize {
    ((lane & 3) << 2) + ((lane >> 2) & 3) + if (lane >> 2) >= 4 { 16 } else { 0 }
}

pub const fn ldmatrix_row_coordinate(
    warp_n: usize,
    atom: usize,
    k8: usize,
    address_lane: usize,
) -> (usize, usize) {
    (
        k8 + (address_lane >> 3) * 4 + (address_lane & 3),
        warp_n + atom * 8 + ((address_lane & 7) >> 2) * 4,
    )
}

pub const fn loaded_coordinate(
    warp_n: usize,
    atom: usize,
    k8: usize,
    lane: usize,
    register: usize,
) -> (usize, usize) {
    let row = lane >> 2;
    (
        k8 + register * 4 + (row & 3),
        warp_n + atom * 8 + (row >> 2) * 4 + (lane & 3),
    )
}

pub fn compose_candidate_source(retained_source: &str) -> Result<String, String> {
    let observed = fnv64(retained_source.as_bytes());
    if observed != EXPECTED_RETAINED_FNV64 {
        return Err(format!(
            "retained N96 source changed: expected {EXPECTED_RETAINED_FNV64:#018x}, observed {observed:#018x}"
        ));
    }
    let mut source = retained_source.to_owned();
    replace_exact(&mut source, OLD_SIGNATURE, NEW_SIGNATURE, "load signature")?;
    replace_exact(&mut source, OLD_B_LOAD, NEW_B_LOAD, "B fragment load")?;
    replace_exact(
        &mut source,
        OLD_FIRST_CALL,
        NEW_FIRST_CALL,
        "first load call",
    )?;
    replace_exact(&mut source, OLD_NEXT_CALL, NEW_NEXT_CALL, "next load call")?;
    replace_exact(&mut source, RETAINED_SYMBOL, SYMBOL, "export symbol")?;
    Ok(source)
}

pub fn restore_retained_source(candidate: &str) -> Result<String, String> {
    let mut source = candidate.to_owned();
    replace_exact(
        &mut source,
        NEW_SIGNATURE,
        OLD_SIGNATURE,
        "restored load signature",
    )?;
    replace_exact(
        &mut source,
        NEW_B_LOAD,
        OLD_B_LOAD,
        "restored B fragment load",
    )?;
    replace_exact(
        &mut source,
        NEW_FIRST_CALL,
        OLD_FIRST_CALL,
        "restored first load call",
    )?;
    replace_exact(
        &mut source,
        NEW_NEXT_CALL,
        OLD_NEXT_CALL,
        "restored next load call",
    )?;
    replace_exact(
        &mut source,
        SYMBOL,
        RETAINED_SYMBOL,
        "restored export symbol",
    )?;
    Ok(source)
}

fn replace_exact(source: &mut String, old: &str, new: &str, label: &str) -> Result<(), String> {
    let count = source.matches(old).count();
    if count != 1 {
        return Err(format!(
            "N96 B-ldmatrix {label} seam changed: expected 1, observed {count}"
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
