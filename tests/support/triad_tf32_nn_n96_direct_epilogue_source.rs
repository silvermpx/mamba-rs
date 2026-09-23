pub const SYMBOL: &str = "nn_triad_sm89_add_half_tf32_direct_epilogue_exp_m128n96_bk32_s3";
pub const RETAINED_SYMBOL: &str = "nn_triad_sm89_add_half_tf32_exp_m128n96_bk32_s3";
const EXPECTED_RETAINED_FNV64: u64 = 0xd582_2aef_d746_6948;

const OLD_EPILOGUE_ENTRY: &str = r#"    __syncthreads();
    float* tile_output = reinterpret_cast<float*>(shared_bytes);"#;

const NEW_EPILOGUE_ENTRY: &str = r#"    __syncthreads();
    bool direct_pair_epilogue = params.alpha == 1.0f && params.beta == 0.0f
        && tile_row + 128 <= params.m && tile_column + 96 <= params.n
        && (params.ldc & 1) == 0
        && (reinterpret_cast<unsigned long long>(output) & 7ULL) == 0ULL;
    if (direct_pair_epilogue) {
#pragma unroll
        for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < 3; ++n_atom) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    int row = tile_row + warp_m + m_atom * 16 + group + half * 8;
                    int column = tile_column + warp_n + n_atom * 8 + 2 * thread;
                    float* destination = output + (long long)row * params.ldc + column;
                    *reinterpret_cast<float2*>(destination) = make_float2(
                        acc[m_atom][n_atom][2 * half],
                        acc[m_atom][n_atom][2 * half + 1]);
                }
            }
        }
        return;
    }
    float* tile_output = reinterpret_cast<float*>(shared_bytes);"#;

pub fn compose_candidate_source(retained_source: &str) -> Result<String, String> {
    let observed = fnv64(retained_source.as_bytes());
    if observed != EXPECTED_RETAINED_FNV64 {
        return Err(format!(
            "retained N96 source changed: expected {EXPECTED_RETAINED_FNV64:#018x}, observed {observed:#018x}"
        ));
    }
    let mut source = retained_source.to_owned();
    replace_exact(
        &mut source,
        OLD_EPILOGUE_ENTRY,
        NEW_EPILOGUE_ENTRY,
        "epilogue entry",
    )?;
    replace_exact(&mut source, RETAINED_SYMBOL, SYMBOL, "export symbol")?;
    Ok(source)
}

pub fn restore_retained_source(candidate: &str) -> Result<String, String> {
    let mut source = candidate.to_owned();
    replace_exact(
        &mut source,
        NEW_EPILOGUE_ENTRY,
        OLD_EPILOGUE_ENTRY,
        "restored epilogue entry",
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
            "N96 direct-epilogue {label} seam changed: expected 1, observed {count}"
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
