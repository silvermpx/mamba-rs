#[path = "triad_half_nt_m64n128_s3_source.rs"]
mod measured;

pub const SYMBOL_PREFIX: &str = "gemm_bi_nt_test_fixed_s3_m64n128_groupm8_";
pub const RETAINED_SYMBOL_PREFIX: &str = measured::SYMBOL_PREFIX;
pub const D768_OUT: (usize, usize, usize) = (2_048, 1_536, 768);
pub const GROUP_M: usize = 8;
pub const DYNAMIC_SHARED_BYTES: usize = 73_728;
pub const BLOCK_THREADS: u32 = 256;
pub const REQUIRED_OCCUPANCY: u32 = 1;
pub const MAX_REGISTERS: i32 = 123;
const EXPECTED_RETAINED_FNV64: u64 = 0x5856_5c11_3487_2c4d;

const ROW_MAJOR_RASTER: &str = r#"    int num_pid_n = (K_out + 127) / 128;
    int pid_m = (int)blockIdx.x / num_pid_n;
    int pid_n = (int)blockIdx.x % num_pid_n;"#;

const GROUP_M8_RASTER: &str = r#"    static constexpr int kGroupM = 8;
    int num_pid_m = (M + 63) / 64;
    int num_pid_n = (K_out + 127) / 128;
    int num_pid_in_group = kGroupM * num_pid_n;
    int tile_id = (int)blockIdx.x;
    int group_id = tile_id / num_pid_in_group;
    int first_pid_m = group_id * kGroupM;
    int group_size_m = min(num_pid_m - first_pid_m, kGroupM);
    int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
    int pid_n = (tile_id % num_pid_in_group) / group_size_m;"#;

pub const fn groupm8_coords(
    tile_id: usize,
    num_pid_m: usize,
    num_pid_n: usize,
) -> Option<(usize, usize)> {
    if num_pid_n == 0 || tile_id >= num_pid_m * num_pid_n {
        return None;
    }
    let num_pid_in_group = GROUP_M * num_pid_n;
    let group_id = tile_id / num_pid_in_group;
    let first_pid_m = group_id * GROUP_M;
    let remaining_m = num_pid_m - first_pid_m;
    let group_size_m = if remaining_m < GROUP_M {
        remaining_m
    } else {
        GROUP_M
    };
    let tile_in_group = tile_id % num_pid_in_group;
    Some((
        first_pid_m + tile_in_group % group_size_m,
        tile_in_group / group_size_m,
    ))
}

pub fn measured_s3_source(swizzle: &str, s3: &str) -> Result<String, String> {
    let source = measured::candidate_source(swizzle, s3)?;
    require_retained_hash(&source)?;
    Ok(source)
}

pub fn candidate_source(swizzle: &str, s3: &str) -> Result<String, String> {
    let mut source = measured_s3_source(swizzle, s3)?;
    replace_exact(&mut source, ROW_MAJOR_RASTER, GROUP_M8_RASTER, "CTA raster")?;
    replace_exact(
        &mut source,
        RETAINED_SYMBOL_PREFIX,
        SYMBOL_PREFIX,
        "export symbol",
    )?;
    Ok(source)
}

pub fn restore_measured_s3_source(candidate: &str) -> Result<String, String> {
    let mut source = candidate.to_owned();
    replace_exact(
        &mut source,
        GROUP_M8_RASTER,
        ROW_MAJOR_RASTER,
        "restored CTA raster",
    )?;
    replace_exact(
        &mut source,
        SYMBOL_PREFIX,
        RETAINED_SYMBOL_PREFIX,
        "restored export symbol",
    )?;
    require_retained_hash(&source)?;
    Ok(source)
}

pub fn all_retained_strata_pass(strata: &[[f64; 2]]) -> bool {
    strata.len() == 4
        && strata
            .iter()
            .flatten()
            .all(|ratio| ratio.is_finite() && *ratio < 0.99)
}

fn replace_exact(source: &mut String, old: &str, new: &str, label: &str) -> Result<(), String> {
    let count = source.matches(old).count();
    if count != 1 {
        return Err(format!(
            "M64N128/S3 GROUP_M8 {label} seam changed: expected 1, observed {count}"
        ));
    }
    *source = source.replacen(old, new, 1);
    Ok(())
}

fn require_retained_hash(source: &str) -> Result<(), String> {
    let observed = fnv64(source.as_bytes());
    if observed != EXPECTED_RETAINED_FNV64 {
        return Err(format!(
            "measured M64N128/S3 parent changed: expected {EXPECTED_RETAINED_FNV64:#018x}, observed {observed:#018x}"
        ));
    }
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
