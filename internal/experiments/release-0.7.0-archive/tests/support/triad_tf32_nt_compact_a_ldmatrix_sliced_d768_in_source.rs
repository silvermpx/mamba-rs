#[path = "triad_tf32_nt_compact_a_ldmatrix_sliced_source.rs"]
mod sliced;

#[cfg(feature = "cuda")]
pub use sliced::{
    BLOCK_THREADS, DYNAMIC_SHARED_BYTES, REGISTER_CAP, REQUIRED_OCCUPANCY, RETAINED_SYMBOL, SYMBOL,
    candidate_source, retained_source,
};

pub const TARGET: (usize, usize, usize) = (2_048, 768, 3_072);
pub const TARGET_GRID: u32 = 192;
