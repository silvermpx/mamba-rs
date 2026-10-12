//! The MIMO kernels size their per-thread rank arrays from `MIMO_MAX_RANK`,
//! and the config admits ranks up to `MAX_MIMO_RANK`. The two must be one
//! number: a rank the config admits past the kernels' bound would overflow
//! their local arrays without an error.

#[test]
fn the_kernel_rank_bound_is_the_config_rank_bound() {
    let src = include_str!("../kernels/mamba3_mimo.cu");
    let define = src
        .lines()
        .find(|l| l.trim_start().starts_with("#define MIMO_MAX_RANK "))
        .expect("mamba3_mimo.cu defines MIMO_MAX_RANK");
    let kernel_bound: usize = define
        .split_whitespace()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .expect("MIMO_MAX_RANK is a plain number");
    assert_eq!(
        kernel_bound,
        mamba_rs::mamba3_mimo::config::MAX_MIMO_RANK,
        "the kernels' rank bound and the config's rank bound differ"
    );
}
