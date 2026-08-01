//! T-1 (scan-audit 2026-08-01): CPU-only pins for the scan-mode dispatch
//! rules — the one scan test CI can run today (no cuda feature needed).
//!
//! Pins: the Auto threshold boundary, the `d_state > 64` sequential-kernel
//! override in BOTH entry points (`use_parallel` and `resolve` — the latter
//! used to omit it), and the explicit modes.

use mamba_rs::config::ScanMode;

const THR: usize = ScanMode::PARALLEL_SCAN_THRESHOLD;

#[test]
fn auto_threshold_boundary() {
    assert!(!ScanMode::Auto.use_parallel(THR - 1, 16));
    assert!(
        !ScanMode::Auto.use_parallel(THR, 16),
        "T == threshold stays sequential"
    );
    assert!(ScanMode::Auto.use_parallel(THR + 1, 16));
}

#[test]
fn d_state_override_forces_parallel_in_every_mode() {
    for mode in [ScanMode::Sequential, ScanMode::Parallel, ScanMode::Auto] {
        assert!(
            mode.use_parallel(1, 65),
            "{mode:?}: d_state > 64 must force the parallel path (the sequential \
             kernels silently no-op past their register cap)"
        );
    }
}

#[test]
fn explicit_modes_ignore_length() {
    assert!(!ScanMode::Sequential.use_parallel(1 << 20, 16));
    assert!(ScanMode::Parallel.use_parallel(1, 16));
}

#[test]
fn resolve_agrees_with_use_parallel_everywhere() {
    for mode in [ScanMode::Sequential, ScanMode::Parallel, ScanMode::Auto] {
        for t in [1, THR - 1, THR, THR + 1, 4621] {
            for ds in [8, 16, 64, 65, 256] {
                let resolved = mode.resolve(t, ds);
                assert_eq!(
                    resolved == ScanMode::Parallel,
                    mode.use_parallel(t, ds),
                    "resolve/use_parallel diverged at mode={mode:?} T={t} ds={ds}"
                );
            }
        }
    }
}
