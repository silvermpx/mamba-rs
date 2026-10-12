//! CPU-only pins for the scan-mode dispatch
//! rules — the one scan test CI can run today (no cuda feature needed).
//!
//! Pins: the Auto threshold boundary, `Auto` taking the parallel scan above
//! 64 states in both entry points (`use_parallel` and `resolve`), and the
//! explicit modes honoured at every length and state size.

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
fn auto_takes_the_parallel_scan_above_64_states() {
    assert!(ScanMode::Auto.use_parallel(1, 65));
    assert!(!ScanMode::Auto.use_parallel(1, 64));
}

#[test]
fn explicit_modes_hold_at_every_state_size() {
    for d_state in [1, 16, 64, 65, 128, 256] {
        assert!(!ScanMode::Sequential.use_parallel(1, d_state), "{d_state}");
        assert!(
            !ScanMode::Sequential.use_parallel(1 << 20, d_state),
            "{d_state}"
        );
        assert!(ScanMode::Parallel.use_parallel(1, d_state), "{d_state}");
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
