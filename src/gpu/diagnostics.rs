//! One-shot process warnings for the GPU dispatch.
//!
//! A decline on the selection path used to be a bare `None`: a board the
//! measured tables do not cover, a stale evidence cohort and a shape no table
//! names were indistinguishable at runtime, and a wrong board looked exactly
//! like a served one. Every such decline now reports once, in the house
//! `mamba-rs WARNING:` idiom, naming what did not match.

use std::sync::Once;

/// Print `message` to stderr the first time this call site fires.
///
/// Each call site owns its own `Once`, so unrelated declines are each
/// reported, while a decline that repeats on every GEMM call is reported
/// exactly once per process.
pub(crate) fn warn_once(site: &'static Once, message: impl FnOnce() -> String) {
    site.call_once(|| eprintln!("mamba-rs WARNING: {}", message()));
}

#[cfg(test)]
mod tests {
    use super::warn_once;
    use std::sync::Once;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn a_site_reports_once_and_never_builds_the_message_again() {
        static SITE: Once = Once::new();
        let built = AtomicUsize::new(0);
        for _ in 0..3 {
            warn_once(&SITE, || {
                built.fetch_add(1, Ordering::SeqCst);
                "diagnostics self-test".to_string()
            });
        }
        assert_eq!(built.load(Ordering::SeqCst), 1);
    }
}
