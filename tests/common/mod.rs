//! Shared test/bench harness modules (not a test target itself).
#![allow(
    dead_code,
    reason = "shared harness: each test binary consumes a subset of these helpers"
)]

#[cfg(feature = "cuda")]
pub mod bench;
pub mod digest;
pub mod evidence;
pub mod source_scan;
