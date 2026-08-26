//! Shared test/bench harness modules (not a test target itself).
#![allow(
    dead_code,
    reason = "shared harness: each test binary consumes a subset of these helpers"
)]

pub mod bench;
pub mod digest;
