//! Hex form of a digest cell for the acceptance evidence TSV. Declare
//! `common/evidence.rs` beside this module; it reaches it through `super`.

use super::evidence::record;

pub fn record_digest(suite: &str, arm: &str, key: &str, digest: u64) -> Result<(), String> {
    record(suite, arm, key, &format!("{digest:016x}"))
}
