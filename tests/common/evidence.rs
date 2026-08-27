//! Acceptance evidence: machine-readable capture of every bit-gate cell.
//!
//! "Re-run and diff stdout" turns a sixteen-cell hash table into a
//! human eyeball test; a merge acceptance needs the cells as data. When
//! `MAMBA_RS_ACCEPTANCE_TSV` names a file, every participating printer
//! appends its cells there as tab-separated rows
//! (`suite<TAB>arm<TAB>key<TAB>value`), and `examples/acceptance_diff`
//! compares two such files. With the variable unset this module is
//! inert, so ordinary runs pay nothing. The TSV shape follows the
//! cublas-probe artifact precedent: append-only, zero dependencies,
//! diffable by eye when the tooling is not at hand.
use std::io::Write as _;

pub fn record(suite: &str, arm: &str, key: &str, value: &str) {
    let Ok(path) = std::env::var("MAMBA_RS_ACCEPTANCE_TSV") else {
        return;
    };
    if path.is_empty() {
        return;
    }
    let line = format!("{suite}\t{arm}\t{key}\t{value}\n");
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = f.write_all(line.as_bytes());
    }
}

pub fn record_digest(suite: &str, arm: &str, key: &str, digest: u64) {
    record(suite, arm, key, &format!("{digest:016x}"));
}
