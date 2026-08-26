//! Compare two acceptance-evidence TSV files cell by cell.
//!
//! Usage: acceptance_diff <base.tsv> <candidate.tsv>
//!
//! Each file holds `suite<TAB>arm<TAB>key<TAB>value` rows written by the
//! test harness under MAMBA_RS_ACCEPTANCE_TSV. The diff prints every
//! cell whose value moved and every cell present on only one side, and
//! exits non-zero if anything differs - so a merge acceptance can gate
//! on it mechanically instead of on a human reading two hash tables.
use std::collections::BTreeMap;
use std::process::ExitCode;

fn load(path: &str) -> BTreeMap<String, String> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let mut cells = BTreeMap::new();
    for (i, line) in text.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.splitn(4, '\t');
        let (Some(suite), Some(arm), Some(key), Some(value)) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            panic!("{path}:{}: malformed row: {line:?}", i + 1);
        };
        // A re-run of the same suite appends again; last write wins,
        // matching what a fresh single-pass capture would hold.
        cells.insert(format!("{suite}\t{arm}\t{key}"), value.to_string());
    }
    cells
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let [_, base_path, cand_path] = args.as_slice() else {
        eprintln!("usage: acceptance_diff <base.tsv> <candidate.tsv>");
        return ExitCode::from(2);
    };
    let base = load(base_path);
    let cand = load(cand_path);
    let mut moved = 0usize;
    let mut only_base = 0usize;
    let mut only_cand = 0usize;
    for (key, bv) in &base {
        match cand.get(key) {
            Some(cv) if cv == bv => {}
            Some(cv) => {
                println!("MOVED {} | {} -> {}", key.replace('\t', " / "), bv, cv);
                moved += 1;
            }
            None => {
                println!("ONLY-BASE {} | {}", key.replace('\t', " / "), bv);
                only_base += 1;
            }
        }
    }
    for (key, cv) in &cand {
        if !base.contains_key(key) {
            println!("ONLY-CANDIDATE {} | {}", key.replace('\t', " / "), cv);
            only_cand += 1;
        }
    }
    let same = base.len() - moved - only_base;
    println!(
        "{same} identical, {moved} moved, {only_base} only in base, {only_cand} only in candidate"
    );
    if moved + only_base + only_cand == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
