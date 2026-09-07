//! The comment gate: a code comment is plain English prose that says WHY
//! a line exists, the way a person writes it to a colleague.
//!
//! Two rules ride the same scan. Language: comments are English (Cyrillic
//! is refused; quoted data carries `comment-language:allow` on its line).
//! Bookkeeping: phase and task indexes, dates, document numbers, audit
//! tags and migration numbers are the author's own notes - they age, the
//! plan and the git log already hold them, and to the next reader they
//! mean nothing. A regression test's identifier belongs in the test
//! function's NAME, never in the prose above it.
//!
//! The existing stock is a shrink-only baseline (a multiset of
//! `path<TAB>comment` signatures): a NEW offending line fails the gate,
//! `--update-baseline` records the shrink and refuses to grow. A genuine
//! standard's name the term list misses is marked `comment-kitchen:allow`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// One scan root: a directory, the extensions to read, and whether the
/// files comment with `#` (shell, python) instead of `//`.
pub struct Scan<'a> {
    pub dir: &'a str,
    pub exts: &'a [&'a str],
    pub hashy: bool,
}

/// Directories never walked.
const SKIP_DIRS: &[&str] = &["target", "node_modules", "dist", ".git", "vendor", "build"];

/// Real-world terms that look like a task index: a hyphenated standard,
/// a hardware name, a hash. Never bookkeeping.
const TERMS: &[&str] = &[
    "UTF", "SHA", "ISO", "RFC", "IPV", "TLS", "HTTP", "AES", "MD", "CRC", "ECDSA", "RSA", "ED",
    "X", "RTX", "GTX", "CUDA", "SM", "BF", "FP", "INT", "EPYC", "USB", "PCIE", "DDR", "NVME",
    "GPT", "M", "A", "H", "P", "UUID", "ULID", "TCP", "UDP", "SSE", "AVX", "ARM", "ANSI", "IEEE",
    "ECMA", "PBKDF", "HMAC", "OTP", "TOTP", "S", "G", "L", "E", "C", "F", "SIMD", "NEON", "AMX",
    "CU", "NVCC", "TF", "PTX", "IMMA", "MMA", "WGMMA", "LDMATRIX", "CUBLAS", "CUTLASS",
];

/// The bookkeeping shape a comment carries, if any - named for the report.
pub fn kitchen_shape(comment: &str) -> Option<&'static str> {
    let words: Vec<&str> = comment
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| matches!(c, '(' | ')' | ',' | '.' | ';' | ':' | '`' | '\'' | '"' | '[' | ']')))
        .filter(|w| !w.is_empty())
        .collect();
    for (i, w) in words.iter().enumerate() {
        let next = words.get(i + 1).copied().unwrap_or("");
        if is_iso_date(w) || is_dotted_date(w) {
            return Some("date");
        }
        if is_document_number(w, next) || w.starts_with('§') && w.len() > 1 || (*w == "§" && starts_digit(next)) {
            return Some("document-number");
        }
        if is_task_index(w) {
            return Some("task-index");
        }
        if is_audit_tag(w, next) {
            return Some("audit-tag");
        }
        if is_phase_index(w, next) {
            return Some("phase-index");
        }
    }
    None
}

fn starts_digit(w: &str) -> bool {
    w.bytes().next().is_some_and(|b| b.is_ascii_digit())
}

fn all_digits(w: &str) -> bool {
    !w.is_empty() && w.bytes().all(|b| b.is_ascii_digit())
}

fn is_iso_date(w: &str) -> bool {
    let b = w.as_bytes();
    b.len() == 10
        && b[4] == b'-'
        && b[7] == b'-'
        && [0, 1, 2, 3, 5, 6, 8, 9].iter().all(|&i| b[i].is_ascii_digit())
}

fn is_dotted_date(w: &str) -> bool {
    let b = w.as_bytes();
    b.len() == 10
        && b[2] == b'.'
        && b[5] == b'.'
        && [0, 1, 3, 4, 6, 7, 8, 9].iter().all(|&i| b[i].is_ascii_digit())
}

fn is_document_number(w: &str, next: &str) -> bool {
    let lower = w.to_ascii_lowercase();
    for prefix in ["docs/", "doc-", "docs-", "doc/"] {
        if let Some(rest) = lower.strip_prefix(prefix) {
            let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
            return !digits.is_empty() && digits.len() <= 3;
        }
    }
    (lower == "doc" || lower == "docs") && all_digits(next) && next.len() <= 3
}

/// `FIX-GATE-1`, `PLACE-POOL-1`, `R2-07`, `PR-3.5`: uppercase segments
/// joined by dashes, the last one a number. `UTF-8` and `SHA-256` are
/// terms (the list above), not indexes.
fn is_task_index(w: &str) -> bool {
    let segs: Vec<&str> = w.split('-').collect();
    if segs.len() < 2 {
        return false;
    }
    let head = segs[0];
    let head_ok = head.bytes().next().is_some_and(|b| b.is_ascii_uppercase())
        && head.bytes().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit());
    if !head_ok {
        return false;
    }
    let letters: String = head.chars().take_while(char::is_ascii_alphabetic).collect();
    if TERMS.contains(&letters.as_str()) {
        return false;
    }
    let last = segs[segs.len() - 1];
    let last_ok = match last.split_once('.') {
        Some((a, b)) => all_digits(a) && all_digits(b),
        None => all_digits(last),
    };
    if !last_ok {
        return false;
    }
    segs[1..segs.len() - 1]
        .iter()
        .all(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit()))
}

fn is_letter_number(w: &str, letter_max: usize) -> bool {
    let letters: String = w.chars().take_while(char::is_ascii_uppercase).collect();
    !letters.is_empty()
        && letters.len() <= letter_max
        && w.len() > letters.len()
        && all_digits(&w[letters.len()..])
}

fn is_audit_tag(w: &str, next: &str) -> bool {
    let lower = w.to_ascii_lowercase();
    let next_lower = next.to_ascii_lowercase();
    // "R3 audit", "audit P1", "lens 23", "H4 R2", "(A4)" as a bare "A4".
    (is_letter_number(w, 1) && w.starts_with('R') && next_lower == "audit")
        || (lower == "audit" && is_letter_number(next, 1) && next.starts_with('P'))
        || (lower == "lens" && all_digits(next))
        || (is_letter_number(w, 1) && is_letter_number(next, 1) && next.starts_with('R') && !w.starts_with('R'))
}

fn is_phase_index(w: &str, next: &str) -> bool {
    let lower = w.to_ascii_lowercase();
    let number = next.strip_prefix('#').unwrap_or(next);
    if matches!(lower.as_str(), "mig" | "migration" | "task" | "phase" | "wave" | "batch")
        && all_digits(number)
    {
        return true;
    }
    // "W2b": a wave with a letter suffix; "B4.4": a batch with a dot.
    let b = w.as_bytes();
    if b.len() >= 3 && b[0] == b'W' && b[1..b.len() - 1].iter().all(u8::is_ascii_digit) && b[b.len() - 1].is_ascii_lowercase() {
        return true;
    }
    if let Some(rest) = w.strip_prefix('B')
        && let Some((a, c)) = rest.split_once('.')
        && all_digits(a)
        && all_digits(c)
    {
        return true;
    }
    false
}

fn has_cyrillic(s: &str) -> bool {
    s.chars().any(|c| ('\u{0400}'..='\u{04FF}').contains(&c))
}

/// The comment part of one line, or None: after the first `//` (or `#`)
/// that sits outside a string literal - an even count of unescaped
/// double quotes before it.
pub fn comment_part(line: &str, hashy: bool) -> Option<&str> {
    let marker = if hashy { "#" } else { "//" };
    let mut from = 0usize;
    while let Some(rel) = line.get(from..)?.find(marker) {
        let pos = from + rel;
        let before = line.get(..pos).unwrap_or("");
        let quotes = before.matches('"').count().saturating_sub(before.matches("\\\"").count());
        if quotes % 2 == 0 {
            // A shebang and a `#[attribute]` are not comments.
            if hashy && (line.starts_with("#!") || line.trim_start().starts_with("#[")) {
                return None;
            }
            return line.get(pos..);
        }
        from = pos + marker.len();
    }
    None
}

fn walk(dir: &Path, exts: &[&str], out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        let name = e.file_name().to_string_lossy().into_owned();
        if p.is_dir() {
            if SKIP_DIRS.contains(&name.as_str()) || name.starts_with('.') {
                continue;
            }
            walk(&p, exts, out);
        } else if exts
            .iter()
            .any(|x| p.extension().is_some_and(|e| e.to_string_lossy() == *x))
        {
            out.push(p);
        }
    }
}

/// One offending line: `path<TAB>comment text` (content-keyed - line
/// numbers shift on every edit) plus the rule it broke.
pub struct Hit {
    pub sig: String,
    pub rule: &'static str,
    pub line: usize,
}

/// Every offending comment line under the scan roots.
pub fn scan(root: &Path, scans: &[Scan<'_>]) -> Vec<Hit> {
    let mut files: Vec<(PathBuf, bool)> = Vec::new();
    for s in scans {
        let mut found = Vec::new();
        walk(&root.join(s.dir), s.exts, &mut found);
        files.extend(found.into_iter().map(|f| (f, s.hashy)));
    }
    files.sort();
    files.dedup();
    let mut hits = Vec::new();
    for (path, hashy) in files {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        for (i, line) in text.lines().enumerate() {
            let Some(comment) = comment_part(line, hashy) else {
                continue;
            };
            let sig = format!("{rel}\t{}", comment.trim());
            if has_cyrillic(comment) && !line.contains("comment-language:allow") {
                hits.push(Hit { sig: sig.clone(), rule: "language", line: i + 1 });
            }
            if !line.contains("comment-kitchen:allow")
                && let Some(shape) = kitchen_shape(comment)
            {
                hits.push(Hit { sig, rule: shape, line: i + 1 });
            }
        }
    }
    hits
}

fn load_baseline(path: &Path) -> HashMap<String, usize> {
    let mut m = HashMap::new();
    if let Ok(text) = std::fs::read_to_string(path) {
        for l in text.lines().filter(|l| !l.is_empty()) {
            *m.entry(l.to_owned()).or_insert(0) += 1;
        }
    }
    m
}

fn count(hits: &[Hit]) -> HashMap<String, usize> {
    let mut m = HashMap::new();
    for h in hits {
        *m.entry(h.sig.clone()).or_insert(0) += 1;
    }
    m
}

/// The gate. `update`: rewrite the baseline, refusing to grow it.
/// Returns the violation lines (empty = green); `Err` when the update
/// was refused or could not be written.
pub fn run(root: &Path, scans: &[Scan<'_>], baseline: &Path, update: bool) -> Result<Vec<String>, String> {
    let hits = scan(root, scans);
    if hits.is_empty() && scans.iter().all(|s| !root.join(s.dir).exists()) {
        return Err("the comment gate collected zero files - refusing to pass on nothing".to_owned());
    }
    let current = count(&hits);
    let existing = load_baseline(baseline);
    let mut new: Vec<&Hit> = Vec::new();
    let mut seen: HashMap<&str, usize> = HashMap::new();
    for h in &hits {
        let n = seen.entry(h.sig.as_str()).or_insert(0);
        *n += 1;
        if *n > existing.get(&h.sig).copied().unwrap_or(0) {
            new.push(h);
        }
    }
    if update {
        if !existing.is_empty() && !new.is_empty() {
            let mut lines: Vec<String> = new
                .iter()
                .take(20)
                .map(|h| format!("REFUSING to ratchet UP [{}]  {}", h.rule, h.sig.replace('\t', "  ")))
                .collect();
            lines.push(format!(
                "the comment baseline is SHRINK-ONLY: {} line(s) exceed it - rewrite the comment as plain prose (say WHY; drop the index, the date, the document number), do not baseline it",
                new.len()
            ));
            return Err(lines.join("\n"));
        }
        let mut sigs: Vec<&String> = current.keys().collect();
        sigs.sort();
        let mut body = String::new();
        for s in sigs {
            for _ in 0..current[s] {
                body.push_str(s);
                body.push('\n');
            }
        }
        if let Some(parent) = baseline.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(baseline, body).map_err(|e| format!("cannot write {}: {e}", baseline.display()))?;
        let before: usize = existing.values().sum();
        eprintln!(
            "comment baseline updated (shrink-only): {} known line(s) recorded, {} removed",
            hits.len(),
            before.saturating_sub(hits.len())
        );
        return Ok(Vec::new());
    }
    let mut out: Vec<String> = new
        .iter()
        .map(|h| {
            let (path, text) = h.sig.split_once('\t').unwrap_or((h.sig.as_str(), ""));
            format!("[comments:{}] {path}:{}: {text}", h.rule, h.line)
        })
        .collect();
    if !out.is_empty() {
        out.push(format!(
            "comments: {} NEW line(s) - a comment says WHY in plain English, as a person would write it (no task index, date, document number or audit tag; Cyrillic only as quoted data with comment-language:allow); test identifiers live in the test function name only",
            new.len()
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bookkeeping_shapes_are_named_and_prose_passes() {
        for (text, want) in [
            ("// the tile follows the kind picked on the card", None),
            ("// UTF-8 and SHA-256 are terms, not indexes", None),
            ("// a P95 latency of 3 ms is fine; CUDA-13 too", None),
            ("// docs/21 P8: the pool doctor", Some("document-number")),
            ("// see doc-12 §7", Some("document-number")),
            ("// FIX-GATE-1 pinned this", Some("task-index")),
            ("// PLACE-POOL-1: worker A refuses", Some("task-index")),
            ("// per R2-07 the resubmit answers", Some("task-index")),
            ("// measured on 2026-09-07", Some("date")),
            ("// снято 07.09.2026", Some("date")),
            ("// R3 audit P1: the write result was discarded", Some("audit-tag")),
            ("// lens 23 R8 wanted this", Some("audit-tag")),
            ("// H4 R2: the queue bookkeeping", Some("audit-tag")),
            ("// mig 971 demoted ten families", Some("phase-index")),
            ("// the W2b carrier law", Some("phase-index")),
            ("// batch 4.4 is B4.4 here", Some("phase-index")),
        ] {
            assert_eq!(kitchen_shape(text), want, "{text}");
        }
    }

    #[test]
    fn comment_part_skips_strings_shebangs_and_attributes() {
        assert_eq!(comment_part("let s = \"http://x\"; // real", false), Some("// real"));
        assert_eq!(comment_part("let s = \"a // b\";", false), None);
        assert_eq!(comment_part("#!/bin/sh", true), None);
        assert_eq!(comment_part("#[derive(Debug)]", true), None);
        assert_eq!(comment_part("x=1  # note", true), Some("# note"));
    }
}
