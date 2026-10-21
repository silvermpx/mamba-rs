//! No kernel choice is keyed to one card. Kernels are built per
//! architecture: SM80 is the base block, each architecture adds its own
//! fast paths, and inside an architecture every card reaches them by
//! itself, whatever its SM count (docs/architecture-dispatch.md). A
//! literal SM count in production code is a choice only one board can
//! take, so this rule refuses three shapes of it:
//!
//! - an SM or multiprocessor count compared for equality with a literal
//!   (`multiprocessors == 170`, `sm_count != 142`);
//! - a struct field of that name set to a literal (`multiprocessor_count:
//!   188`), which is how a frozen per-board identity is written;
//! - a constant or static named for SM counts and set to a literal
//!   (`BLACKWELL_SMS = 170`, a dispatch tuning constant pinned to a card).
//!
//! Zero and one stay allowed: they are the "unknown device" sentinel, not a
//! card. Test modules are not read, since a test may stand in a board of
//! any size. The stock that predates the rule is a shrink-only baseline;
//! the repair of those places empties it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// The field and binding names that hold a card's SM count.
const COUNT_NAMES: &[&str] = &[
    "multiprocessors",
    "multiprocessor_count",
    "sm_count",
    "num_sms",
    "sms",
];

/// Words that mark a constant as an SM count.
const CONST_WORDS: &[&str] = &["SMS", "SM_COUNT", "MULTIPROCESSOR", "SM_TUNING"];

/// Directories never walked.
const SKIP_DIRS: &[&str] = &["target", ".git", "vendor"];

/// The value of an integer literal token (`170`, `1_024`, `142u32`), if it
/// is one.
fn literal_value(token: &str) -> Option<u64> {
    let digits: String = token
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '_')
        .filter(|c| *c != '_')
        .collect();
    if digits.is_empty() {
        return None;
    }
    let rest = &token[token
        .char_indices()
        .find(|(_, c)| !(c.is_ascii_digit() || *c == '_'))
        .map_or(token.len(), |(i, _)| i)..];
    let suffix_ok = rest.is_empty()
        || ["u8", "u16", "u32", "u64", "usize", "i32", "i64", "isize"].contains(&rest);
    if !suffix_ok {
        return None;
    }
    digits.parse().ok()
}

/// A literal that names a card: two or more. Zero and one are sentinels.
fn names_a_card(token: &str) -> bool {
    literal_value(token).is_some_and(|v| v >= 2)
}

/// The source line with its `//` comment and the contents of its string
/// literals blanked, so prose and messages about a card never trip the rule.
pub fn code_part(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    let mut in_str = false;
    while let Some(c) = chars.next() {
        if in_str {
            if c == '\\' {
                chars.next();
                continue;
            }
            if c == '"' {
                in_str = false;
                out.push('"');
            }
            continue;
        }
        if c == '"' {
            in_str = true;
            out.push('"');
            continue;
        }
        if c == '/' && chars.peek() == Some(&'/') {
            break;
        }
        out.push(c);
    }
    out
}

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// The tokens of a code line: identifier paths (`device.multiprocessors`),
/// literals and single punctuation, with `==`, `!=` and `::` kept whole.
fn tokens(code: &str) -> Vec<String> {
    let mut out = Vec::new();
    let b: Vec<char> = code.chars().collect();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if is_ident_char(c) {
            let start = i;
            while i < b.len() && (is_ident_char(b[i]) || b[i] == '.') {
                i += 1;
            }
            out.push(b[start..i].iter().collect());
            continue;
        }
        if i + 1 < b.len() {
            let pair: String = b[i..i + 2].iter().collect();
            if pair == "==" || pair == "!=" || pair == "::" {
                out.push(pair);
                i += 2;
                continue;
            }
        }
        out.push(c.to_string());
        i += 1;
    }
    out
}

/// The last segment of an identifier path.
fn last_segment(token: &str) -> &str {
    token.rsplit('.').next().unwrap_or(token)
}

fn is_count_name(token: &str) -> bool {
    COUNT_NAMES.contains(&last_segment(token))
}

/// The shape of a per-card pin on this code line, if it carries one.
pub fn pin_shape(code: &str) -> Option<&'static str> {
    let t = tokens(code);
    for (i, tok) in t.iter().enumerate() {
        let next = t.get(i + 1).map(String::as_str).unwrap_or("");
        let after = t.get(i + 2).map(String::as_str).unwrap_or("");
        let prev = if i > 0 { t[i - 1].as_str() } else { "" };
        let before = if i > 1 { t[i - 2].as_str() } else { "" };
        if is_count_name(tok) {
            if (next == "==" || next == "!=") && names_a_card(after) {
                return Some("count-compared");
            }
            if (prev == "==" || prev == "!=") && names_a_card(before) {
                return Some("count-compared");
            }
            // A field set to a literal: `name: 170`, but not a type
            // annotation (`name: u32`) or a path (`name::x`).
            if next == ":" && names_a_card(after) && !tok.contains('.') {
                return Some("count-field");
            }
        }
        if (tok == "const" || tok == "static")
            && let Some(name) = t.get(i + 1)
            && CONST_WORDS.iter().any(|w| name.contains(w))
            && let Some(eq) = t.iter().skip(i + 2).position(|x| x == "=")
            && let Some(value) = t.get(i + 2 + eq + 1)
            && names_a_card(value)
        {
            return Some("count-constant");
        }
    }
    None
}

/// The lines of a file that belong to test code: an item under
/// `#[cfg(test)]` (a test module, helper or impl), from its first line to
/// the brace that closes it.
fn test_lines(lines: &[&str]) -> Vec<bool> {
    let mut test = vec![false; lines.len()];
    let mut i = 0;
    while i < lines.len() {
        let trimmed = lines[i].trim_start();
        let opens_test =
            trimmed.starts_with("#[cfg(test)]") || trimmed.starts_with("#[cfg(all(test");
        if !opens_test {
            i += 1;
            continue;
        }
        let mut depth: i64 = 0;
        let mut opened = false;
        let mut j = i;
        while j < lines.len() {
            test[j] = true;
            let code = code_part(lines[j]);
            for c in code.chars() {
                match c {
                    '{' => {
                        depth += 1;
                        opened = true;
                    }
                    '}' => depth -= 1,
                    _ => {}
                }
            }
            let ends_bare = !opened && code.trim_end().ends_with(';');
            if (opened && depth <= 0) || ends_bare {
                break;
            }
            j += 1;
        }
        i = j + 1;
    }
    test
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
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
            walk(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs")
            && !name.ends_with("_tests.rs")
            && name != "tests.rs"
        {
            out.push(p);
        }
    }
}

/// One offending line: `path<TAB>code` (content-keyed, so the baseline
/// survives edits elsewhere in the file) plus where and what it is.
pub struct Hit {
    pub sig: String,
    pub shape: &'static str,
    pub line: usize,
}

/// Every per-card pin in the production code under the scan roots.
pub fn scan(root: &Path, dirs: &[&str]) -> Vec<Hit> {
    let mut files = Vec::new();
    for d in dirs {
        walk(&root.join(d), &mut files);
    }
    files.sort();
    let mut hits = Vec::new();
    for path in files {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        let lines: Vec<&str> = text.lines().collect();
        let test = test_lines(&lines);
        for (i, line) in lines.iter().enumerate() {
            if test[i] {
                continue;
            }
            if let Some(shape) = pin_shape(&code_part(line)) {
                hits.push(Hit {
                    sig: format!("{rel}\t{}", line.trim()),
                    shape,
                    line: i + 1,
                });
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

/// The gate. `update` rewrites the baseline and refuses to grow it.
/// Returns the violation lines (empty = green).
pub fn run(
    root: &Path,
    dirs: &[&str],
    baseline: &Path,
    update: bool,
) -> Result<Vec<String>, String> {
    if dirs.iter().all(|d| !root.join(d).exists()) {
        return Err("the board gate found none of its scan roots".to_owned());
    }
    let hits = scan(root, dirs);
    let existing = load_baseline(baseline);
    let mut seen: HashMap<&str, usize> = HashMap::new();
    let new: Vec<&Hit> = hits
        .iter()
        .filter(|h| {
            let n = seen.entry(h.sig.as_str()).or_insert(0);
            *n += 1;
            *n > existing.get(&h.sig).copied().unwrap_or(0)
        })
        .collect();
    if update {
        if !new.is_empty() {
            let mut lines: Vec<String> = new
                .iter()
                .map(|h| {
                    format!(
                        "REFUSING to baseline [{}]  {}",
                        h.shape,
                        h.sig.replace('\t', "  ")
                    )
                })
                .collect();
            lines.push(format!(
                "the board baseline is SHRINK-ONLY: {} new pin(s) - derive the choice from the architecture and the live device, never from one card's SM count",
                new.len()
            ));
            return Err(lines.join("\n"));
        }
        let mut body: Vec<&str> = hits.iter().map(|h| h.sig.as_str()).collect();
        body.sort_unstable();
        let mut text = body.join("\n");
        if !text.is_empty() {
            text.push('\n');
        }
        std::fs::write(baseline, text)
            .map_err(|e| format!("cannot write {}: {e}", baseline.display()))?;
        let before: usize = existing.values().sum();
        eprintln!(
            "board baseline updated (shrink-only): {} pin(s) left, {} removed",
            hits.len(),
            before.saturating_sub(hits.len())
        );
        return Ok(Vec::new());
    }
    let mut out: Vec<String> = new
        .iter()
        .map(|h| {
            let (path, code) = h.sig.split_once('\t').unwrap_or((h.sig.as_str(), ""));
            format!("[boards:{}] {path}:{}: {code}", h.shape, h.line)
        })
        .collect();
    if !out.is_empty() {
        out.push(format!(
            "boards: {} NEW per-card pin(s) - kernels are chosen per architecture and every card of an architecture reaches its fast routes by itself; read the SM count from the live device, never compare it to one card's number (docs/architecture-dispatch.md)",
            new.len()
        ));
    }
    Ok(out)
}

/// Builds the first baseline when none exists yet. Unlike `--update-baseline`
/// it takes the whole current stock; it refuses once a baseline is there.
pub fn seed(root: &Path, dirs: &[&str], baseline: &Path) -> Result<Vec<String>, String> {
    if baseline.exists() {
        return Err(format!(
            "{} exists - the board baseline is seeded once and then only shrinks",
            baseline.display()
        ));
    }
    let hits = scan(root, dirs);
    let mut body: Vec<&str> = hits.iter().map(|h| h.sig.as_str()).collect();
    body.sort_unstable();
    let mut text = body.join("\n");
    if !text.is_empty() {
        text.push('\n');
    }
    std::fs::write(baseline, text)
        .map_err(|e| format!("cannot write {}: {e}", baseline.display()))?;
    eprintln!("board baseline seeded with {} pin(s)", hits.len());
    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_card_pins_are_named_and_live_reads_pass() {
        for (line, want) in [
            ("if device.multiprocessors == 170 {", Some("count-compared")),
            ("|| device.multiprocessors != 170", Some("count-compared")),
            ("&& 142 == ctx.sm_count", Some("count-compared")),
            ("multiprocessor_count: 188,", Some("count-field")),
            ("sm_count: 142u32,", Some("count-field")),
            (
                "pub(super) const DISPATCH_SM_TUNING: u32 = 170;",
                Some("count-constant"),
            ),
            ("const BLACKWELL_SMS: usize = 170;", Some("count-constant")),
            ("if device.multiprocessors == 0 {", None),
            ("multiprocessor_count: u32,", None),
            ("let waves = tiles.div_ceil(device.multiprocessors);", None),
            ("multiprocessors: device.multiprocessors,", None),
            ("const TILE_ROWS: usize = 128;", None),
            ("// tuned on 170 SMs: multiprocessors == 170", None),
            ("let s = \"multiprocessors == 170\";", None),
        ] {
            assert_eq!(pin_shape(&code_part(line)), want, "{line}");
        }
    }

    #[test]
    fn test_items_are_skipped_to_their_closing_brace() {
        let src = [
            "fn live() {}",
            "#[cfg(test)]",
            "mod tests {",
            "    fn board() { let multiprocessors = 1; }",
            "}",
            "fn after() {}",
            "#[cfg(test)]",
            "use std::fmt;",
            "fn last() {}",
        ];
        let test = test_lines(&src);
        assert_eq!(
            test,
            vec![false, true, true, true, true, false, true, true, false]
        );
    }
}
