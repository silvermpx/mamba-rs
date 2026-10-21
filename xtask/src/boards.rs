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
//! card. Test modules in production code are not read, since a test may
//! stand in a board of any size. The integration tests and the
//! qualification tools are read for one shape only: a check that the live
//! card is one particular board (`device.multiprocessor_count() != 170`,
//! `assert_eq!(device.multiprocessor_count(), 142)`), which would keep a
//! test or a tournament from running on the rest of its architecture. The
//! stock that predates the rule is a shrink-only baseline; the repair of
//! those places empties it.

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

/// The code of every line of a source file, with comments and the contents
/// of string and character literals blanked, so prose, messages and the
/// CUDA source a string carries never trip the rule or skew the brace
/// count. Literals and block comments may span lines; the scan follows them
/// across the whole file.
pub fn code_lines(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut lines = vec![String::new()];
    let mut i = 0;
    let push = |lines: &mut Vec<String>, c: char| {
        if c == '\n' {
            lines.push(String::new());
        } else if let Some(line) = lines.last_mut() {
            line.push(c);
        }
    };
    let newlines = |lines: &mut Vec<String>, from: usize, to: usize, chars: &[char]| {
        for &c in &chars[from..to.min(chars.len())] {
            if c == '\n' {
                lines.push(String::new());
            }
        }
    };
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        let prev_ident = i > 0 && is_ident_char(chars[i - 1]);
        if c == '/' && next == Some('/') {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if c == '/' && next == Some('*') {
            let start = i;
            let mut depth = 0usize;
            while i < chars.len() {
                if chars[i] == '/' && chars.get(i + 1) == Some(&'*') {
                    depth += 1;
                    i += 2;
                } else if chars[i] == '*' && chars.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    i += 1;
                }
            }
            newlines(&mut lines, start, i, &chars);
            continue;
        }
        // A raw string: r"..", r#".."#, br"..".
        if !prev_ident && (c == 'r' || (c == 'b' && next == Some('r'))) {
            let mut j = if c == 'b' { i + 2 } else { i + 1 };
            let mut hashes = 0usize;
            while chars.get(j) == Some(&'#') {
                hashes += 1;
                j += 1;
            }
            if chars.get(j) == Some(&'"') {
                let start = j + 1;
                let mut k = start;
                loop {
                    if k >= chars.len() {
                        break;
                    }
                    if chars[k] == '"' && (0..hashes).all(|h| chars.get(k + 1 + h) == Some(&'#')) {
                        break;
                    }
                    k += 1;
                }
                push(&mut lines, '"');
                newlines(&mut lines, start, k, &chars);
                push(&mut lines, '"');
                i = k + 1 + hashes;
                continue;
            }
        }
        if c == '"' {
            let start = i + 1;
            let mut k = start;
            while k < chars.len() && chars[k] != '"' {
                if chars[k] == '\\' {
                    k += 1;
                }
                k += 1;
            }
            push(&mut lines, '"');
            newlines(&mut lines, start, k, &chars);
            push(&mut lines, '"');
            i = k + 1;
            continue;
        }
        // A character literal ('{', '\''); a lifetime ('a) stays code.
        if c == '\'' {
            if next == Some('\\') {
                let mut k = i + 2;
                while k < chars.len() && chars[k] != '\'' {
                    k += 1;
                }
                push(&mut lines, ' ');
                i = k + 1;
                continue;
            }
            if chars.get(i + 2) == Some(&'\'') {
                push(&mut lines, ' ');
                i += 3;
                continue;
            }
        }
        push(&mut lines, c);
        i += 1;
    }
    lines
}

/// The code of one line, for the rule's own tests.
#[cfg(test)]
fn code_part(line: &str) -> String {
    code_lines(line).into_iter().next().unwrap_or_default()
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

/// Which shapes a scan root refuses. Production code refuses all of them.
/// Tests and qualification tools may stand in a board of any size (a field
/// or constant set to a count describes the board a host test simulates),
/// but a check that the live card is one particular board keeps a test or a
/// tournament from running on the rest of its architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    Production,
    LiveChecks,
}

/// The shape of a per-card pin on this code line, if it carries one.
pub fn pin_shape(code: &str, scope: Scope) -> Option<&'static str> {
    let t = tokens(code);
    let asserts = t.iter().any(|tok| tok == "assert_eq" || tok == "assert_ne");
    for (i, tok) in t.iter().enumerate() {
        // A method read (`device.multiprocessor_count()`) compares like a
        // field: look past its empty argument list.
        let call = t.get(i + 1).is_some_and(|x| x == "(") && t.get(i + 2).is_some_and(|x| x == ")");
        let skip = if call { 2 } else { 0 };
        let next = t.get(i + 1 + skip).map(String::as_str).unwrap_or("");
        let after = t.get(i + 2 + skip).map(String::as_str).unwrap_or("");
        let prev = if i > 0 { t[i - 1].as_str() } else { "" };
        let before = if i > 1 { t[i - 2].as_str() } else { "" };
        if is_count_name(tok) {
            if (next == "==" || next == "!=") && names_a_card(after) {
                return Some("count-compared");
            }
            if (prev == "==" || prev == "!=") && names_a_card(before) {
                return Some("count-compared");
            }
            if asserts && next == "," && names_a_card(after) {
                return Some("count-asserted");
            }
            // A field set to a literal: `name: 170`, but not a type
            // annotation (`name: u32`) or a path (`name::x`).
            if scope == Scope::Production
                && !call
                && next == ":"
                && names_a_card(after)
                && !tok.contains('.')
            {
                return Some("count-field");
            }
        }
        if scope == Scope::Production
            && (tok == "const" || tok == "static")
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
fn test_lines(lines: &[&str], code: &[String]) -> Vec<bool> {
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
            let line_code = code.get(j).map(String::as_str).unwrap_or("");
            for c in line_code.chars() {
                match c {
                    '{' => {
                        depth += 1;
                        opened = true;
                    }
                    '}' => depth -= 1,
                    _ => {}
                }
            }
            let ends_bare = !opened && line_code.trim_end().ends_with(';');
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

/// Every per-card pin under the scan roots, each root read in its scope.
pub fn scan(root: &Path, dirs: &[(&str, Scope)]) -> Vec<Hit> {
    let mut files = Vec::new();
    for (d, scope) in dirs {
        let mut found = Vec::new();
        walk(&root.join(d), &mut found);
        files.extend(found.into_iter().map(|path| (path, *scope)));
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    let mut hits = Vec::new();
    for (path, scope) in files {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        let lines: Vec<&str> = text.lines().collect();
        let code = code_lines(&text);
        let test = test_lines(&lines, &code);
        for (i, line) in lines.iter().enumerate() {
            if test[i] {
                continue;
            }
            let line_code = code.get(i).map(String::as_str).unwrap_or("");
            if let Some(shape) = pin_shape(line_code, scope) {
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
    dirs: &[(&str, Scope)],
    baseline: &Path,
    update: bool,
) -> Result<Vec<String>, String> {
    if dirs.iter().all(|(d, _)| !root.join(d).exists()) {
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
pub fn seed(root: &Path, dirs: &[(&str, Scope)], baseline: &Path) -> Result<Vec<String>, String> {
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
            (
                "if device.multiprocessor_count() != 170 {",
                Some("count-compared"),
            ),
            (
                "assert_eq!(device.multiprocessor_count(), 142);",
                Some("count-asserted"),
            ),
            ("if device.multiprocessors == 0 {", None),
            ("multiprocessor_count: u32,", None),
            ("let waves = tiles.div_ceil(device.multiprocessors);", None),
            ("multiprocessors: device.multiprocessors,", None),
            ("const TILE_ROWS: usize = 128;", None),
            ("// tuned on 170 SMs: multiprocessors == 170", None),
            ("let s = \"multiprocessors == 170\";", None),
        ] {
            assert_eq!(
                pin_shape(&code_part(line), Scope::Production),
                want,
                "{line}"
            );
        }
    }

    #[test]
    fn literals_spanning_lines_keep_their_braces_out_of_the_count() {
        let src = "#[cfg(test)]\nmod tests {\n    const SRC: &str = \"{\n  ret;\n}}\";\n    const C: char = '{';\n    const R: &str = r#\"}\"#;\n}\nconst SMS: u32 = 170;";
        let lines: Vec<&str> = src.lines().collect();
        let code = code_lines(src);
        assert_eq!(code.len(), lines.len());
        let test = test_lines(&lines, &code);
        assert_eq!(
            test,
            vec![true, true, true, true, true, true, true, true, false]
        );
        assert_eq!(
            pin_shape(&code[8], Scope::Production),
            Some("count-constant")
        );
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
        let code = code_lines(&src.join("\n"));
        let test = test_lines(&src, &code);
        assert_eq!(
            test,
            vec![false, true, true, true, true, false, true, true, false]
        );
    }

    #[test]
    fn live_check_roots_refuse_board_checks_but_not_simulated_boards() {
        for (line, want) in [
            (
                "assert_eq!(device.multiprocessor_count(), 170);",
                Some("count-asserted"),
            ),
            (
                "if identity.multiprocessor_count != 142 {",
                Some("count-compared"),
            ),
            ("multiprocessor_count: 188,", None),
            ("const BLACKWELL_SMS: usize = 170;", None),
        ] {
            assert_eq!(
                pin_shape(&code_part(line), Scope::LiveChecks),
                want,
                "{line}"
            );
        }
    }
}
