//! Shared helper for host-side source scans: comment stripping that
//! survives multi-line block comments, so a scan never flags (or
//! misses) code because of where a comment happens to wrap.

/// Return the source split into lines with every comment removed:
/// line comments, single-line blocks, and multi-line blocks whose
/// state carries across lines.
pub fn strip_comments_lines(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_block = false;
    for line in text.lines() {
        let mut clean = String::new();
        let mut rest = line;
        loop {
            if in_block {
                match rest.find("*/") {
                    Some(j) => {
                        rest = &rest[j + 2..];
                        in_block = false;
                    }
                    None => break,
                }
            } else if let Some(i) = rest.find("/*") {
                let before = &rest[..i];
                if let Some(sl) = before.find("//") {
                    clean.push_str(&before[..sl]);
                    break;
                }
                clean.push_str(before);
                rest = &rest[i + 2..];
                in_block = true;
            } else {
                match rest.find("//") {
                    Some(sl) => clean.push_str(&rest[..sl]),
                    None => clean.push_str(rest),
                }
                break;
            }
        }
        out.push(clean);
    }
    out
}
