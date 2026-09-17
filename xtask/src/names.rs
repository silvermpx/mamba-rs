//! No identifier carries a version stamp. A name says what a thing is;
//! `V1`, `V2`, `_v3` say only that someone once meant to replace it and
//! never did, and when two of them coexist the number hides the actual
//! difference. This rule is zero-and-stay-zero: there is no baseline to
//! ratchet, every hit fails the gate.
//!
//! What it reads: every identifier-shaped token in Rust and CUDA source
//! outside comments and string literals. Kernel symbols are covered where
//! they are defined, in the CUDA sources; a quoted name in Rust has to
//! match a definition or the module would not load. Foreign API names keep
//! their own spelling: the CUDA driver, cuBLAS, cuDNN and NVRTC entry
//! points carry `_v2` suffixes NVIDIA chose, and those are not ours to
//! rename.

use std::path::{Path, PathBuf};

/// A version stamp anywhere in an identifier: `_v<digits>` as a whole
/// snake-case segment (`nn_tile_v1`, `tf32_v2_m64n64`), or a camel-case
/// `V<digits>` word after a lowercase letter or digit (`ScalarFmaV1`,
/// `PolicyV4`), or `_V<digits>` closing a screaming-case name
/// (`CUBLAS_POLICY_V2`).
pub fn has_version_stamp(ident: &str) -> bool {
    let b = ident.as_bytes();
    let n = b.len();
    let mut i = 0;
    while i < n {
        if (b[i] == b'v' || b[i] == b'V') && i + 1 < n && b[i + 1].is_ascii_digit() {
            let mut j = i + 1;
            while j < n && b[j].is_ascii_digit() {
                j += 1;
            }
            let at_end = j == n;
            let next_is_sep = j < n && b[j] == b'_';
            let prev_is_sep = i > 0 && b[i - 1] == b'_';
            let prev_is_word =
                i > 0 && (b[i - 1].is_ascii_lowercase() || b[i - 1].is_ascii_digit());
            let next_is_camel = j < n && b[j].is_ascii_uppercase();
            let snake = prev_is_sep && (at_end || next_is_sep);
            let camel = b[i] == b'V' && prev_is_word && (at_end || next_is_camel);
            if snake || camel {
                return true;
            }
            i = j;
            continue;
        }
        i += 1;
    }
    false
}

/// Entry points and types named by a vendor, not by this crate: the
/// CUDA driver's `cuX_v2` functions and its `CUDA_X_v2` structs, cuBLAS,
/// cuDNN, NVRTC and NVML.
fn foreign(ident: &str) -> bool {
    let vendor = ["cu", "cublas", "cudnn", "nvrtc", "nvml"];
    vendor.iter().any(|p| {
        ident.starts_with(p)
            && ident.len() > p.len()
            && ident.as_bytes()[p.len()].is_ascii_uppercase()
    }) || ident.starts_with("CUDA_")
        || ident.starts_with("CU_")
}

/// Identifier tokens of one source line with comments and string
/// literals blanked, so prose about an old name cannot trip the rule.
fn identifiers(line: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let b = line.as_bytes();
    let mut i = 0;
    let mut in_str: Option<u8> = None;
    while i < b.len() {
        let c = b[i];
        if let Some(q) = in_str {
            if c == b'\\' {
                i += 2;
                continue;
            }
            if c == q {
                in_str = None;
                i += 1;
                continue;
            }
            i += 1;
            continue;
        }
        if c == b'"' || c == b'\'' {
            in_str = Some(c);
            i += 1;
            continue;
        }
        if c == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            break;
        }
        if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            out.push(&line[start..i]);
            continue;
        }
        i += 1;
    }
    out
}

fn walk(dir: &Path, exts: &[&str], out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        let name = e.file_name().to_string_lossy().into_owned();
        if p.is_dir() {
            if name == "target" || name.starts_with('.') {
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

/// Every versioned identifier under the roots, as `path:line  name`.
pub fn run(root: &Path, dirs: &[(&str, &[&str])]) -> Result<Vec<String>, String> {
    let mut files = Vec::new();
    for (dir, exts) in dirs {
        walk(&root.join(dir), exts, &mut files);
    }
    if files.is_empty() {
        return Err(
            "the naming gate collected zero files - refusing to pass on nothing".to_owned(),
        );
    }
    files.sort();
    let mut hits = Vec::new();
    let mut block_comment = false;
    for path in files {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        block_comment = false;
        for (n, line) in text.lines().enumerate() {
            let mut visible = line;
            if block_comment {
                match visible.find("*/") {
                    Some(end) => {
                        block_comment = false;
                        visible = &visible[end + 2..];
                    }
                    None => continue,
                }
            }
            if let Some(start) = visible.find("/*") {
                if !visible[start..].contains("*/") {
                    block_comment = true;
                }
                visible = &visible[..start];
            }
            for ident in identifiers(visible) {
                if has_version_stamp(ident) && !foreign(ident) {
                    let rel = path.strip_prefix(root).unwrap_or(&path).display();
                    hits.push(format!("{rel}:{}  {ident}", n + 1));
                }
            }
        }
    }
    let _ = block_comment;
    if !hits.is_empty() {
        hits.push(format!(
            "{} version-stamped name(s): a name says what a thing is, not which attempt it was - drop the stamp, or name the difference when two coexist",
            hits.len()
        ));
    }
    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stamped(base: &str, tail: &str) -> String {
        format!("{base}{tail}")
    }

    #[test]
    fn stamp_shapes() {
        for yes in [
            stamped("ExactScalarFma", "V1"),
            stamped("nn_", "v1"),
            stamped("CUBLAS_POLICY_", "V2"),
            stamped("nn_tf32_", "v1_m64n64_bk32_s2"),
            stamped("nn_sm89_tc128_s3_", "v1_bf16"),
            stamped("Sm120TmaFma", "V1Fixed"),
        ] {
            assert!(has_version_stamp(&yes), "{yes}");
        }
        for no in [
            "ExactScalarFma",
            "Tf32",
            "sm120",
            "V1",
            "v1",
            "TV1",
            "Sm89N64",
            "bk16_s2",
            "F32",
            "Uint32",
            "x86_64",
            "half_ulp",
            "vec2",
            "regpipe_vec2_lane",
            "conv1d",
            "Sm90aWgmma",
        ] {
            assert!(!has_version_stamp(no), "{no}");
        }
    }

    #[test]
    fn vendor_names_are_theirs() {
        for f in [
            "cuMemcpyDtoDAsync_v2",
            "cublasSetWorkspace_v2",
            "cuGraphGetEdges_v2",
            "CUDA_KERNEL_NODE_PARAMS_v2",
        ] {
            assert!(foreign(f), "{f}");
        }
        assert!(!foreign("cublas_policy_v2"));
        assert!(!foreign("CUBLAS_POLICY_V2"));
    }

    #[test]
    fn comments_and_strings_are_blind_and_identifiers_are_not() {
        let comment = format!("let x = 1; // ScalarFma{}", "V1");
        assert!(identifiers(&comment).iter().all(|i| !has_version_stamp(i)));
        let quoted = format!("let s = \"nn_tf32_{}_m64n64\";", "v1");
        assert!(identifiers(&quoted).iter().all(|i| !has_version_stamp(i)));
        let bare = format!("let x = ScalarFma{};", "V1");
        assert!(identifiers(&bare).iter().any(|i| has_version_stamp(i)));
    }
}
