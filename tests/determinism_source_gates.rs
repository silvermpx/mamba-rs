//! Host-side source gates for the determinism laws that nothing else
//! enforces mechanically. No GPU, no cuda feature - CI runs these.

mod common;

use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

/// The one law with no other enforcement: no numeric atomics anywhere
/// in the kernel tree. Every reduction is a fixed-order tree or a
/// single-owner store; an atomic would make the result depend on the
/// scheduler and silently break run-to-run bit identity.
#[test]
fn kernels_contain_no_numeric_atomics() {
    let dir = root().join("kernels");
    let mut offenders = Vec::new();
    let mut stack = vec![dir];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).expect("kernel dir") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
                continue;
            };
            if ext != "cu" && ext != "cuh" {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("kernel source");
            let stripped = common::source_scan::strip_comments_lines(&text);
            for (i, (raw, line)) in text.lines().zip(&stripped).enumerate() {
                let hit = ["atomicAdd", "atomicCAS", "atomicExch", "atomicMin", "atomicMax"]
                    .iter()
                    .any(|p| line.contains(p))
                    || line.contains("atom.")
                    // The PTX reduction opcode; a plain substring would
                    // also match "shared." so require a boundary.
                    || line
                        .char_indices()
                        .filter(|(_, _)| line.contains("red."))
                        .any(|(j, _)| {
                            line[j..].starts_with("red.")
                                && (j == 0
                                    || !line.as_bytes()[j - 1].is_ascii_alphanumeric())
                        });
                if hit {
                    offenders.push(format!(
                        "{}:{}: {}",
                        path.strip_prefix(root()).unwrap().display(),
                        i + 1,
                        raw.trim()
                    ));
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "numeric atomic or reduction opcode in the kernel tree - the \
         determinism law forbids scheduler-ordered arithmetic:\n{}",
        offenders.join("\n")
    );
}

/// The NVRTC option builders must never grow fast-math flags: they
/// flush denormals and swap exact operations for approximations, which
/// changes bits per target and per toolchain.
#[test]
fn option_builders_contain_no_fast_math() {
    let mut offenders = Vec::new();
    let mut stack = vec![root().join("src")];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).expect("src dir") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("source");
            let stripped = common::source_scan::strip_comments_lines(&text);
            for (i, (raw, line)) in text.lines().zip(&stripped).enumerate() {
                if [
                    "--use_fast_math",
                    "--ftz=true",
                    "-prec-div=false",
                    "-prec-sqrt=false",
                ]
                .iter()
                .any(|p| line.contains(p))
                {
                    offenders.push(format!(
                        "{}:{}: {}",
                        path.strip_prefix(root()).unwrap().display(),
                        i + 1,
                        raw.trim()
                    ));
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "fast-math flag in an option builder:\n{}",
        offenders.join("\n")
    );
}

/// Every `cargo test --test <name>` command quoted in the public docs
/// must name a suite that exists: a whole release once shipped with
/// every repro command stale after a rename.
#[test]
fn doc_commands_name_real_suites() {
    let mut offenders = Vec::new();
    let mut doc_paths: Vec<PathBuf> = vec![root().join("README.md"), root().join("CHANGELOG.md")];
    let mut stack = vec![root().join("docs")];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for entry in entries {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|e| e.to_str()) == Some("md") {
                doc_paths.push(path);
            }
        }
    }
    for path in doc_paths {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        for (i, line) in text.lines().enumerate() {
            let mut rest = line;
            while let Some(pos) = rest.find("--test ") {
                rest = &rest[pos + "--test ".len()..];
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                if name.is_empty() {
                    continue;
                }
                if !root().join(format!("tests/{name}.rs")).exists() {
                    offenders.push(format!(
                        "{}:{}: --test {name} (no tests/{name}.rs)",
                        path.strip_prefix(root()).unwrap().display(),
                        i + 1,
                    ));
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "stale test names in docs:\n{}",
        offenders.join("\n")
    );
}
