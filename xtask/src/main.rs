//! The repository gate. One binary, run the same way by `make ci-fast`,
//! the pre-push hook and CI, so a human, a hook and a runner grade the
//! identical thing.
//!
//! Rules today: `comments` - a code comment is plain English prose that
//! says WHY (no Cyrillic, no task indexes, dates, document numbers or
//! audit tags), ratcheted against a shrink-only baseline.

mod comments;

use std::path::{Path, PathBuf};

/// What the comment gate reads: the crate, its tests and examples, the
/// CUDA kernels, this gate, and the shell scripts.
const COMMENT_SCANS: &[comments::Scan<'static>] = &[
    comments::Scan {
        dir: "src",
        exts: &["rs"],
        hashy: false,
    },
    comments::Scan {
        dir: "tests",
        exts: &["rs"],
        hashy: false,
    },
    comments::Scan {
        dir: "examples",
        exts: &["rs"],
        hashy: false,
    },
    comments::Scan {
        dir: "kernels",
        exts: &["cu", "cuh"],
        hashy: false,
    },
    comments::Scan {
        dir: "xtask",
        exts: &["rs"],
        hashy: false,
    },
    comments::Scan {
        dir: "qual",
        exts: &["sh"],
        hashy: true,
    },
];
const COMMENT_BASELINE: &str = "tools/comment_kitchen_baseline.txt";

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let update = args.iter().any(|a| a == "--update-baseline");
    let verb = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .map(String::as_str)
        .unwrap_or("gate");
    let root = workspace_root();
    let result = match verb {
        "gate" | "comments" => {
            comments::run(&root, COMMENT_SCANS, &root.join(COMMENT_BASELINE), update)
        }
        other => Err(format!(
            "unknown rule: {other}\nusage: cargo run -q -p xtask -- [gate | comments] [--update-baseline]"
        )),
    };
    match result {
        Ok(v) if v.is_empty() => {
            println!("xtask gate: all rules green");
            std::process::ExitCode::SUCCESS
        }
        Ok(v) => {
            eprintln!("xtask gate: {} violation(s)", v.len().saturating_sub(1));
            for line in v {
                eprintln!("  {line}");
            }
            std::process::ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("{e}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn workspace_root() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_owned());
    Path::new(&manifest)
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}
