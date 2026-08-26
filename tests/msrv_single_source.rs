//! The minimum supported Rust version has one source of truth: the
//! `rust-version` field in Cargo.toml. Every other place that spells it
//! - the CI toolchain pins, the release workflow, the README - must
//! agree, or the published claim silently drifts from what is tested.

use std::path::Path;

#[test]
fn msrv_is_spelled_identically_everywhere() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).expect("Cargo.toml");
    let msrv = manifest
        .lines()
        .find_map(|l| {
            l.strip_prefix("rust-version = \"")
                .and_then(|r| r.strip_suffix('"'))
        })
        .expect("rust-version in Cargo.toml");

    let mut offenders = Vec::new();
    for rel in [".github/workflows/ci.yml", ".github/workflows/release.yml"] {
        let text = std::fs::read_to_string(root.join(rel)).expect(rel);
        for (i, line) in text.lines().enumerate() {
            let t = line.trim();
            let Some(pin) = t
                .strip_prefix("toolchain: \"")
                .and_then(|r| r.strip_suffix('"'))
            else {
                continue;
            };
            // Only numeric pins are MSRV claims; channel names are not.
            if pin.chars().next().is_some_and(|c| c.is_ascii_digit()) && pin != msrv {
                offenders.push(format!("{rel}:{}: toolchain {pin} != {msrv}", i + 1));
            }
        }
    }
    let readme = std::fs::read_to_string(root.join("README.md")).expect("README.md");
    if !readme.contains(&format!("MSRV {msrv}")) {
        offenders.push(format!("README.md: no `MSRV {msrv}` mention"));
    }
    assert!(
        offenders.is_empty(),
        "MSRV copies disagree with Cargo.toml's rust-version = {msrv}:\n{}",
        offenders.join("\n")
    );
}
