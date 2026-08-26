//! Lane census: qual/lanes.toml and tests/ must agree.
//!
//! Cargo has one selection bit (#[ignore]) and the qualification story
//! needs three lanes (gate / contract / record) plus a host lane for
//! suites that need no GPU. The manifest carries the judgment; this
//! census - which itself needs no GPU, so CI runs it - makes drift
//! impossible: a new test file must take a lane in the same commit, a
//! deleted file must leave the manifest, and a gate-lane file may not
//! quietly grow an #[ignore] arm that then never runs anywhere.

use std::collections::BTreeMap;
use std::path::Path;

fn manifest() -> BTreeMap<String, String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(root.join("qual/lanes.toml")).expect("qual/lanes.toml");
    let mut lanes = BTreeMap::new();
    let mut in_table = false;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        if line == "[lanes]" {
            in_table = true;
            continue;
        }
        if line.starts_with('[') {
            in_table = false;
            continue;
        }
        if !in_table {
            continue;
        }
        let (name, lane) = line.split_once('=').expect("name = lane row");
        let clean = |s: &str| s.trim().trim_matches('"').to_string();
        let prev = lanes.insert(clean(name), clean(lane));
        assert!(prev.is_none(), "duplicate manifest row for {}", clean(name));
    }
    lanes
}

#[test]
fn every_suite_has_exactly_one_lane_and_gate_files_hide_nothing() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let lanes = manifest();
    let known = ["gate", "contract", "record", "host"];
    for (name, lane) in &lanes {
        assert!(
            known.contains(&lane.as_str()),
            "{name}: unknown lane {lane:?}"
        );
    }

    let mut seen = Vec::new();
    for entry in std::fs::read_dir(root.join("tests")).expect("tests dir") {
        let path = entry.expect("entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let text = std::fs::read_to_string(&path).expect("test source");
        if !text.contains("#[test]") {
            continue; // shared harness modules live under tests/common
        }
        let lane = lanes.get(&name).unwrap_or_else(|| {
            panic!("tests/{name}.rs has no lane in qual/lanes.toml - assign one in this commit")
        });
        if lane == "gate" {
            assert!(
                !text.contains("#[ignore"),
                "tests/{name}.rs is in the gate lane but contains #[ignore] \
                 tests that no lane would ever run - move the file to the \
                 contract lane or un-ignore the tests"
            );
        }
        seen.push(name);
    }
    for name in lanes.keys() {
        assert!(
            seen.contains(name),
            "qual/lanes.toml names {name} but tests/{name}.rs does not exist - \
             remove the stale row"
        );
    }
}
