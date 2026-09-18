//! Lane census: `qual/lanes.toml` and the targets declared in `Cargo.toml`
//! must agree.
//!
//! Cargo has one selection bit (`#[ignore]`) and the qualification story
//! needs more: gate / contract / record / host suites under `tests/`,
//! benchmarks under `benches/` with their own `main`, and qualification
//! instruments under `tools/qualification/` behind a non-default feature.
//! The lane table carries the judgment; this census, which needs no GPU so
//! CI runs it, makes drift impossible: every declared target takes a lane
//! in the same commit, a removed target leaves the table, a gate-lane file
//! may not quietly grow an `#[ignore]` arm that then never runs anywhere,
//! and nothing in the published target graph points into the archive.

use std::collections::BTreeMap;
use std::path::Path;

const KNOWN_LANES: [&str; 7] = [
    "gate",
    "contract",
    "record",
    "host",
    "toolkit",
    "bench",
    "qualification",
];

/// A test or bench target as `Cargo.toml` declares it. The manifest is read
/// with the same narrow line reader as the lane table: `[[test]]` and
/// `[[bench]]` blocks whose `name`, `path`, `harness` and
/// `required-features` lines are plain `key = value` rows.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Target {
    kind: String,
    name: String,
    path: String,
    harness: bool,
    required_features: Vec<String>,
}

fn unquote(s: &str) -> String {
    s.trim().trim_matches('"').to_string()
}

fn parse_lanes(text: &str) -> BTreeMap<String, String> {
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
        let prev = lanes.insert(unquote(name), unquote(lane));
        assert!(
            prev.is_none(),
            "duplicate manifest row for {}",
            unquote(name)
        );
    }
    lanes
}

fn parse_targets(manifest: &str) -> Vec<Target> {
    let mut targets = Vec::new();
    let mut current: Option<Target> = None;
    // `cargo package` normalizes the manifest and may write a multi-element
    // array over several lines; join such a row before reading it.
    let mut open_row = String::new();
    for raw in manifest.lines() {
        let line = raw.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let joined;
        let line = if !open_row.is_empty() {
            open_row.push(' ');
            open_row.push_str(line);
            if !line.contains(']') {
                continue;
            }
            joined = std::mem::take(&mut open_row);
            joined.as_str()
        } else if line.contains("= [") && !line.contains(']') {
            open_row.push_str(line);
            continue;
        } else {
            line
        };
        if line.starts_with('[') {
            if let Some(target) = current.take() {
                targets.push(target);
            }
            if line == "[[test]]" || line == "[[bench]]" {
                current = Some(Target {
                    kind: line.trim_matches(['[', ']']).to_string(),
                    name: String::new(),
                    path: String::new(),
                    harness: true,
                    required_features: Vec::new(),
                });
            }
            continue;
        }
        let Some(target) = current.as_mut() else {
            continue;
        };
        let (key, value) = line.split_once('=').expect("key = value row");
        match key.trim() {
            "name" => target.name = unquote(value),
            "path" => target.path = unquote(value),
            "harness" => target.harness = value.trim() != "false",
            "required-features" => {
                target.required_features = value
                    .trim()
                    .trim_matches(['[', ']'])
                    .split(',')
                    .map(unquote)
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            _ => {}
        }
    }
    if let Some(target) = current.take() {
        targets.push(target);
    }
    targets
}

/// The checks the census applies to one manifest, one lane table and the
/// sources the targets name. `read` resolves a target path to its text so the
/// same rules run against inline fixtures below.
fn check(
    manifest: &str,
    lanes: &BTreeMap<String, String>,
    read: &dyn Fn(&str) -> Option<String>,
) -> Result<(), String> {
    let targets = parse_targets(manifest);
    if targets.is_empty() {
        return Err("Cargo.toml declares no [[test]] or [[bench]] target".into());
    }
    for (name, lane) in lanes {
        if !KNOWN_LANES.contains(&lane.as_str()) {
            return Err(format!("{name}: unknown lane {lane:?}"));
        }
    }
    let mut names = Vec::new();
    let mut paths = Vec::new();
    for target in &targets {
        if target.name.is_empty() || target.path.is_empty() {
            return Err(format!("{:?} target without a name or path", target.kind));
        }
        if names.contains(&target.name) {
            return Err(format!("{} is declared twice", target.name));
        }
        if paths.contains(&target.path) {
            return Err(format!("{} is declared under two names", target.path));
        }
        if target.path.starts_with("internal/") {
            return Err(format!(
                "{} points into the archive ({}); archived stands stay out of the manifest",
                target.name, target.path
            ));
        }
        let Some(text) = read(&target.path) else {
            return Err(format!("{}: {} does not exist", target.name, target.path));
        };
        let Some(lane) = lanes.get(&target.name) else {
            return Err(format!(
                "{} ({}) has no lane in qual/lanes.toml - assign one in this commit",
                target.name, target.path
            ));
        };
        match lane.as_str() {
            "gate" | "toolkit" => {
                for line in text.lines() {
                    let line = line.trim();
                    if line.starts_with("#[ignore") && !line.contains("= \"invoked by") {
                        return Err(format!(
                            "{} is in the {lane} lane but contains {line} - a test no lane would \
                             ever run; move the file to the contract lane, un-ignore the test, \
                             or mark a child a parent launches `#[ignore = \"invoked by ...\"]`",
                            target.path
                        ));
                    }
                }
            }
            "bench" => {
                if target.kind != "bench" || target.harness {
                    return Err(format!(
                        "{} is in the bench lane but is not a harness = false [[bench]]",
                        target.name
                    ));
                }
                if !target.path.starts_with("benches/") {
                    return Err(format!(
                        "{} is in the bench lane outside benches/",
                        target.name
                    ));
                }
            }
            "qualification" => {
                if !target.path.starts_with("tools/qualification/") {
                    return Err(format!(
                        "{} is in the qualification lane outside tools/qualification/",
                        target.name
                    ));
                }
                if !target
                    .required_features
                    .iter()
                    .any(|f| f == "qualification")
                {
                    return Err(format!(
                        "{} is a qualification instrument without the qualification feature",
                        target.name
                    ));
                }
            }
            _ => {
                if !target.path.starts_with("tests/") {
                    return Err(format!(
                        "{} is in the {lane} lane but lives outside tests/",
                        target.name
                    ));
                }
            }
        }
        if target.kind == "test"
            && target
                .required_features
                .iter()
                .any(|f| f == "qualification")
            && lane != "qualification"
        {
            return Err(format!(
                "{} requires the qualification feature but sits in the {lane} lane",
                target.name
            ));
        }
        names.push(target.name.clone());
        paths.push(target.path.clone());
    }
    for name in lanes.keys() {
        if !names.contains(name) {
            return Err(format!(
                "qual/lanes.toml lists {name} but Cargo.toml declares no such target - \
                 drop the row or declare the target"
            ));
        }
    }
    Ok(())
}

#[test]
fn every_declared_target_has_exactly_one_lane_and_gate_files_hide_nothing() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).expect("Cargo.toml");
    let lanes = parse_lanes(
        &std::fs::read_to_string(root.join("qual/lanes.toml")).expect("qual/lanes.toml"),
    );
    let read = |path: &str| std::fs::read_to_string(root.join(path)).ok();
    if let Err(message) = check(&manifest, &lanes, &read) {
        panic!("{message}");
    }
}

/// A source file under tests/ that is not a declared target is dead weight:
/// either declare it (with a lane) or move it to the archive.
#[test]
fn every_test_source_file_is_a_declared_target_or_a_shared_module() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).expect("Cargo.toml");
    let declared: Vec<String> = parse_targets(&manifest)
        .into_iter()
        .map(|t| t.path)
        .collect();
    for dir in ["tests", "benches", "tools/qualification"] {
        for entry in std::fs::read_dir(root.join(dir)).expect("target dir") {
            let path = entry.expect("entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let relative = path
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            assert!(
                declared.contains(&relative),
                "{relative} is not a declared target - declare it in Cargo.toml with a lane \
                 in qual/lanes.toml, or move it to internal/experiments"
            );
        }
    }
}

#[cfg(test)]
mod fixtures {
    use super::*;

    const MANIFEST: &str = r#"
[package]
name = "x"

[[test]]
name = "alpha"
path = "tests/alpha.rs"

[[bench]]
name = "speed"
path = "benches/speed.rs"
harness = false

[[test]]
name = "census_tool"
path = "tools/qualification/census_tool.rs"
required-features = ["qualification", "cuda"]
"#;

    fn lanes(rows: &str) -> BTreeMap<String, String> {
        parse_lanes(&format!("[lanes]\n{rows}"))
    }

    fn sources(alpha: &str) -> impl Fn(&str) -> Option<String> + '_ {
        move |path: &str| match path {
            "tests/alpha.rs" => Some(alpha.to_string()),
            "benches/speed.rs" => Some("fn main() {}".to_string()),
            "tools/qualification/census_tool.rs" => Some("#[test] fn t() {}".to_string()),
            _ => None,
        }
    }

    const CLEAN: &str = r#""alpha" = "gate"
"speed" = "bench"
"census_tool" = "qualification"
"#;

    #[test]
    fn multi_line_feature_arrays_parse_like_single_line_ones() {
        let normalized = "[[test]]\nname = \"x\"\npath = \"tests/x.rs\"\nrequired-features = [\n    \"qualification\",\n    \"cuda\",\n]\n";
        let targets = parse_targets(normalized);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].required_features, ["qualification", "cuda"]);
    }

    #[test]
    fn a_consistent_layout_passes() {
        let read = sources("#[test] fn a() {}");
        check(MANIFEST, &lanes(CLEAN), &read).unwrap();
    }

    #[test]
    fn a_parent_invoked_child_is_allowed_in_the_gate_lane() {
        let read = sources("#[test]\n#[ignore = \"invoked by the parent\"]\nfn child() {}");
        check(MANIFEST, &lanes(CLEAN), &read).unwrap();
    }

    #[test]
    fn a_hidden_ignored_arm_fails_the_gate_lane() {
        let read = sources("#[test]\n#[ignore]\nfn slow() {}");
        let message = check(MANIFEST, &lanes(CLEAN), &read).unwrap_err();
        assert!(
            message.contains("gate lane but contains #[ignore]"),
            "{message}"
        );
    }

    #[test]
    fn a_missing_lane_fails() {
        let read = sources("#[test] fn a() {}");
        let message = check(
            MANIFEST,
            &lanes("\"alpha\" = \"gate\"\n\"speed\" = \"bench\"\n"),
            &read,
        )
        .unwrap_err();
        assert!(
            message.contains("census_tool") && message.contains("no lane"),
            "{message}"
        );
    }

    #[test]
    fn an_unknown_lane_fails() {
        let read = sources("#[test] fn a() {}");
        let rows = CLEAN.replace("\"gate\"", "\"weekly\"");
        let message = check(MANIFEST, &lanes(&rows), &read).unwrap_err();
        assert!(message.contains("unknown lane"), "{message}");
    }

    #[test]
    fn a_stale_row_fails() {
        let read = sources("#[test] fn a() {}");
        let rows = format!("{CLEAN}\"ghost\" = \"host\"\n");
        let message = check(MANIFEST, &lanes(&rows), &read).unwrap_err();
        assert!(
            message.contains("ghost") && message.contains("no such target"),
            "{message}"
        );
    }

    #[test]
    fn a_duplicate_declaration_fails() {
        let read = sources("#[test] fn a() {}");
        let manifest =
            format!("{MANIFEST}\n[[test]]\nname = \"alpha\"\npath = \"tests/alpha2.rs\"\n");
        let message = check(&manifest, &lanes(CLEAN), &read).unwrap_err();
        assert!(message.contains("declared twice"), "{message}");
    }

    #[test]
    fn a_duplicate_lane_row_panics() {
        let caught =
            std::panic::catch_unwind(|| lanes("\"alpha\" = \"gate\"\n\"alpha\" = \"host\"\n"));
        assert!(caught.is_err());
    }

    #[test]
    fn an_archive_path_fails() {
        let read = sources("#[test] fn a() {}");
        let manifest = MANIFEST.replace("tests/alpha.rs", "internal/experiments/alpha.rs");
        let message = check(&manifest, &lanes(CLEAN), &read).unwrap_err();
        assert!(message.contains("points into the archive"), "{message}");
    }

    #[test]
    fn a_qualification_tool_without_its_feature_fails() {
        let read = sources("#[test] fn a() {}");
        let manifest = MANIFEST.replace("[\"qualification\", \"cuda\"]", "[\"cuda\"]");
        let message = check(&manifest, &lanes(CLEAN), &read).unwrap_err();
        assert!(
            message.contains("without the qualification feature"),
            "{message}"
        );
    }

    #[test]
    fn a_bench_lane_entry_must_be_a_harness_free_bench() {
        let read = sources("#[test] fn a() {}");
        let manifest = MANIFEST.replace("harness = false\n", "");
        let message = check(&manifest, &lanes(CLEAN), &read).unwrap_err();
        assert!(message.contains("harness = false"), "{message}");
    }
}
