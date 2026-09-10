use super::*;
use std::collections::BTreeMap;

const EXACT_LITERALS: [&str; 8] = [
    "hot_a:0", "hot_a:1", "hot_b:0", "hot_b:1", "hot_d:0", "hot_d:1", "hot_e:0", "hot_e:1",
];
const TF32_LITERALS: [&str; 2] = ["hot_c:0", "hot_c:1"];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Family {
    Exact,
    Tf32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RunMode {
    Task7,
    PostAuto44,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Stage {
    Smoke1,
    Screen21,
    Confirm101,
    PostSmoke1,
    Post101,
}

#[derive(Debug, Eq, PartialEq)]
struct Config {
    mode: RunMode,
    family: Family,
    stage: Stage,
    windows: usize,
    toolkit: String,
    literals: Vec<String>,
    source_sha: String,
    binary_sha: String,
    screen_sha: Option<String>,
    screen_artifact_sha: Option<String>,
    jsonl: String,
}

fn parse_config(environment: &BTreeMap<String, String>) -> Result<Config, String> {
    const ALLOWED: [&str; 16] = [
        "MAMBA_FIXED_ADA_TOOLKIT_ADMISSION",
        "MAMBA_FIXED_ADA_VENDOR",
        "MAMBA_FIXED_ADA_ROWS",
        "MAMBA_FIXED_ADA_LITERALS",
        "MAMBA_FIXED_ADA_WINDOWS",
        "MAMBA_FIXED_ADA_TOOLKIT",
        "MAMBA_FIXED_ADA_TUNING_REVISION",
        "MAMBA_FIXED_ADA_STAGE",
        "MAMBA_FIXED_ADA_SOURCE_SHA",
        "MAMBA_FIXED_ADA_BINARY_SHA",
        "MAMBA_FIXED_ADA_SCREEN_SHA",
        "MAMBA_FIXED_ADA_SCREEN_ARTIFACT_SHA",
        "MAMBA_FIXED_ADA_JSONL",
        "MAMBA_FIXED_VENDOR_TILES",
        "MAMBA_FIXED_VENDOR_PATHS",
        "MAMBA_FIXED_VENDOR_EXACT_CC",
    ];
    for key in environment.keys() {
        let scoped = key.starts_with("MAMBA_FIXED_ADA_")
            || key.starts_with("MAMBA_FIXED_VENDOR_")
            || matches!(
                key.as_str(),
                "MAMBA_FIXED_HALF_TILE_CANDIDATE"
                    | "MAMBA_FIXED_AUTO_VENDOR_ROW"
                    | "MAMBA_FIXED_AUTO_VENDOR_CELL"
                    | "MAMBA_FIXED_AUTO_VENDOR_BIAS"
                    | "NVIDIA_TF32_OVERRIDE"
            );
        if scoped && !ALLOWED.contains(&key.as_str()) {
            return Err(format!("stale or foreign Task7 control {key}"));
        }
    }
    let required = |key: &str| {
        environment
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| format!("missing {key}"))
    };
    if required("MAMBA_FIXED_ADA_TOOLKIT_ADMISSION")? != "1" {
        return Err("MAMBA_FIXED_ADA_TOOLKIT_ADMISSION must be exactly 1".into());
    }
    if required("MAMBA_FIXED_ADA_VENDOR")? != "1" {
        return Err("MAMBA_FIXED_ADA_VENDOR must be exactly 1".into());
    }
    if required("MAMBA_FIXED_VENDOR_PATHS")? != "eager,graph" {
        return Err("MAMBA_FIXED_VENDOR_PATHS must be exactly eager,graph".into());
    }
    if required("MAMBA_FIXED_VENDOR_EXACT_CC")? != "8.9" {
        return Err("MAMBA_FIXED_VENDOR_EXACT_CC must be exactly 8.9".into());
    }
    if required("MAMBA_FIXED_ADA_TUNING_REVISION")? != "43" {
        return Err("MAMBA_FIXED_ADA_TUNING_REVISION must be exactly 43".into());
    }
    let toolkit = required("MAMBA_FIXED_ADA_TOOLKIT")?;
    if !matches!(toolkit, "12.8" | "13.0") {
        return Err(format!("unsupported Task7 toolkit {toolkit:?}"));
    }
    let (family, inventory, expected_tile) = match required("MAMBA_FIXED_ADA_ROWS")? {
        "f32_exact_fast" => (Family::Exact, &EXACT_LITERALS[..], "F32Sm89N64CopyPlan"),
        "tf32" => (Family::Tf32, &TF32_LITERALS[..], "Tf32M64S2"),
        row => return Err(format!("unsupported Task7 row {row:?}")),
    };
    if required("MAMBA_FIXED_VENDOR_TILES")? != expected_tile {
        return Err(format!("wrong Task7 candidate for {family:?}"));
    }
    let (stage, windows) = match required("MAMBA_FIXED_ADA_STAGE")? {
        "smoke1" => (Stage::Smoke1, 1),
        "screen21" => (Stage::Screen21, 21),
        "confirm101" => (Stage::Confirm101, 101),
        value => return Err(format!("unknown Task7 stage {value:?}")),
    };
    if required("MAMBA_FIXED_ADA_WINDOWS")? != windows.to_string() {
        return Err(format!("stage/window mismatch for {stage:?}"));
    }
    let parse_sha = |key: &str| -> Result<String, String> {
        let value = required(key)?;
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(format!("{key} must be a lowercase SHA256"));
        }
        Ok(value.to_owned())
    };
    let source_sha = parse_sha("MAMBA_FIXED_ADA_SOURCE_SHA")?;
    let binary_sha = parse_sha("MAMBA_FIXED_ADA_BINARY_SHA")?;
    let jsonl = required("MAMBA_FIXED_ADA_JSONL")?;
    if !jsonl.starts_with('/') || !jsonl.ends_with(".jsonl") || jsonl.contains(char::is_whitespace)
    {
        return Err("MAMBA_FIXED_ADA_JSONL must be an absolute whitespace-free .jsonl path".into());
    }
    let literal_text = required("MAMBA_FIXED_ADA_LITERALS")?;
    if literal_text.is_empty() {
        return Err("MAMBA_FIXED_ADA_LITERALS must not be empty".into());
    }
    let mut literals = Vec::new();
    for literal in literal_text.split(',') {
        if !inventory.contains(&literal) {
            return Err(format!("foreign Task7 literal {literal:?}"));
        }
        if literals.iter().any(|seen| seen == literal) {
            return Err(format!("duplicate Task7 literal {literal:?}"));
        }
        literals.push(literal.to_owned());
    }
    let (screen_sha, screen_artifact_sha) = match stage {
        Stage::Smoke1 | Stage::Screen21 => {
            if literals
                .iter()
                .map(String::as_str)
                .ne(inventory.iter().copied())
            {
                return Err(format!(
                    "{stage:?} requires the complete canonical literal inventory"
                ));
            }
            if environment.contains_key("MAMBA_FIXED_ADA_SCREEN_SHA")
                || environment.contains_key("MAMBA_FIXED_ADA_SCREEN_ARTIFACT_SHA")
            {
                return Err(format!("{stage:?} must not bind a prior screen"));
            }
            (None, None)
        }
        Stage::Confirm101 => (
            Some(parse_sha("MAMBA_FIXED_ADA_SCREEN_SHA")?),
            Some(parse_sha("MAMBA_FIXED_ADA_SCREEN_ARTIFACT_SHA")?),
        ),
        Stage::PostSmoke1 | Stage::Post101 => unreachable!("Task7 parser produced Task8 stage"),
    };
    Ok(Config {
        mode: RunMode::Task7,
        family,
        stage,
        windows,
        toolkit: toolkit.to_owned(),
        literals,
        source_sha,
        binary_sha,
        screen_sha,
        screen_artifact_sha,
        jsonl: jsonl.to_owned(),
    })
}

fn parse_post_config(environment: &BTreeMap<String, String>) -> Result<Config, String> {
    const ALLOWED: [&str; 13] = [
        "MAMBA_FIXED_ADA_EXACT_POST_AUTO",
        "MAMBA_FIXED_ADA_VENDOR",
        "MAMBA_FIXED_ADA_EXACT_POST_STAGE",
        "MAMBA_FIXED_ADA_EXACT_POST_WINDOWS",
        "MAMBA_FIXED_ADA_EXACT_POST_TOOLKIT",
        "MAMBA_FIXED_ADA_EXACT_POST_LITERALS",
        "MAMBA_FIXED_ADA_EXACT_POST_TUNING_REVISION",
        "MAMBA_FIXED_ADA_EXACT_POST_SOURCE_SHA",
        "MAMBA_FIXED_ADA_EXACT_POST_BINARY_SHA",
        "MAMBA_FIXED_ADA_EXACT_POST_JSONL",
        "MAMBA_FIXED_VENDOR_TILES",
        "MAMBA_FIXED_VENDOR_PATHS",
        "MAMBA_FIXED_VENDOR_EXACT_CC",
    ];
    for key in environment.keys() {
        let scoped = key.starts_with("MAMBA_FIXED_ADA_")
            || key.starts_with("MAMBA_FIXED_VENDOR_")
            || matches!(
                key.as_str(),
                "MAMBA_FIXED_HALF_TILE_CANDIDATE"
                    | "MAMBA_FIXED_AUTO_VENDOR_ROW"
                    | "MAMBA_FIXED_AUTO_VENDOR_CELL"
                    | "MAMBA_FIXED_AUTO_VENDOR_BIAS"
                    | "NVIDIA_TF32_OVERRIDE"
            );
        if scoped && !ALLOWED.contains(&key.as_str()) {
            return Err(format!("stale or foreign Task8 control {key}"));
        }
    }
    let required = |key: &str| {
        environment
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| format!("missing {key}"))
    };
    for (key, expected) in [
        ("MAMBA_FIXED_ADA_EXACT_POST_AUTO", "1"),
        ("MAMBA_FIXED_ADA_VENDOR", "1"),
        ("MAMBA_FIXED_ADA_EXACT_POST_TUNING_REVISION", "44"),
        ("MAMBA_FIXED_VENDOR_TILES", "Legacy,F32Sm89N64CopyPlan"),
        ("MAMBA_FIXED_VENDOR_PATHS", "eager,graph"),
        ("MAMBA_FIXED_VENDOR_EXACT_CC", "8.9"),
    ] {
        if required(key)? != expected {
            return Err(format!("{key} must be exactly {expected}"));
        }
    }
    let toolkit = required("MAMBA_FIXED_ADA_EXACT_POST_TOOLKIT")?;
    if !matches!(toolkit, "12.8" | "13.0") {
        return Err(format!("unsupported Task8 toolkit {toolkit:?}"));
    }
    let (stage, windows) = match required("MAMBA_FIXED_ADA_EXACT_POST_STAGE")? {
        "smoke1" => (Stage::PostSmoke1, 1),
        "post101" => (Stage::Post101, 101),
        value => return Err(format!("unknown Task8 stage {value:?}")),
    };
    if required("MAMBA_FIXED_ADA_EXACT_POST_WINDOWS")? != windows.to_string() {
        return Err(format!("Task8 stage/window mismatch for {stage:?}"));
    }
    let parse_sha = |key: &str| -> Result<String, String> {
        let value = required(key)?;
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(format!("{key} must be a lowercase SHA256"));
        }
        Ok(value.to_owned())
    };
    let literal_text = required("MAMBA_FIXED_ADA_EXACT_POST_LITERALS")?;
    let literals = literal_text.split(',').collect::<Vec<_>>();
    if literals != EXACT_LITERALS {
        return Err(
            "Task8 post-AUTO requires the complete canonical exact literal inventory".into(),
        );
    }
    let jsonl = required("MAMBA_FIXED_ADA_EXACT_POST_JSONL")?;
    if !jsonl.starts_with('/') || !jsonl.ends_with(".jsonl") || jsonl.contains(char::is_whitespace)
    {
        return Err(
            "MAMBA_FIXED_ADA_EXACT_POST_JSONL must be an absolute whitespace-free .jsonl path"
                .into(),
        );
    }
    Ok(Config {
        mode: RunMode::PostAuto44,
        family: Family::Exact,
        stage,
        windows,
        toolkit: toolkit.to_owned(),
        literals: literals.into_iter().map(str::to_owned).collect(),
        source_sha: parse_sha("MAMBA_FIXED_ADA_EXACT_POST_SOURCE_SHA")?,
        binary_sha: parse_sha("MAMBA_FIXED_ADA_EXACT_POST_BINARY_SHA")?,
        screen_sha: None,
        screen_artifact_sha: None,
        jsonl: jsonl.to_owned(),
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Observation {
    window: usize,
    traversal: usize,
    comparison: usize,
    position: usize,
    arm: usize,
    us: f64,
}

fn schedule(window: usize, start_parity: usize) -> Result<Vec<Observation>, String> {
    if start_parity > 1 {
        return Err("start parity must be 0 or 1".into());
    }
    let reverse = (window + start_parity) % 2 == 1;
    let comparisons = if reverse { [2, 1, 0] } else { [0, 1, 2] };
    let mut observations = Vec::with_capacity(12);
    for (traversal, comparison) in comparisons.into_iter().enumerate() {
        let (a, b) = [(0, 1), (2, 0), (2, 1)][comparison];
        let arms = if reverse { [b, a, a, b] } else { [a, b, b, a] };
        for (position, arm) in arms.into_iter().enumerate() {
            observations.push(Observation {
                window,
                traversal,
                comparison,
                position,
                arm,
                us: 0.0,
            });
        }
    }
    Ok(observations)
}

fn ratio(observations: &[Observation]) -> Result<f64, String> {
    if observations.len() != 4 {
        return Err("mirrored bracket must have four observations".into());
    }
    let first = observations[0];
    if first.comparison > 2 || first.traversal > 2 {
        return Err("invalid comparison or traversal".into());
    }
    let (a, b) = [(0, 1), (2, 0), (2, 1)][first.comparison];
    let arms: Vec<_> = observations
        .iter()
        .map(|observation| observation.arm)
        .collect();
    if arms != [a, b, b, a] && arms != [b, a, a, b] {
        return Err("invalid ABBA/BAAB arms".into());
    }
    for (position, observation) in observations.iter().enumerate() {
        if observation.window != first.window
            || observation.traversal != first.traversal
            || observation.comparison != first.comparison
            || observation.position != position
            || !observation.us.is_finite()
            || observation.us <= 0.0
        {
            return Err("invalid mirrored raw observation".into());
        }
    }
    let sum = |arm| {
        observations
            .iter()
            .filter(|observation| observation.arm == arm)
            .map(|observation| observation.us)
            .sum::<f64>()
    };
    Ok(sum(b) / sum(a))
}

fn quantiles(ratios: &[f64]) -> Result<(f64, f64), String> {
    if ratios.is_empty()
        || ratios
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err("invalid ratio set".into());
    }
    let mut sorted = ratios.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = |fraction: f64| ((sorted.len() - 1) as f64 * fraction).round() as usize;
    Ok((sorted[index(0.5)], sorted[index(0.95)]))
}

fn literal_admitted(strata: &[(f64, f64)]) -> Result<bool, String> {
    if strata.len() != 4
        || strata
            .iter()
            .any(|(p50, p95)| !p50.is_finite() || *p50 <= 0.0 || !p95.is_finite() || *p95 <= 0.0)
    {
        return Err("literal requires four finite positive path/parity strata".into());
    }
    Ok(strata.iter().all(|(p50, p95)| *p50 < 1.0 && *p95 < 1.0))
}

#[derive(Clone, Debug, PartialEq)]
struct LaunchContract<'a> {
    symbol: &'a str,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    dynamic_shared: u32,
    pointers: [u64; 4],
    parameters: Vec<u32>,
    abi: Vec<(usize, usize)>,
    terminal_rejected: bool,
}

fn validate_launch(
    actual: &LaunchContract<'_>,
    expected: &LaunchContract<'_>,
) -> Result<(), String> {
    if actual != expected {
        return Err(format!(
            "physical launch differs: actual={actual:?} expected={expected:?}"
        ));
    }
    Ok(())
}

fn launch_json(actual: &LaunchContract<'_>) -> String {
    let abi = actual
        .abi
        .iter()
        .map(|(offset, size)| format!("[{offset},{size}]"))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"symbol\":\"{}\",\"grid\":[{},{},{}],\"block\":[{},{},{}],\"dynamic_shared\":{},\"pointers\":{:?},\"parameter_words\":{:?},\"driver_abi\":[{abi}],\"terminal_rejected\":{}}}",
        fixed_sm120_tf32_bd_json_escape(actual.symbol),
        actual.grid.0,
        actual.grid.1,
        actual.grid.2,
        actual.block.0,
        actual.block.1,
        actual.block.2,
        actual.dynamic_shared,
        actual.pointers,
        actual.parameters,
        actual.terminal_rejected,
    )
}

fn unchanged_words(saved: &[u32], observed: &[u32], label: &str) -> Result<(), String> {
    if saved != observed {
        return Err(format!("Task7 immutable {label} changed"));
    }
    Ok(())
}

fn validate_configuration(
    windows: usize,
    start_parity: usize,
    observations: &[Observation],
    pair_ratios: &[(usize, usize, f64)],
    summaries: &[(usize, f64, f64)],
    completed: bool,
) -> Result<[Vec<f64>; 3], String> {
    if !completed
        || observations.len() != 12 * windows
        || pair_ratios.len() != 3 * windows
        || summaries.len() != 3
    {
        return Err("incomplete configuration closure".into());
    }
    let mut expected = Vec::with_capacity(12 * windows);
    for window in 0..windows {
        expected.extend(schedule(window, start_parity)?);
    }
    for (actual, expected) in observations.iter().zip(&expected) {
        if (
            actual.window,
            actual.traversal,
            actual.comparison,
            actual.position,
            actual.arm,
        ) != (
            expected.window,
            expected.traversal,
            expected.comparison,
            expected.position,
            expected.arm,
        ) {
            return Err("raw chronology differs from mirrored schedule".into());
        }
    }
    let mut recomputed: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::with_capacity(windows));
    for (index, bracket) in observations.chunks_exact(4).enumerate() {
        let value = ratio(bracket)?;
        let (window, comparison, recorded) = pair_ratios[index];
        if (window, comparison) != (bracket[0].window, bracket[0].comparison)
            || (recorded - value).abs() > 1e-12
        {
            return Err("pair record differs from raw arithmetic".into());
        }
        recomputed[comparison].push(value);
    }
    for (expected_comparison, (comparison, p50, p95)) in summaries.iter().copied().enumerate() {
        let expected = quantiles(&recomputed[expected_comparison])?;
        if comparison != expected_comparison
            || (p50 - expected.0).abs() > 1e-12
            || (p95 - expected.1).abs() > 1e-12
        {
            return Err("summary differs from raw ratios".into());
        }
    }
    Ok(recomputed)
}

const SCHEMA: &str = "MambaBiFixedAdaToolkitAdmissionV1";
const POST_SCHEMA: &str = "MambaBiFixedAdaExactPostAutoV1";
const GUARD_WORDS: usize = 64;
const CANARY: u32 = 0x5a5a_5a5a;

impl RunMode {
    fn schema(self) -> &'static str {
        match self {
            Self::Task7 => SCHEMA,
            Self::PostAuto44 => POST_SCHEMA,
        }
    }

    fn tuning_revision(self) -> u16 {
        match self {
            Self::Task7 => 43,
            Self::PostAuto44 => 44,
        }
    }

    fn family_name(self, family: Family) -> &'static str {
        match self {
            Self::Task7 => family.name(),
            Self::PostAuto44 => "f32_exact_post_auto",
        }
    }

    fn arms(self) -> [&'static str; 3] {
        match self {
            Self::Task7 => ["actualAUTO", "candidate", "Fast"],
            Self::PostAuto44 => ["Legacy", "AUTO", "Fast"],
        }
    }

    fn directions(self) -> [&'static str; 3] {
        match self {
            Self::Task7 => ["candidate/AUTO", "AUTO/Fast", "candidate/Fast"],
            Self::PostAuto44 => ["AUTO/Legacy", "Legacy/Fast", "AUTO/Fast"],
        }
    }

    fn own_rule(self) -> &'static str {
        match self {
            Self::Task7 => "candidate/AUTO p50 and p95 < 1 in all four path/start strata",
            Self::PostAuto44 => "AUTO/Legacy p50 and p95 < 1 in all four path/start strata",
        }
    }
}

impl Family {
    fn name(self) -> &'static str {
        match self {
            Self::Exact => "f32_exact_fast",
            Self::Tf32 => "tf32",
        }
    }

    fn policy(self) -> F32TriadPolicy {
        match self {
            Self::Exact => F32TriadPolicy::ExactScalarFmaV1,
            Self::Tf32 => F32TriadPolicy::AllowDeterministicTf32V1,
        }
    }

    fn candidate(self) -> InferenceTile {
        match self {
            Self::Exact => InferenceTile::F32Sm89N64CopyPlan,
            Self::Tf32 => InferenceTile::Tf32M64S2,
        }
    }

    fn incumbent(self) -> InferenceTile {
        match self {
            Self::Exact => InferenceTile::Legacy,
            Self::Tf32 => InferenceTile::Tf32RnaM128N128S3,
        }
    }

    fn custom_tolerance(self) -> f64 {
        match self {
            Self::Exact => 0.0002,
            Self::Tf32 => 0.0025,
        }
    }
}

impl Stage {
    fn name(self) -> &'static str {
        match self {
            Self::Smoke1 => "smoke1",
            Self::Screen21 => "screen21",
            Self::Confirm101 => "confirm101",
            Self::PostSmoke1 => "smoke1",
            Self::Post101 => "post101",
        }
    }
}

struct Jsonl {
    schema: &'static str,
    writer: BufWriter<File>,
    digest: Sha256,
    lines: usize,
}

impl Jsonl {
    fn create(path: &str, schema: &'static str) -> Result<Self, String> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| format!("create new Task7 JSONL {path}: {error}"))?;
        Ok(Self {
            schema,
            writer: BufWriter::new(file),
            digest: Sha256::new(),
            lines: 0,
        })
    }

    fn emit(&mut self, fields: &str) -> Result<(), String> {
        let line = format!("{{\"schema\":\"{}\",{fields}}}\n", self.schema);
        self.writer
            .write_all(line.as_bytes())
            .map_err(|error| format!("write Task7 JSONL: {error}"))?;
        self.writer
            .flush()
            .map_err(|error| format!("flush Task7 JSONL: {error}"))?;
        self.digest.update(line.as_bytes());
        self.lines += 1;
        print!("{line}");
        Ok(())
    }

    fn complete(
        mut self,
        configurations: usize,
        windows: usize,
        literals: usize,
    ) -> Result<(), String> {
        let digest = format!("{:x}", self.digest.clone().finalize());
        self.emit(&format!(
            concat!(
                "\"kind\":\"complete\",\"configurations\":{},",
                "\"expected_configurations\":{},\"samples\":{},\"pairs\":{},",
                "\"summaries\":{},\"literals\":{},\"rejected\":0,",
                "\"preceding_lines\":{},\"preceding_jsonl_sha256\":\"{}\",",
                "\"all_gates_passed\":true,\"passed\":true"
            ),
            configurations,
            literals * 4,
            configurations * 12 * windows,
            configurations * 3 * windows,
            configurations * 3,
            literals,
            self.lines,
            digest,
        ))?;
        self.writer
            .get_ref()
            .sync_all()
            .map_err(|error| format!("sync Task7 JSONL: {error}"))
    }
}

struct Guarded {
    device: DtypedBuf,
    initial: Vec<u32>,
    active: usize,
}

impl Guarded {
    fn new(ctx: &GpuCtx, active: Vec<f32>) -> Result<Self, String> {
        let mut initial = vec![CANARY; active.len() + 2 * GUARD_WORDS];
        for (destination, source) in initial[GUARD_WORDS..GUARD_WORDS + active.len()]
            .iter_mut()
            .zip(active)
        {
            *destination = source.to_bits();
        }
        let device = DtypedBuf::zeros(&ctx.stream, initial.len(), WeightDtype::F32)?;
        let floats = initial
            .iter()
            .copied()
            .map(f32::from_bits)
            .collect::<Vec<_>>();
        device.upload_f32(&ctx.stream, &floats)?;
        Ok(Self {
            device,
            initial,
            active: floats.len() - 2 * GUARD_WORDS,
        })
    }

    fn ptr(&self) -> u64 {
        self.device.cached_ptr() + (GUARD_WORDS * 4) as u64
    }

    fn words(&self, ctx: &GpuCtx) -> Result<Vec<u32>, String> {
        let bytes = fixed_explicit_vendor_raw_bytes(ctx, &self.device);
        let words = bytes
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        if words[..GUARD_WORDS].iter().any(|word| *word != CANARY)
            || words[GUARD_WORDS + self.active..]
                .iter()
                .any(|word| *word != CANARY)
        {
            return Err("Task7 guarded allocation canary changed".into());
        }
        Ok(words)
    }

    fn read(&self, ctx: &GpuCtx) -> Result<Vec<u32>, String> {
        let words = self.words(ctx)?;
        Ok(words[GUARD_WORDS..GUARD_WORDS + self.active].to_vec())
    }

    fn unchanged(&self, ctx: &GpuCtx) -> Result<(), String> {
        unchanged_words(&self.initial, &self.words(ctx)?, "guarded input")
    }

    fn upload_words(&self, ctx: &GpuCtx, words: &[u32]) -> Result<(), String> {
        if words.len() != self.active {
            return Err("Task7 guarded upload length differs".into());
        }
        let result = unsafe {
            cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                self.ptr(),
                words.as_ptr().cast(),
                std::mem::size_of_val(words),
                ctx.stream.cu_stream(),
            )
        };
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(format!("Task7 guarded upload failed: {result:?}"));
        }
        ctx.stream
            .synchronize()
            .map_err(|error| format!("Task7 guarded upload sync: {error}"))
    }

    fn output_gate(
        &self,
        ctx: &GpuCtx,
        expected: &[u32],
        replay: impl FnOnce() -> Result<(), String>,
    ) -> Result<(), String> {
        let complement = expected.iter().map(|word| !word).collect::<Vec<_>>();
        self.upload_words(ctx, &complement)?;
        let observed = self.read(ctx)?;
        if observed != complement
            || observed
                .iter()
                .zip(expected)
                .any(|(actual, expected)| actual == expected)
        {
            return Err("Task7 complement poison upload/readback gate failed".into());
        }
        replay()?;
        if self.read(ctx)? != expected {
            return Err("Task7 replay did not overwrite every expected storage word".into());
        }
        Ok(())
    }
}

struct Case {
    mode: RunMode,
    family: Family,
    shape: InferenceShape,
    has_bias: bool,
    a: Guarded,
    b: Guarded,
    bias: Guarded,
    outputs: Vec<Guarded>,
    selected: Cell<Option<InferenceTile>>,
}

impl Case {
    fn new(
        ctx: &GpuCtx,
        mode: RunMode,
        family: Family,
        shape: InferenceShape,
        has_bias: bool,
    ) -> Result<Self, String> {
        let a = Guarded::new(ctx, synth(shape.m * shape.k, 0x0ada_a001))?;
        let b = Guarded::new(ctx, synth(shape.k * shape.n, 0x0ada_b001))?;
        let bias = Guarded::new(ctx, synth(shape.n, 0x0ada_b1a5))?;
        let outputs = (0..4)
            .map(|_| Guarded::new(ctx, vec![0.0; shape.m * shape.n]))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            mode,
            family,
            shape,
            has_bias,
            a,
            b,
            bias,
            outputs,
            selected: Cell::new(None),
        })
    }

    fn operands(&self, arm: usize) -> InferenceFwdOperands {
        InferenceFwdOperands {
            c: TypedPtr {
                ptr: self.outputs[arm].ptr(),
                dtype: WeightDtype::F32,
            },
            x: TypedPtr {
                ptr: self.a.ptr(),
                dtype: WeightDtype::F32,
            },
            w: TypedPtr {
                ptr: self.b.ptr(),
                dtype: WeightDtype::F32,
            },
            bias_ptr: self.has_bias.then(|| self.bias.ptr()),
        }
    }

    fn launch(&self, ctx: &GpuCtx, arm: usize) -> Result<(), String> {
        let operands = self.operands(arm);
        match (self.mode, arm) {
            (RunMode::Task7, 0) | (RunMode::PostAuto44, 1) => {
                let selected = launch_fixed_auto_vendor_custom(ctx, operands, self.shape);
                let expected = match self.mode {
                    RunMode::Task7 => self.family.incumbent(),
                    RunMode::PostAuto44 => InferenceTile::F32Sm89N64CopyPlan,
                };
                if selected != expected {
                    return Err(format!(
                        "actual AUTO {selected:?} differs from required {expected:?} for {:?}",
                        self.mode,
                    ));
                }
                if self
                    .selected
                    .get()
                    .is_some_and(|previous| previous != selected)
                {
                    return Err("actual AUTO enum drifted between launches".into());
                }
                self.selected.set(Some(selected));
                Ok(())
            }
            (RunMode::Task7, 1) => {
                inference_forward_with_tile(ctx, operands, self.shape, self.family.candidate())
            }
            (RunMode::PostAuto44, 0) => {
                inference_forward_with_tile(ctx, operands, self.shape, InferenceTile::Legacy)
            }
            (_, 2) => {
                fixed_ada_vendor_launch(
                    ctx,
                    operands,
                    self.shape,
                    cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                );
                Ok(())
            }
            (_, 3) => {
                fixed_ada_vendor_launch(
                    ctx,
                    operands,
                    self.shape,
                    cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC,
                );
                Ok(())
            }
            _ => Err("unknown toolkit admission arm".into()),
        }
    }

    fn inputs(&self, ctx: &GpuCtx) -> Result<(), String> {
        self.a.unchanged(ctx)?;
        self.b.unchanged(ctx)?;
        self.bias.unchanged(ctx)?;
        for arm in 0..4 {
            let operands = self.operands(arm);
            if operands.x.ptr != self.a.ptr()
                || operands.w.ptr != self.b.ptr()
                || operands.bias_ptr != self.has_bias.then(|| self.bias.ptr())
            {
                return Err("Task7 common input pointer contract differs".into());
            }
        }
        Ok(())
    }

    fn timing_boundary(&self, ctx: &GpuCtx) -> Result<(), String> {
        self.inputs(ctx)?;
        for output in &self.outputs {
            output.words(ctx)?;
        }
        Ok(())
    }
}

fn cuda(result: cudarc::driver::sys::CUresult, label: &str) -> Result<(), String> {
    if result == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{label}: {result:?}"))
    }
}

fn graph_nodes(graph: &CudaGraph) -> Result<Vec<cudarc::driver::sys::CUgraphNode>, String> {
    use cudarc::driver::sys;
    let mut count = 0;
    cuda(
        unsafe { sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count) },
        "Task7 graph node count",
    )?;
    if count == 0 {
        return Err("Task7 captured an empty measured workflow".into());
    }
    let mut nodes = vec![std::ptr::null_mut(); count];
    cuda(
        unsafe { sys::cuGraphGetNodes(graph.cu_graph(), nodes.as_mut_ptr(), &mut count) },
        "Task7 graph node inventory",
    )?;
    if count != nodes.len() {
        return Err("Task7 graph node count drift".into());
    }
    Ok(nodes)
}

fn captured<T: Copy>(
    params: &cudarc::driver::sys::CUDA_KERNEL_NODE_PARAMS_v2,
    index: usize,
) -> Result<T, String> {
    if params.kernelParams.is_null() || !params.extra.is_null() {
        return Err("Task7 custom graph did not expose kernelParams".into());
    }
    let pointer = unsafe { *params.kernelParams.add(index) };
    if pointer.is_null() {
        return Err(format!("Task7 missing captured argument {index}"));
    }
    Ok(unsafe { std::ptr::read_unaligned(pointer.cast::<T>()) })
}

fn inspect_custom_graph(
    graph: &CudaGraph,
    case: &Case,
    arm: usize,
    logical_ops: usize,
) -> Result<String, String> {
    use cudarc::driver::sys;
    let nodes = graph_nodes(graph)?;
    if nodes.len() != logical_ops {
        return Err(format!(
            "Task7 custom workflow has {} nodes, expected {logical_ops}",
            nodes.len()
        ));
    }
    let mut rendered = Vec::with_capacity(nodes.len());
    for node in nodes {
        let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        cuda(
            unsafe { sys::cuGraphNodeGetType(node, &mut kind) },
            "Task7 graph node kind",
        )?;
        if kind != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
            return Err("Task7 custom workflow contains a non-kernel node".into());
        }
        let mut params: sys::CUDA_KERNEL_NODE_PARAMS_v2 = unsafe { std::mem::zeroed() };
        cuda(
            unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) },
            "Task7 custom graph kernel params",
        )?;
        let mut name = std::ptr::null();
        cuda(
            unsafe { sys::cuFuncGetName(&mut name, params.func) },
            "Task7 custom graph symbol",
        )?;
        if name.is_null() {
            return Err("Task7 custom graph symbol is null".into());
        }
        let symbol = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|error| format!("Task7 custom graph symbol UTF-8: {error}"))?;
        let (argument_count, expected_symbol, grid, block, shared, parameters, abi) =
            match (case.family, arm) {
                (Family::Exact, 0) => (
                    12,
                    "gemm_bi_f32_f32_s2",
                    (
                        (case.shape.m.div_ceil(64) * case.shape.n.div_ceil(64)) as u32,
                        1,
                        1,
                    ),
                    (128, 1, 1),
                    0,
                    vec![
                        1.0f32.to_bits(),
                        0,
                        case.shape.m as u32,
                        case.shape.n as u32,
                        case.shape.k as u32,
                        case.shape.k as u32,
                        case.shape.n as u32,
                        case.shape.n as u32,
                    ],
                    (0..4)
                        .map(|index| (index * 8, 8))
                        .chain((0..8).map(|index| (32 + index * 4, 4)))
                        .collect::<Vec<_>>(),
                ),
                (Family::Exact, 1) => (
                    5,
                    "gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1",
                    (
                        (case.shape.m.div_ceil(64) * case.shape.n.div_ceil(64)) as u32,
                        1,
                        1,
                    ),
                    (128, 1, 1),
                    0,
                    vec![
                        1.0f32.to_bits(),
                        0,
                        case.shape.m as u32,
                        case.shape.n as u32,
                        case.shape.k as u32,
                        case.shape.k as u32,
                        case.shape.n as u32,
                        case.shape.n as u32,
                    ],
                    vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)],
                ),
                (Family::Tf32, 0) => (
                    5,
                    "gemm_bi_nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3",
                    (111, 1, 1),
                    (256, 1, 1),
                    98_304,
                    vec![1.0f32.to_bits(), 0, 4621, 1928, 384, 1928, 384, 384],
                    vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)],
                ),
                (Family::Tf32, 1) => (
                    5,
                    "gemm_bi_nn_tf32_v1_m64n64_bk32_s2",
                    (438, 1, 1),
                    (128, 1, 1),
                    32_768,
                    vec![4621, 1928, 384, 1928, 384, 384],
                    vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 24)],
                ),
                _ => return Err("Task7 physical inspection only accepts custom arms".into()),
            };
        let mut actual_abi = Vec::with_capacity(argument_count);
        for index in 0..argument_count {
            let (mut offset, mut size) = (0, 0);
            cuda(
                unsafe { sys::cuFuncGetParamInfo(params.func, index, &mut offset, &mut size) },
                "Task7 custom Driver ABI",
            )?;
            actual_abi.push((offset, size));
        }
        let (mut offset, mut size) = (0, 0);
        let terminal_rejected =
            unsafe { sys::cuFuncGetParamInfo(params.func, argument_count, &mut offset, &mut size) }
                == sys::CUresult::CUDA_ERROR_INVALID_VALUE;
        if actual_abi != abi || !terminal_rejected {
            return Err(format!(
                "Task7 custom ABI rejected before captured reads: abi={actual_abi:?} terminal={terminal_rejected}"
            ));
        }
        let pointers = [
            captured(&params, 0)?,
            captured(&params, 1)?,
            captured(&params, 2)?,
            captured(&params, 3)?,
        ];
        let actual_parameters = if argument_count == 12 {
            (4..12)
                .map(|index| captured::<u32>(&params, index))
                .collect::<Result<Vec<_>, _>>()?
        } else if case.family == Family::Exact || arm == 0 {
            captured::<[u32; 8]>(&params, 4)?.to_vec()
        } else {
            captured::<[u32; 6]>(&params, 4)?.to_vec()
        };
        let expected = LaunchContract {
            symbol: expected_symbol,
            grid,
            block,
            dynamic_shared: shared,
            pointers: [
                case.operands(arm).c.ptr,
                case.a.ptr(),
                case.b.ptr(),
                case.has_bias.then(|| case.bias.ptr()).unwrap_or(0),
            ],
            parameters,
            abi,
            terminal_rejected: true,
        };
        let actual = LaunchContract {
            symbol,
            grid: (params.gridDimX, params.gridDimY, params.gridDimZ),
            block: (params.blockDimX, params.blockDimY, params.blockDimZ),
            dynamic_shared: params.sharedMemBytes,
            pointers,
            parameters: actual_parameters,
            abi: actual_abi,
            terminal_rejected,
        };
        if case.family == Family::Tf32 && arm == 0 {
            fixed_explicit_vendor_rna_wide_graph_contract(
                1,
                actual.symbol,
                actual.grid,
                actual.block,
                actual.dynamic_shared,
                actual
                    .parameters
                    .as_slice()
                    .try_into()
                    .map_err(|_| "Task7 RNA AUTO bundle is not eight words")?,
            )?;
        }
        validate_launch(&actual, &expected)?;
        if case.family == Family::Exact && arm == 1 {
            let mut static_shared = 0;
            cuda(
                unsafe {
                    sys::cuFuncGetAttribute(
                        &mut static_shared,
                        sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                        params.func,
                    )
                },
                "Task7 CopyPlan static shared",
            )?;
            if static_shared != 32_768 {
                return Err(format!("Task7 CopyPlan static shared is {static_shared}"));
            }
        }
        rendered.push(launch_json(&actual));
    }
    Ok(format!("[{}]", rendered.join(",")))
}

fn inspect_vendor_graph(
    graph: &CudaGraph,
    has_bias: bool,
    logical_ops: usize,
) -> Result<String, String> {
    use cudarc::driver::sys;
    let nodes = graph_nodes(graph)?;
    let expected_nodes = logical_ops * if has_bias { 2 } else { 1 };
    if nodes.len() != expected_nodes {
        return Err(format!(
            "Task7 vendor workflow has {} nodes, expected {expected_nodes}",
            nodes.len()
        ));
    }
    let mut bias_nodes = 0;
    let mut rendered = Vec::with_capacity(nodes.len());
    for node in nodes {
        let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        cuda(
            unsafe { sys::cuGraphNodeGetType(node, &mut kind) },
            "Task7 vendor node kind",
        )?;
        if kind != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
            return Err("Task7 vendor workflow contains a non-kernel node".into());
        }
        let mut params: sys::CUDA_KERNEL_NODE_PARAMS_v2 = unsafe { std::mem::zeroed() };
        cuda(
            unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) },
            "Task7 vendor graph kernel params",
        )?;
        let mut name = std::ptr::null();
        cuda(
            unsafe { sys::cuFuncGetName(&mut name, params.func) },
            "Task7 vendor graph symbol",
        )?;
        if name.is_null() {
            return Err("Task7 vendor graph symbol is null".into());
        }
        let symbol = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|error| format!("Task7 vendor symbol UTF-8: {error}"))?;
        bias_nodes += usize::from(symbol == "bias_broadcast");
        if params.gridDimX == 0 || params.blockDimX == 0 {
            return Err("Task7 vendor graph has zero launch geometry".into());
        }
        rendered.push(format!(
            "{{\"symbol\":\"{}\",\"grid\":[{},{},{}],\"block\":[{},{},{}],\"shared_bytes\":{}}}",
            fixed_sm120_tf32_bd_json_escape(symbol),
            params.gridDimX,
            params.gridDimY,
            params.gridDimZ,
            params.blockDimX,
            params.blockDimY,
            params.blockDimZ,
            params.sharedMemBytes,
        ));
    }
    if bias_nodes != usize::from(has_bias) * logical_ops {
        return Err(format!("Task7 vendor bias node count is {bias_nodes}"));
    }
    Ok(format!("[{}]", rendered.join(",")))
}

fn cublas_modes(ctx: &GpuCtx) -> Result<String, String> {
    use cudarc::cublas::sys::*;
    let handle = *ctx.blas.handle();
    let success = cublasStatus_t::CUBLAS_STATUS_SUCCESS;
    let mut math = cublasMath_t::CUBLAS_DEFAULT_MATH;
    let mut pointer = cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST;
    let mut atomics = cublasAtomicsMode_t::CUBLAS_ATOMICS_NOT_ALLOWED;
    unsafe {
        if cublasSetMathMode(handle, math) != success
            || cublasSetPointerMode_v2(handle, pointer) != success
            || cublasSetAtomicsMode(handle, atomics) != success
            || cublasGetMathMode(handle, &mut math) != success
            || cublasGetPointerMode_v2(handle, &mut pointer) != success
            || cublasGetAtomicsMode(handle, &mut atomics) != success
        {
            return Err("Task7 cuBLAS mode set/query failed".into());
        }
    }
    if math != cublasMath_t::CUBLAS_DEFAULT_MATH
        || pointer != cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST
        || atomics != cublasAtomicsMode_t::CUBLAS_ATOMICS_NOT_ALLOWED
    {
        return Err("Task7 cuBLAS modes differ from native contract".into());
    }
    Ok(format!(
        "\"vendor_compute\":\"CUBLAS_COMPUTE_32F_FAST_TF32\",\"vendor_algorithm\":\"CUBLAS_GEMM_DEFAULT\",\"vendor_math\":\"{math:?}\",\"vendor_pointer_mode\":\"{pointer:?}\",\"vendor_atomics\":\"{atomics:?}\""
    ))
}

fn source_sha() -> Result<String, String> {
    let mut digest = Sha256::new();
    for path in [
        "src/mamba_ssm/gpu/gemm_bi_inference.rs",
        "src/mamba_ssm/gpu/kernel_identity.rs",
        "tests/gemm_bi_fixed_performance.rs",
        "tests/support/fixed_sm89_toolkit_admission.rs",
    ] {
        digest.update(path.as_bytes());
        digest.update([0]);
        digest.update(std::fs::read(path).map_err(|error| format!("read {path}: {error}"))?);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn binary_sha() -> Result<String, String> {
    let path = std::env::current_exe().map_err(|error| format!("Task7 current exe: {error}"))?;
    Ok(format!(
        "{:x}",
        Sha256::digest(std::fs::read(path).map_err(|error| format!("read Task7 exe: {error}"))?)
    ))
}

fn literal(literal: &str) -> Result<(InferenceShape, bool), String> {
    let (cell, bias) = literal
        .split_once(':')
        .ok_or_else(|| format!("malformed Task7 literal {literal}"))?;
    let shape = FIXED_AUTO_VENDOR_EXACT_CELLS
        .iter()
        .find(|candidate| candidate.label == cell)
        .map(|candidate| candidate.shape)
        .ok_or_else(|| format!("unknown Task7 cell {cell}"))?;
    Ok((shape, bias == "1"))
}

fn ordered_dot(a: &[f32], b: &[f32], bias: Option<f32>) -> Result<u32, String> {
    if a.len() != b.len() {
        return Err("Task7 ordered dot extent differs".into());
    }
    let mut value = 0.0f32;
    for (&a, &b) in a.iter().zip(b) {
        if !a.is_finite() || !b.is_finite() {
            return Err("Task7 exact order proof requires finite inputs".into());
        }
        value = a.mul_add(b, value);
    }
    if let Some(bias) = bias {
        if !bias.is_finite() {
            return Err("Task7 exact order proof requires finite bias".into());
        }
        value += bias;
    }
    if !value.is_finite() {
        return Err("Task7 exact ordered result is nonfinite".into());
    }
    Ok(value.to_bits())
}

fn single_term_controls(ctx: &GpuCtx, family: Family) -> Result<(), String> {
    let (shape, a_values, b_values, bias_values) = match family {
        Family::Exact => (
            InferenceShape { m: 2, k: 1, n: 3 },
            vec![2.0, -3.0],
            vec![5.0, 7.0, -11.0],
            vec![0.25, 0.5, 0.75],
        ),
        Family::Tf32 => (
            InferenceShape { m: 2, k: 4, n: 4 },
            vec![2.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0],
            vec![
                5.0, 7.0, -11.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![0.25, 0.5, 0.75, 1.0],
        ),
    };
    let a = Guarded::new(ctx, a_values)?;
    let b = Guarded::new(ctx, b_values)?;
    let bias = Guarded::new(ctx, bias_values)?;
    for has_bias in [false, true] {
        for tile in [family.incumbent(), family.candidate()] {
            let output = Guarded::new(ctx, vec![0.0; shape.m * shape.n])?;
            let operands = InferenceFwdOperands {
                c: TypedPtr {
                    ptr: output.ptr(),
                    dtype: WeightDtype::F32,
                },
                x: TypedPtr {
                    ptr: a.ptr(),
                    dtype: WeightDtype::F32,
                },
                w: TypedPtr {
                    ptr: b.ptr(),
                    dtype: WeightDtype::F32,
                },
                bias_ptr: has_bias.then(|| bias.ptr()),
            };
            inference_forward_with_tile(ctx, operands, shape, tile)?;
            let actual = output.read(ctx)?;
            let mut expected = Vec::with_capacity(shape.m * shape.n);
            for row in 0..shape.m {
                for column in 0..shape.n {
                    let mut value = 0.0f32;
                    for reduction in 0..shape.k {
                        value = f32::from_bits(a.initial[GUARD_WORDS + row * shape.k + reduction])
                            .mul_add(
                                f32::from_bits(
                                    b.initial[GUARD_WORDS + reduction * shape.n + column],
                                ),
                                value,
                            );
                    }
                    if has_bias {
                        value += f32::from_bits(bias.initial[GUARD_WORDS + column]);
                    }
                    expected.push(value.to_bits());
                }
            }
            if actual != expected {
                return Err(format!(
                    "Task7 single-term/bias-orientation control failed for {family:?}/{tile:?}/bias={has_bias}"
                ));
            }
            output.output_gate(ctx, &expected, || {
                inference_forward_with_tile(ctx, operands, shape, tile)
            })?;
        }
    }
    a.unchanged(ctx)?;
    b.unchanged(ctx)?;
    bias.unchanged(ctx)
}

fn finite_order_control(case: &Case, outputs: &[Vec<u32>]) -> Result<(), String> {
    if outputs
        .iter()
        .flatten()
        .any(|word| !f32::from_bits(*word).is_finite())
    {
        return Err("Task7 full-shape output contains nonfinite values".into());
    }
    if case.family != Family::Exact {
        return Ok(());
    }
    let indices = [0, case.shape.n - 1, case.shape.m * case.shape.n / 2];
    for index in indices {
        let row = index / case.shape.n;
        let column = index % case.shape.n;
        let a = (0..case.shape.k)
            .map(|k| f32::from_bits(case.a.initial[GUARD_WORDS + row * case.shape.k + k]))
            .collect::<Vec<_>>();
        let b = (0..case.shape.k)
            .map(|k| f32::from_bits(case.b.initial[GUARD_WORDS + k * case.shape.n + column]))
            .collect::<Vec<_>>();
        let bias = case
            .has_bias
            .then(|| f32::from_bits(case.bias.initial[GUARD_WORDS + column]));
        let ordered = ordered_dot(&a, &b, bias)?;
        if outputs[0][index] != ordered || outputs[1][index] != ordered {
            return Err(format!(
                "Task7 exact ascending-FMA certificate failed at output {index}"
            ));
        }
    }
    Ok(())
}

fn prelaunch_gate<T>(
    numeric_abi_revision: u16,
    schedule_revision: u16,
    expected_artifact: Option<&str>,
    actual_artifact: &str,
    launch: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    if numeric_abi_revision != 5 || schedule_revision != 8 {
        return Err(format!(
            "Task7 compiler revisions differ: numeric={numeric_abi_revision} schedule={schedule_revision}"
        ));
    }
    if expected_artifact.is_some_and(|expected| expected != actual_artifact) {
        return Err(format!(
            "Task7 Fixed artifact differs from bound screen: {actual_artifact}"
        ));
    }
    launch()
}

fn timing_boundary_gate<T>(
    check: impl FnOnce() -> Result<(), String>,
    post_timing_workflows: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    check()?;
    post_timing_workflows()
}

fn run_inner(mode: RunMode) -> Result<(), String> {
    let environment = std::env::vars().collect::<BTreeMap<_, _>>();
    let config = match mode {
        RunMode::Task7 => parse_config(&environment)?,
        RunMode::PostAuto44 => parse_post_config(&environment)?,
    };
    if cfg!(debug_assertions) {
        return Err("toolkit admission requires --release".into());
    }
    if TUNING_TABLE_REVISION != config.mode.tuning_revision() {
        return Err(format!(
            "{:?} requires tuning revision{}, got {TUNING_TABLE_REVISION}",
            config.mode,
            config.mode.tuning_revision(),
        ));
    }
    let measured_source_sha = source_sha()?;
    let measured_binary_sha = binary_sha()?;
    if config.source_sha != measured_source_sha || config.binary_sha != measured_binary_sha {
        return Err(format!(
            "Task7 source/binary binding mismatch source={measured_source_sha} binary={measured_binary_sha}"
        ));
    }
    let mut output = Jsonl::create(&config.jsonl, config.mode.schema())?;
    let device = GpuDevice::new(0).map_err(|error| format!("Task7 CUDA device: {error}"))?;
    if device.compute_capability != (8, 9) || device.multiprocessor_count() != 142 {
        return Err(format!(
            "Task7 wrong physical device CC{:?}/{}SM",
            device.compute_capability,
            device.multiprocessor_count()
        ));
    }
    let ctx = GpuCtx::new(&device).map_err(|error| format!("Task7 GPU context: {error}"))?;
    let compiler = ctx.kernels.compiler_identity();
    let actual_toolkit = format!("{}.{}", compiler.nvrtc_version.0, compiler.nvrtc_version.1);
    if actual_toolkit != config.toolkit
        || !compiler.nvrtc_library_known
        || compiler.target.as_str() != "sm_89"
    {
        return Err(format!(
            "Task7 compiler identity mismatch toolkit={actual_toolkit} known={} target={:?}",
            compiler.nvrtc_library_known, compiler.target
        ));
    }
    configure_fixed_auto_vendor_custom(&ctx, config.family.policy());
    let modes = cublas_modes(&ctx)?;
    let artifact = ctx.kernels.artifact_set_identity().fixed;
    let artifact_sha = digest_hex(&artifact.artifact_digest);
    let promotion_basis = match config.mode {
        RunMode::Task7 => String::new(),
        RunMode::PostAuto44 => concat!(
            "\"promotion_basis\":{",
            "\"task7_source_sha\":\"97f3d43fc315c96238f41f9f39bc15418518b5493e216fb9bc0a8f03a46e5bc7\",",
            "\"cuda128_screen_sha\":\"d43c360d844065a3691f933b0b743a18e8482d2d36de787ef0ed2392e30e53bf\",",
            "\"cuda128_confirm_sha\":\"eb3b0abee0e336c7c93f9c5eddec698dae8a3fc1d8363ca337047b835758ad96\",",
            "\"cuda130_screen_sha\":\"cae9db1864bd64700ee3033196eae4c6f8a0abd7a3a1cf09a407889cb7727682\",",
            "\"cuda130_confirm_sha\":\"0b27351512f3265b156a29aaf7fadcaf4a84870cac06e1c1b36ce3e358f20711\",",
            "\"task7_final_review_sha\":\"750e0d02b524229c7a987894eee214af5e33e57f75499779b7263352b33697ac\",",
            "\"task7_selected_manifest_sha\":\"822560b7978f641543418e5971033b872dd83198157d8456d20317998c4c58d7\"},"
        )
        .to_owned(),
    };
    prelaunch_gate(
        compiler.numeric_abi_revision,
        compiler.schedule_revision,
        config.screen_artifact_sha.as_deref(),
        &artifact_sha,
        || single_term_controls(&ctx, config.family),
    )?;
    output.emit(&format!(
        concat!(
            "\"kind\":\"identity\",\"family\":\"{}\",\"stage\":\"{}\",",
            "\"windows\":{},\"toolkit\":\"{}\",\"tuning_revision\":{},",
            "\"numeric_abi_revision\":{},\"schedule_revision\":{},",
            "\"uuid\":\"{}\",\"cc\":\"8.9\",\"sm_count\":142,",
            "\"compiler_target\":\"sm_89\",\"nvrtc_library_known\":true,",
            "\"source_sha\":\"{}\",\"binary_sha\":\"{}\",",
            "\"screen_sha\":{},\"screen_artifact_sha\":{},\"fixed_source_digest\":\"{}\",",
            "\"fixed_invocation_digest\":\"{}\",\"fixed_artifact_digest\":\"{}\",",
            "\"header_manifest_digest\":\"{}\",\"nvrtc_library_domain\":\"{}\",",
            "\"literal_control\":\"{}\",\"paths\":[\"eager\",\"graph\"],",
            "\"start_parities\":[0,1],\"warmup_eager\":128,\"logical_ops\":20,{}{}"
        ),
        config.mode.family_name(config.family),
        config.stage.name(),
        config.windows,
        config.toolkit,
        config.mode.tuning_revision(),
        compiler.numeric_abi_revision,
        compiler.schedule_revision,
        "GPU-d1edd7be-e88d-aed6-047d-622163306f0e",
        measured_source_sha,
        measured_binary_sha,
        config
            .screen_sha
            .as_deref()
            .map(|sha| format!("\"{sha}\""))
            .unwrap_or_else(|| "null".into()),
        config
            .screen_artifact_sha
            .as_deref()
            .map(|sha| format!("\"{sha}\""))
            .unwrap_or_else(|| "null".into()),
        digest_hex(&compiler.source_digest),
        digest_hex(&compiler.invocation_digest),
        artifact_sha,
        digest_hex(&compiler.header_manifest_digest),
        digest_hex(&compiler.nvrtc_library_domain),
        config.literals.join(","),
        promotion_basis,
        modes,
    ))?;
    let arms = config.mode.arms();
    let directions = config.mode.directions();
    let mut configurations = 0;
    for literal_name in &config.literals {
        let (shape, has_bias) = literal(literal_name)?;
        if config.family == Family::Tf32
            && shape
                != (InferenceShape {
                    m: 4621,
                    k: 1928,
                    n: 384,
                })
        {
            return Err("Task7 TF32 literal shape differs from C".into());
        }
        let case = Case::new(&ctx, config.mode, config.family, shape, has_bias)?;
        for arm in 0..4 {
            case.launch(&ctx, arm)?;
        }
        let expected = case
            .outputs
            .iter()
            .map(|buffer| buffer.read(&ctx))
            .collect::<Result<Vec<_>, _>>()?;
        if expected[0] != expected[1] {
            return Err(format!(
                "custom owner arms changed exact raw bits for {literal_name}"
            ));
        }
        finite_order_control(&case, &expected)?;
        fixed_ada_normalized_error(
            &expected[0],
            &expected[3],
            config.family.custom_tolerance(),
            &format!("Task7 custom numerical {literal_name}"),
        );
        fixed_ada_normalized_error(
            &expected[2],
            &expected[3],
            0.0025,
            &format!("Task7 Fast numerical {literal_name}"),
        );
        for arm in 0..3 {
            for _ in 0..2 {
                case.outputs[arm].output_gate(&ctx, &expected[arm], || case.launch(&ctx, arm))?;
            }
        }
        let mut one = Vec::with_capacity(3);
        let mut twenty = Vec::with_capacity(3);
        let mut physical = Vec::with_capacity(3);
        for arm in 0..3 {
            let graph = unsafe {
                capture_into_graph(&ctx.stream, || {
                    case.launch(&ctx, arm)?;
                    Ok(())
                })
            }
            .map_err(|error| format!("Task7 capture one-op graph: {error}"))?;
            let graph20 = unsafe {
                capture_into_graph(&ctx.stream, || {
                    for _ in 0..20 {
                        case.launch(&ctx, arm)?;
                    }
                    Ok(())
                })
            }
            .map_err(|error| format!("Task7 capture twenty-op graph: {error}"))?;
            let inventory = if arm < 2 {
                format!(
                    "{{\"one\":{},\"twenty\":{}}}",
                    inspect_custom_graph(&graph, &case, arm, 1)?,
                    inspect_custom_graph(&graph20, &case, arm, 20)?,
                )
            } else {
                let bias_symbol = has_bias.then_some("bias_broadcast");
                let existing =
                    fixed_explicit_vendor_graph_inventory(&graph, "Task7 Fast", None, bias_symbol);
                format!(
                    "{{\"one_existing\":{},\"one\":{},\"twenty\":{}}}",
                    existing,
                    inspect_vendor_graph(&graph, has_bias, 1)?,
                    inspect_vendor_graph(&graph20, has_bias, 20)?,
                )
            };
            for replay in [&graph, &graph20] {
                for _ in 0..2 {
                    case.outputs[arm].output_gate(&ctx, &expected[arm], || {
                        replay
                            .launch()
                            .map_err(|error| format!("Task7 graph replay: {error}"))
                    })?;
                }
            }
            physical.push(inventory);
            one.push(graph);
            twenty.push(graph20);
        }
        let noop = unsafe { capture_into_graph(&ctx.stream, || Ok(())) }
            .map_err(|error| format!("Task7 empty graph capture: {error}"))?;
        let noop_error = case.outputs[1]
            .output_gate(&ctx, &expected[1], || {
                noop.launch()
                    .map_err(|error| format!("Task7 no-op launch: {error}"))
            })
            .expect_err("Task7 real empty graph must leave complement poison unchanged");
        if noop_error != "Task7 replay did not overwrite every expected storage word" {
            return Err(format!(
                "Task7 empty graph failed for the wrong reason: {noop_error}"
            ));
        }
        case.launch(&ctx, 1)?;
        case.inputs(&ctx)?;
        let route_fields = match config.mode {
            RunMode::Task7 => format!(
                "\"actual_auto\":\"{:?}\",\"candidate\":\"{:?}\",\"graphs\":{{\"actualAUTO\":{},\"candidate\":{},\"Fast\":{}}}",
                config.family.incumbent(),
                config.family.candidate(),
                physical[0],
                physical[1],
                physical[2],
            ),
            RunMode::PostAuto44 => format!(
                "\"former_incumbent\":\"Legacy\",\"actual_auto\":\"F32Sm89N64CopyPlan\",\"public_auto_enum_verified\":true,\"graphs\":{{\"Legacy\":{},\"AUTO\":{},\"Fast\":{}}}",
                physical[0], physical[1], physical[2],
            ),
        };
        output.emit(&format!(
            concat!(
                "\"kind\":\"physical\",\"family\":\"{}\",\"stage\":\"{}\",",
                "\"literal\":\"{}\",\"bias\":{},\"shape\":[{},{},{}],{},",
                "\"custom_bits_equal\":true,\"fast_repeat_bits\":true,",
                "\"poison_upload_readback\":true,\"noop_rejected\":true,",
                "\"guards\":true,\"immutable_inputs\":true,\"bias_orientation\":true,",
                "\"finite_ordering_controls\":true"
            ),
            config.mode.family_name(config.family),
            config.stage.name(),
            literal_name,
            has_bias,
            shape.m,
            shape.k,
            shape.n,
            route_fields,
        ))?;
        let mut own_strata = Vec::with_capacity(4);
        for path in ["eager", "graph"] {
            for start_parity in 0..2 {
                let scheduled = (0..config.windows)
                    .flat_map(|window| schedule(window, start_parity).unwrap())
                    .collect::<Vec<_>>();
                let mut observations = Vec::with_capacity(12 * config.windows);
                case.inputs(&ctx)?;
                for _ in 0..128 {
                    for arm in 0..3 {
                        case.launch(&ctx, arm)?;
                    }
                }
                for arm in 0..3 {
                    one[arm]
                        .launch()
                        .map_err(|error| format!("Task7 warm one graph: {error}"))?;
                    twenty[arm]
                        .launch()
                        .map_err(|error| format!("Task7 warm twenty graph: {error}"))?;
                }
                ctx.stream
                    .synchronize()
                    .map_err(|error| format!("Task7 timing preparation sync: {error}"))?;
                for mut observation in scheduled {
                    let arm = observation.arm;
                    observation.us = if path == "eager" {
                        fixed_ada_event_window_us(&ctx, 20, || {
                            case.launch(&ctx, arm).expect("Task7 timed eager launch")
                        })
                    } else {
                        fixed_ada_event_window_us(&ctx, 1, || {
                            twenty[arm].launch().expect("Task7 timed graph launch")
                        }) / 20.0
                    };
                    observations.push(observation);
                }
                timing_boundary_gate(
                    || case.timing_boundary(&ctx),
                    || {
                        for arm in 0..3 {
                            if case.outputs[arm].read(&ctx)? != expected[arm] {
                                return Err("Task7 post-timing output bits differ".into());
                            }
                            case.outputs[arm]
                                .output_gate(&ctx, &expected[arm], || case.launch(&ctx, arm))?;
                            for replay in [&one[arm], &twenty[arm]] {
                                case.outputs[arm].output_gate(&ctx, &expected[arm], || {
                                    replay.launch().map_err(|error| {
                                        format!("Task7 post-timing graph: {error}")
                                    })
                                })?;
                            }
                        }
                        Ok(())
                    },
                )?;
                case.inputs(&ctx)?;
                let mut ratio_sets: [Vec<f64>; 3] =
                    std::array::from_fn(|_| Vec::with_capacity(config.windows));
                let mut pair_records = Vec::with_capacity(3 * config.windows);
                for (chronology, observation) in observations.iter().enumerate() {
                    output.emit(&format!(
                        concat!(
                            "\"kind\":\"sample\",\"family\":\"{}\",\"stage\":\"{}\",",
                            "\"literal\":\"{}\",\"path\":\"{}\",\"start_parity\":{},",
                            "\"chronology\":{},\"window\":{},\"traversal\":{},",
                            "\"comparison\":{},\"position\":{},\"arm\":\"{}\",",
                            "\"logical_ops\":20,\"us\":{}"
                        ),
                        config.mode.family_name(config.family),
                        config.stage.name(),
                        literal_name,
                        path,
                        start_parity,
                        chronology,
                        observation.window,
                        observation.traversal,
                        observation.comparison,
                        observation.position,
                        arms[observation.arm],
                        observation.us,
                    ))?;
                }
                for (bracket, values) in observations.chunks_exact(4).enumerate() {
                    let value = ratio(values)?;
                    let comparison = values[0].comparison;
                    ratio_sets[comparison].push(value);
                    pair_records.push((values[0].window, comparison, value));
                    output.emit(&format!(
                        concat!(
                            "\"kind\":\"pair\",\"family\":\"{}\",\"stage\":\"{}\",",
                            "\"literal\":\"{}\",\"path\":\"{}\",\"start_parity\":{},",
                            "\"window\":{},\"traversal\":{},\"comparison\":{},",
                            "\"direction\":\"{}\",\"observations\":[{},{},{},{}],",
                            "\"ratio\":{}"
                        ),
                        config.mode.family_name(config.family),
                        config.stage.name(),
                        literal_name,
                        path,
                        start_parity,
                        values[0].window,
                        values[0].traversal,
                        comparison,
                        directions[comparison],
                        bracket * 4,
                        bracket * 4 + 1,
                        bracket * 4 + 2,
                        bracket * 4 + 3,
                        value,
                    ))?;
                }
                let mut summaries = Vec::with_capacity(3);
                for comparison in 0..3 {
                    let (p50, p95) = quantiles(&ratio_sets[comparison])?;
                    summaries.push((comparison, p50, p95));
                    output.emit(&format!(
                        concat!(
                            "\"kind\":\"summary\",\"family\":\"{}\",\"stage\":\"{}\",",
                            "\"literal\":\"{}\",\"path\":\"{}\",\"start_parity\":{},",
                            "\"comparison\":{},\"direction\":\"{}\",\"windows\":{},",
                            "\"p50\":{},\"p95\":{}"
                        ),
                        config.mode.family_name(config.family),
                        config.stage.name(),
                        literal_name,
                        path,
                        start_parity,
                        comparison,
                        directions[comparison],
                        config.windows,
                        p50,
                        p95,
                    ))?;
                }
                validate_configuration(
                    config.windows,
                    start_parity,
                    &observations,
                    &pair_records,
                    &summaries,
                    true,
                )?;
                own_strata.push((summaries[0].1, summaries[0].2));
                output.emit(&format!(
                    concat!(
                        "\"kind\":\"configuration_complete\",\"family\":\"{}\",",
                        "\"stage\":\"{}\",\"literal\":\"{}\",\"path\":\"{}\",",
                        "\"start_parity\":{},\"samples\":{},\"pairs\":{},",
                        "\"summaries\":3,\"physical_bits_guards_inputs\":true"
                    ),
                    config.mode.family_name(config.family),
                    config.stage.name(),
                    literal_name,
                    path,
                    start_parity,
                    12 * config.windows,
                    3 * config.windows,
                ))?;
                configurations += 1;
            }
        }
        output.emit(&format!(
            concat!(
                "\"kind\":\"literal_decision\",\"family\":\"{}\",\"stage\":\"{}\",",
                "\"literal\":\"{}\",\"own_admission\":{},",
                "\"own_rule\":\"{}\""
            ),
            config.mode.family_name(config.family),
            config.stage.name(),
            literal_name,
            literal_admitted(&own_strata)?,
            config.mode.own_rule(),
        ))?;
    }
    if configurations != config.literals.len() * 4 {
        return Err(format!("Task7 configuration closure is {configurations}"));
    }
    output.complete(configurations, config.windows, config.literals.len())
}

pub(super) fn run() {
    run_inner(RunMode::Task7)
        .unwrap_or_else(|error| panic!("Task7 toolkit admission rejected: {error}"));
}

pub(super) fn run_post_auto() {
    run_inner(RunMode::PostAuto44)
        .unwrap_or_else(|error| panic!("Task8 post-AUTO admission rejected: {error}"));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn exact_screen() -> BTreeMap<String, String> {
        BTreeMap::from([
            ("MAMBA_FIXED_ADA_TOOLKIT_ADMISSION".into(), "1".into()),
            ("MAMBA_FIXED_ADA_VENDOR".into(), "1".into()),
            ("MAMBA_FIXED_ADA_ROWS".into(), "f32_exact_fast".into()),
            ("MAMBA_FIXED_ADA_LITERALS".into(), EXACT_LITERALS.join(",")),
            ("MAMBA_FIXED_ADA_WINDOWS".into(), "21".into()),
            ("MAMBA_FIXED_ADA_TOOLKIT".into(), "12.8".into()),
            ("MAMBA_FIXED_ADA_TUNING_REVISION".into(), "43".into()),
            ("MAMBA_FIXED_ADA_STAGE".into(), "screen21".into()),
            ("MAMBA_FIXED_ADA_SOURCE_SHA".into(), sha('a')),
            ("MAMBA_FIXED_ADA_BINARY_SHA".into(), sha('b')),
            (
                "MAMBA_FIXED_ADA_JSONL".into(),
                "/root/evidence/task7-screen.jsonl".into(),
            ),
            (
                "MAMBA_FIXED_VENDOR_TILES".into(),
                "F32Sm89N64CopyPlan".into(),
            ),
            ("MAMBA_FIXED_VENDOR_PATHS".into(), "eager,graph".into()),
            ("MAMBA_FIXED_VENDOR_EXACT_CC".into(), "8.9".into()),
        ])
    }

    fn exact_post(stage: &str, windows: &str) -> BTreeMap<String, String> {
        BTreeMap::from([
            ("MAMBA_FIXED_ADA_EXACT_POST_AUTO".into(), "1".into()),
            ("MAMBA_FIXED_ADA_VENDOR".into(), "1".into()),
            ("MAMBA_FIXED_ADA_EXACT_POST_STAGE".into(), stage.into()),
            ("MAMBA_FIXED_ADA_EXACT_POST_WINDOWS".into(), windows.into()),
            ("MAMBA_FIXED_ADA_EXACT_POST_TOOLKIT".into(), "12.8".into()),
            (
                "MAMBA_FIXED_ADA_EXACT_POST_LITERALS".into(),
                EXACT_LITERALS.join(","),
            ),
            (
                "MAMBA_FIXED_ADA_EXACT_POST_TUNING_REVISION".into(),
                "44".into(),
            ),
            ("MAMBA_FIXED_ADA_EXACT_POST_SOURCE_SHA".into(), sha('c')),
            ("MAMBA_FIXED_ADA_EXACT_POST_BINARY_SHA".into(), sha('d')),
            (
                "MAMBA_FIXED_ADA_EXACT_POST_JSONL".into(),
                "/root/evidence/task8-post.jsonl".into(),
            ),
            (
                "MAMBA_FIXED_VENDOR_TILES".into(),
                "Legacy,F32Sm89N64CopyPlan".into(),
            ),
            ("MAMBA_FIXED_VENDOR_PATHS".into(), "eager,graph".into()),
            ("MAMBA_FIXED_VENDOR_EXACT_CC".into(), "8.9".into()),
        ])
    }

    #[test]
    fn post44_controls_schema_roles_and_directions_are_explicit_and_disjoint() {
        let smoke = parse_post_config(&exact_post("smoke1", "1")).unwrap();
        assert_eq!(smoke.mode, RunMode::PostAuto44);
        assert_eq!(smoke.family, Family::Exact);
        assert_eq!(smoke.stage, Stage::PostSmoke1);
        assert_eq!(smoke.windows, 1);
        assert_eq!(smoke.literals, EXACT_LITERALS.map(str::to_owned));
        assert_eq!(smoke.mode.schema(), "MambaBiFixedAdaExactPostAutoV1");
        assert_eq!(smoke.mode.tuning_revision(), 44);
        assert_eq!(smoke.mode.arms(), ["Legacy", "AUTO", "Fast"]);
        assert_eq!(
            smoke.mode.directions(),
            ["AUTO/Legacy", "Legacy/Fast", "AUTO/Fast"]
        );

        let post = parse_post_config(&exact_post("post101", "101")).unwrap();
        assert_eq!(post.stage, Stage::Post101);
        assert_eq!(post.windows, 101);
        assert!(parse_config(&exact_post("smoke1", "1")).is_err());
        assert!(parse_post_config(&exact_screen()).is_err());
    }

    #[test]
    fn post44_controls_reject_missing_malformed_stale_family_or_subset() {
        for key in [
            "MAMBA_FIXED_ADA_EXACT_POST_AUTO",
            "MAMBA_FIXED_ADA_VENDOR",
            "MAMBA_FIXED_ADA_EXACT_POST_STAGE",
            "MAMBA_FIXED_ADA_EXACT_POST_WINDOWS",
            "MAMBA_FIXED_ADA_EXACT_POST_TOOLKIT",
            "MAMBA_FIXED_ADA_EXACT_POST_LITERALS",
            "MAMBA_FIXED_ADA_EXACT_POST_TUNING_REVISION",
            "MAMBA_FIXED_ADA_EXACT_POST_SOURCE_SHA",
            "MAMBA_FIXED_ADA_EXACT_POST_BINARY_SHA",
            "MAMBA_FIXED_ADA_EXACT_POST_JSONL",
            "MAMBA_FIXED_VENDOR_TILES",
            "MAMBA_FIXED_VENDOR_PATHS",
            "MAMBA_FIXED_VENDOR_EXACT_CC",
        ] {
            let mut environment = exact_post("smoke1", "1");
            environment.remove(key);
            assert!(
                parse_post_config(&environment).is_err(),
                "accepted missing {key}"
            );
        }
        for (key, value) in [
            ("MAMBA_FIXED_ADA_EXACT_POST_AUTO", "true"),
            ("MAMBA_FIXED_ADA_EXACT_POST_STAGE", "confirm101"),
            ("MAMBA_FIXED_ADA_EXACT_POST_WINDOWS", "21"),
            ("MAMBA_FIXED_ADA_EXACT_POST_TOOLKIT", "13.2"),
            ("MAMBA_FIXED_ADA_EXACT_POST_TUNING_REVISION", "43"),
            ("MAMBA_FIXED_ADA_EXACT_POST_LITERALS", "hot_a:0"),
            ("MAMBA_FIXED_VENDOR_TILES", "F32Sm89N64CopyPlan"),
        ] {
            let mut environment = exact_post("smoke1", "1");
            environment.insert(key.into(), value.into());
            assert!(
                parse_post_config(&environment).is_err(),
                "accepted {key}={value}"
            );
        }
        for stale in [
            "MAMBA_FIXED_ADA_TOOLKIT_ADMISSION",
            "MAMBA_FIXED_ADA_ROWS",
            "MAMBA_FIXED_ADA_SCREEN_SHA",
            "MAMBA_FIXED_ADA_EXACT_POST_FAMILY",
            "NVIDIA_TF32_OVERRIDE",
        ] {
            let mut environment = exact_post("smoke1", "1");
            environment.insert(stale.into(), "1".into());
            assert!(
                parse_post_config(&environment).is_err(),
                "accepted stale {stale}"
            );
        }
    }

    #[test]
    fn strict_controls_bind_family_stage_toolkit_source_binary_and_literals() {
        let config = parse_config(&exact_screen()).unwrap();
        assert_eq!(
            config,
            Config {
                mode: RunMode::Task7,
                family: Family::Exact,
                stage: Stage::Screen21,
                windows: 21,
                toolkit: "12.8".into(),
                literals: EXACT_LITERALS.map(str::to_owned).to_vec(),
                source_sha: sha('a'),
                binary_sha: sha('b'),
                screen_sha: None,
                screen_artifact_sha: None,
                jsonl: "/root/evidence/task7-screen.jsonl".into(),
            }
        );

        let mut tf32 = exact_screen();
        tf32.insert("MAMBA_FIXED_ADA_ROWS".into(), "tf32".into());
        tf32.insert("MAMBA_FIXED_ADA_LITERALS".into(), TF32_LITERALS.join(","));
        tf32.insert("MAMBA_FIXED_VENDOR_TILES".into(), "Tf32M64S2".into());
        tf32.insert("MAMBA_FIXED_ADA_TOOLKIT".into(), "13.0".into());
        tf32.insert("MAMBA_FIXED_ADA_STAGE".into(), "smoke1".into());
        tf32.insert("MAMBA_FIXED_ADA_WINDOWS".into(), "1".into());
        let config = parse_config(&tf32).unwrap();
        assert_eq!(
            (config.family, config.stage, config.windows),
            (Family::Tf32, Stage::Smoke1, 1)
        );
    }

    #[test]
    fn strict_controls_reject_missing_malformed_stale_or_cartesian_inputs() {
        for key in [
            "MAMBA_FIXED_ADA_TOOLKIT_ADMISSION",
            "MAMBA_FIXED_ADA_VENDOR",
            "MAMBA_FIXED_ADA_ROWS",
            "MAMBA_FIXED_ADA_LITERALS",
            "MAMBA_FIXED_ADA_WINDOWS",
            "MAMBA_FIXED_ADA_TOOLKIT",
            "MAMBA_FIXED_ADA_TUNING_REVISION",
            "MAMBA_FIXED_ADA_STAGE",
            "MAMBA_FIXED_ADA_SOURCE_SHA",
            "MAMBA_FIXED_ADA_BINARY_SHA",
            "MAMBA_FIXED_ADA_JSONL",
            "MAMBA_FIXED_VENDOR_TILES",
            "MAMBA_FIXED_VENDOR_PATHS",
            "MAMBA_FIXED_VENDOR_EXACT_CC",
        ] {
            let mut environment = exact_screen();
            environment.remove(key);
            assert!(
                parse_config(&environment).is_err(),
                "accepted missing {key}"
            );
        }
        for (key, value) in [
            ("MAMBA_FIXED_ADA_TOOLKIT_ADMISSION", "true"),
            ("MAMBA_FIXED_ADA_VENDOR", "0"),
            ("MAMBA_FIXED_ADA_ROWS", "f32_exact"),
            ("MAMBA_FIXED_ADA_WINDOWS", "101"),
            ("MAMBA_FIXED_ADA_TOOLKIT", "13.2"),
            ("MAMBA_FIXED_ADA_TUNING_REVISION", "42"),
            ("MAMBA_FIXED_ADA_STAGE", "screen"),
            ("MAMBA_FIXED_ADA_SOURCE_SHA", "ABC"),
            ("MAMBA_FIXED_ADA_BINARY_SHA", "abc"),
            ("MAMBA_FIXED_VENDOR_TILES", "Legacy"),
            ("MAMBA_FIXED_VENDOR_PATHS", "graph,eager"),
            ("MAMBA_FIXED_VENDOR_EXACT_CC", "89"),
        ] {
            let mut environment = exact_screen();
            environment.insert(key.into(), value.into());
            assert!(
                parse_config(&environment).is_err(),
                "accepted {key}={value}"
            );
        }
        for stale in [
            "MAMBA_FIXED_ADA_CELLS",
            "MAMBA_FIXED_ADA_BIAS",
            "MAMBA_FIXED_HALF_TILE_CANDIDATE",
            "MAMBA_FIXED_AUTO_VENDOR_ROW",
            "MAMBA_FIXED_ADA_S3_PAIR",
            "MAMBA_FIXED_VENDOR_FORCE_CANDIDATE",
            "NVIDIA_TF32_OVERRIDE",
        ] {
            let mut environment = exact_screen();
            environment.insert(stale.into(), "1".into());
            assert!(
                parse_config(&environment).is_err(),
                "accepted stale {stale}"
            );
        }
    }

    #[test]
    fn literal_inventory_is_exact_for_smoke_and_screen_and_subset_only_for_confirm() {
        for literals in [
            "",
            "hot_a:0,hot_a:0,hot_b:0,hot_b:1,hot_d:0,hot_d:1,hot_e:0,hot_e:1",
            "hot_a:0,hot_a:1,hot_b:0,hot_b:1,hot_c:0,hot_d:1,hot_e:0,hot_e:1",
            "hot_a:0,hot_a:1,hot_b:0,hot_b:1,hot_d:0,hot_d:1,hot_e:0",
        ] {
            let mut environment = exact_screen();
            environment.insert("MAMBA_FIXED_ADA_LITERALS".into(), literals.into());
            assert!(parse_config(&environment).is_err(), "accepted {literals:?}");
        }

        let mut confirm = exact_screen();
        confirm.insert("MAMBA_FIXED_ADA_STAGE".into(), "confirm101".into());
        confirm.insert("MAMBA_FIXED_ADA_WINDOWS".into(), "101".into());
        confirm.insert("MAMBA_FIXED_ADA_LITERALS".into(), "hot_b:1,hot_e:0".into());
        confirm.insert("MAMBA_FIXED_ADA_SCREEN_SHA".into(), sha('c'));
        confirm.insert("MAMBA_FIXED_ADA_SCREEN_ARTIFACT_SHA".into(), sha('d'));
        assert_eq!(
            parse_config(&confirm).unwrap().literals,
            ["hot_b:1", "hot_e:0"]
        );
        for bad in ["", "hot_c:0", "hot_b:1,hot_b:1"] {
            confirm.insert("MAMBA_FIXED_ADA_LITERALS".into(), bad.into());
            assert!(parse_config(&confirm).is_err(), "accepted confirm {bad:?}");
        }
    }

    #[test]
    fn mirrored_schedule_reverses_comparison_traversal_and_four_positions() {
        let forward = schedule(0, 0).unwrap();
        assert_eq!(
            forward
                .iter()
                .map(|o| (o.traversal, o.comparison, o.position, o.arm))
                .collect::<Vec<_>>(),
            vec![
                (0, 0, 0, 0),
                (0, 0, 1, 1),
                (0, 0, 2, 1),
                (0, 0, 3, 0),
                (1, 1, 0, 2),
                (1, 1, 1, 0),
                (1, 1, 2, 0),
                (1, 1, 3, 2),
                (2, 2, 0, 2),
                (2, 2, 1, 1),
                (2, 2, 2, 1),
                (2, 2, 3, 2)
            ]
        );
        let reverse = schedule(0, 1).unwrap();
        assert_eq!(
            reverse.iter().map(|o| o.comparison).collect::<Vec<_>>(),
            [2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0]
        );
        assert_eq!(
            reverse.iter().map(|o| o.arm).collect::<Vec<_>>(),
            [1, 2, 2, 1, 0, 2, 2, 0, 1, 0, 0, 1]
        );
        assert_eq!(
            schedule(1, 0)
                .unwrap()
                .iter()
                .map(|o| (o.traversal, o.comparison, o.position, o.arm))
                .collect::<Vec<_>>(),
            reverse
                .iter()
                .map(|o| (o.traversal, o.comparison, o.position, o.arm))
                .collect::<Vec<_>>()
        );
        assert!(schedule(0, 2).is_err());
    }

    #[test]
    fn raw_ratio_is_sum_b_over_sum_a_and_rejects_wrong_or_invalid_records() {
        let mut observations = schedule(0, 0).unwrap()[0..4].to_vec();
        for (observation, us) in observations.iter_mut().zip([10.0, 4.0, 8.0, 20.0]) {
            observation.us = us;
        }
        assert_eq!(ratio(&observations).unwrap(), 0.4);
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            observations[1].us = bad;
            assert!(ratio(&observations).is_err());
        }
        observations[1].us = 4.0;
        observations[1].arm = 2;
        assert!(ratio(&observations).is_err());
    }

    #[test]
    fn launch_contract_rejects_every_physical_field() {
        let expected = LaunchContract {
            symbol: "gemm_bi_nn_tf32_v1_m64n64_bk32_s2",
            grid: (438, 1, 1),
            block: (128, 1, 1),
            dynamic_shared: 32_768,
            pointers: [11, 12, 13, 14],
            parameters: vec![4621, 1928, 384, 1928, 384, 384],
            abi: vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 24)],
            terminal_rejected: true,
        };
        assert!(validate_launch(&expected, &expected).is_ok());
        let mut mutations = Vec::new();
        let mut wrong = expected.clone();
        wrong.symbol = "label_only";
        mutations.push(wrong);
        let mut wrong = expected.clone();
        wrong.grid.0 = 437;
        mutations.push(wrong);
        let mut wrong = expected.clone();
        wrong.block.0 = 256;
        mutations.push(wrong);
        let mut wrong = expected.clone();
        wrong.dynamic_shared = 0;
        mutations.push(wrong);
        let mut wrong = expected.clone();
        wrong.pointers[3] = 0;
        mutations.push(wrong);
        let mut wrong = expected.clone();
        wrong.parameters[1] = 384;
        mutations.push(wrong);
        let mut wrong = expected.clone();
        wrong.abi[4].1 = 32;
        mutations.push(wrong);
        let mut wrong = expected.clone();
        wrong.terminal_rejected = false;
        mutations.push(wrong);
        for mutation in mutations {
            assert!(
                validate_launch(&mutation, &expected).is_err(),
                "accepted {mutation:?}"
            );
        }
        let encoded = launch_json(&expected);
        assert!(encoded.contains("\"driver_abi\":[[0,8],[8,8],[16,8],[24,8],[32,24]]"));
        assert!(encoded.contains("\"terminal_rejected\":true"));
        assert!(!encoded.contains("(0, 8)"));
    }

    #[test]
    fn tf32_actual_auto_contract_is_rna_wide_not_old_m128s2() {
        assert_eq!(Family::Tf32.incumbent(), InferenceTile::Tf32RnaM128N128S3);
        assert_ne!(Family::Tf32.incumbent(), InferenceTile::Tf32M128S2);
        let bundle = [1.0f32.to_bits(), 0, 4621, 1928, 384, 1928, 384, 384];
        fixed_explicit_vendor_rna_wide_graph_contract(
            1,
            "gemm_bi_nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3",
            (111, 1, 1),
            (256, 1, 1),
            98_304,
            bundle,
        )
        .unwrap();
        let expected = LaunchContract {
            symbol: "gemm_bi_nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3",
            grid: (111, 1, 1),
            block: (256, 1, 1),
            dynamic_shared: 98_304,
            pointers: [11, 12, 13, 14],
            parameters: bundle.to_vec(),
            abi: vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)],
            terminal_rejected: true,
        };
        let old_incumbent = LaunchContract {
            symbol: "gemm_bi_nn_tf32_v1_m128n64_bk32_s2",
            grid: (222, 1, 1),
            dynamic_shared: 55_296,
            parameters: vec![4621, 1928, 384, 1928, 384, 384],
            abi: vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 24)],
            ..expected.clone()
        };
        assert!(validate_launch(&old_incumbent, &expected).is_err());
        assert!(
            fixed_explicit_vendor_rna_wide_graph_contract(
                1,
                expected.symbol,
                expected.grid,
                expected.block,
                expected.dynamic_shared,
                [4621, 1928, 384, 1928, 384, 384, 0, 0],
            )
            .is_err(),
            "old 24-byte portable bundle must not pass the RNA 32-byte contract",
        );
    }

    #[test]
    fn prelaunch_gate_rejects_revisions_and_artifact_before_first_launch() {
        let launches = Cell::new(0);
        let launch = || {
            launches.set(launches.get() + 1);
            Ok(())
        };
        assert!(prelaunch_gate(5, 8, None, "artifact", launch).is_ok());
        assert_eq!(launches.get(), 1);
        for (numeric, schedule) in [(4, 8), (5, 7)] {
            launches.set(0);
            assert!(prelaunch_gate(numeric, schedule, None, "artifact", launch).is_err());
            assert_eq!(launches.get(), 0);
        }
        launches.set(0);
        assert!(prelaunch_gate(5, 8, Some("screen-artifact"), "live-artifact", launch).is_err());
        assert_eq!(
            launches.get(),
            0,
            "artifact mismatch must reject before launch"
        );
    }

    #[test]
    fn timing_boundary_rejects_a_b_bias_before_a_restoring_replay() {
        use std::cell::RefCell;
        let saved = vec![1u32, 2, 3, 4];
        for label in ["A", "B", "bias"] {
            let current = RefCell::new(saved.clone());
            current.borrow_mut()[2] ^= 1;
            let replayed = Cell::new(false);
            let result = timing_boundary_gate(
                || unchanged_words(&saved, &current.borrow(), label),
                || {
                    *current.borrow_mut() = saved.clone();
                    replayed.set(true);
                    Ok(())
                },
            );
            assert!(result.is_err(), "timed {label} mutation must reject");
            assert!(
                !replayed.get(),
                "restoring replay ran before {label} rejection"
            );
            assert_ne!(*current.borrow(), saved);
        }
    }

    #[test]
    fn immutable_gate_rejects_changed_a_b_or_bias_and_order_proof_is_directional() {
        let saved = [1u32, 2, 3, 4];
        unchanged_words(&saved, &saved, "A").unwrap();
        for label in ["A", "B", "bias"] {
            let mut changed = saved;
            changed[2] ^= 1;
            assert!(unchanged_words(&saved, &changed, label).is_err());
        }
        let a = [1.0f32; 6];
        let b = [16_777_216.0, 1.0, -16_777_216.0, 2.0, 3.0, 4.0];
        let ordered = ordered_dot(&a, &b, Some(0.25)).unwrap();
        let reversed = ordered_dot(
            &a.iter().copied().rev().collect::<Vec<_>>(),
            &b.iter().copied().rev().collect::<Vec<_>>(),
            Some(0.25),
        )
        .unwrap();
        assert_ne!(ordered, reversed, "order witness must distinguish reversal");
        assert!(ordered_dot(&a, &b[..5], None).is_err());
        assert!(ordered_dot(&[f32::NAN], &[1.0], None).is_err());
    }

    fn closed_configuration(
        windows: usize,
        start: usize,
    ) -> (
        Vec<Observation>,
        Vec<(usize, usize, f64)>,
        Vec<(usize, f64, f64)>,
    ) {
        let mut raw = Vec::new();
        let mut pairs = Vec::new();
        let mut by_comparison: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
        for window in 0..windows {
            let mut scheduled = schedule(window, start).unwrap();
            for observation in &mut scheduled {
                observation.us = 10.0 + observation.arm as f64;
            }
            for bracket in scheduled.chunks_exact(4) {
                let value = ratio(bracket).unwrap();
                pairs.push((window, bracket[0].comparison, value));
                by_comparison[bracket[0].comparison].push(value);
            }
            raw.extend(scheduled);
        }
        let summaries = by_comparison
            .iter()
            .enumerate()
            .map(|(comparison, values)| {
                let (p50, p95) = quantiles(values).unwrap();
                (comparison, p50, p95)
            })
            .collect();
        (raw, pairs, summaries)
    }

    #[test]
    fn raw_closure_recomputes_pairs_and_summaries_and_rejects_forgery() {
        let (raw, pairs, summaries) = closed_configuration(21, 1);
        assert!(validate_configuration(21, 1, &raw, &pairs, &summaries, true).is_ok());
        let mut bad_raw = raw.clone();
        bad_raw.pop();
        assert!(validate_configuration(21, 1, &bad_raw, &pairs, &summaries, true).is_err());
        let mut bad_raw = raw.clone();
        bad_raw[7].position = 0;
        assert!(validate_configuration(21, 1, &bad_raw, &pairs, &summaries, true).is_err());
        let mut bad_pairs = pairs.clone();
        bad_pairs[0].2 += 0.01;
        assert!(validate_configuration(21, 1, &raw, &bad_pairs, &summaries, true).is_err());
        let mut bad_summaries = summaries.clone();
        bad_summaries[0].2 += 0.01;
        assert!(validate_configuration(21, 1, &raw, &pairs, &bad_summaries, true).is_err());
        assert!(validate_configuration(21, 1, &raw, &pairs, &summaries, false).is_err());
    }

    #[test]
    fn real_loss_or_mixed_p95_is_valid_evidence_but_not_admission() {
        assert_eq!(literal_admitted(&[(0.8, 0.9); 4]).unwrap(), true);
        assert_eq!(
            literal_admitted(&[(0.8, 0.9), (0.8, 1.0), (0.8, 0.9), (0.8, 0.9)]).unwrap(),
            false
        );
        assert_eq!(literal_admitted(&[(1.1, 1.2); 4]).unwrap(), false);
        assert!(literal_admitted(&[(0.8, 0.9); 3]).is_err());
        assert!(literal_admitted(&[(f64::NAN, 0.9); 4]).is_err());
    }
}
