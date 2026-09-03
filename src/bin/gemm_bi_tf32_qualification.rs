use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Write as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_OUTPUT_TEMP: AtomicU64 = AtomicU64::new(1);

const K_CASES: &str = "0,1,7,8,9,15,16,17,24,31,32,33,65,97,129,257";
const TAIL_CASES: &str = "1,7,8,9,15,16,17,31,32,33,63,64,65,127,128,129";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SuiteArg {
    Full,
    Sanitizer,
}

#[derive(Debug, PartialEq, Eq)]
struct Arguments {
    exact_cc: (u32, u32),
    expected_routes: usize,
    repeat: u32,
    artifact_output: PathBuf,
    driver_abi_proof_output: PathBuf,
    suite: SuiteArg,
}

struct ArgumentCursor {
    arguments: Vec<String>,
    index: usize,
}

impl ArgumentCursor {
    fn new(arguments: Vec<String>) -> Self {
        Self {
            arguments,
            index: 0,
        }
    }

    fn expect(&mut self, expected: &str) -> Result<(), String> {
        let actual = self
            .arguments
            .get(self.index)
            .ok_or_else(|| format!("missing qualification argument {expected}"))?;
        if actual != expected {
            return Err(format!(
                "qualification argument {} must be {expected}, got {actual}",
                self.index
            ));
        }
        self.index += 1;
        Ok(())
    }

    fn value(&mut self, flag: &str) -> Result<String, String> {
        let value = self
            .arguments
            .get(self.index)
            .ok_or_else(|| format!("qualification argument {flag} requires a value"))?
            .clone();
        self.index += 1;
        Ok(value)
    }

    fn is_finished(&self) -> bool {
        self.index == self.arguments.len()
    }

    fn finish(self) -> Result<(), String> {
        if self.is_finished() {
            Ok(())
        } else {
            Err(format!(
                "unexpected qualification argument {}",
                self.arguments[self.index]
            ))
        }
    }
}

fn parse_u32(value: &str, flag: &str) -> Result<u32, String> {
    let parsed = value
        .parse::<u32>()
        .map_err(|_| format!("qualification argument {flag} must be an unsigned integer"))?;
    if parsed.to_string() != value {
        return Err(format!(
            "qualification argument {flag} must use canonical decimal notation"
        ));
    }
    Ok(parsed)
}

fn parse_usize(value: &str, flag: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("qualification argument {flag} must be an unsigned integer"))?;
    if parsed.to_string() != value {
        return Err(format!(
            "qualification argument {flag} must use canonical decimal notation"
        ));
    }
    Ok(parsed)
}

fn parse_cc(value: &str) -> Result<(u32, u32), String> {
    let (major, minor) = value
        .split_once('.')
        .ok_or_else(|| "qualification argument --exact-cc must use major.minor".to_owned())?;
    let cc = (
        parse_u32(major, "--exact-cc")?,
        parse_u32(minor, "--exact-cc")?,
    );
    if format!("{}.{}", cc.0, cc.1) != value {
        return Err("qualification argument --exact-cc must be canonical".to_owned());
    }
    Ok(cc)
}

fn expected_routes_for_cc(cc: (u32, u32)) -> Option<usize> {
    match cc {
        (8, 0 | 6 | 7 | 9) => Some(18),
        (9, 0) => Some(24),
        (10, 0 | 3) | (11, 0) => Some(54),
        (12, 0 | 1) => Some(36),
        _ => None,
    }
}

fn expect_value(cursor: &mut ArgumentCursor, flag: &str, expected: &str) -> Result<(), String> {
    cursor.expect(flag)?;
    let actual = cursor.value(flag)?;
    if actual != expected {
        return Err(format!(
            "qualification argument {flag} must be {expected}, got {actual}"
        ));
    }
    Ok(())
}

fn resolved_output_path(path: &Path) -> Result<PathBuf, String> {
    let name = path
        .file_name()
        .ok_or_else(|| format!("qualification output path {path:?} has no file name"))?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent = std::fs::canonicalize(parent)
        .map_err(|error| format!("resolve qualification output directory {parent:?}: {error}"))?;
    Ok(parent.join(name))
}

fn distinct_output_paths(artifact: &Path, driver_abi: &Path) -> Result<(PathBuf, PathBuf), String> {
    let artifact = resolved_output_path(artifact)?;
    let driver_abi = resolved_output_path(driver_abi)?;
    if artifact == driver_abi {
        return Err("qualification artifact and Driver ABI proof outputs must be distinct".into());
    }
    Ok((artifact, driver_abi))
}

fn parse_arguments<I, S>(arguments: I) -> Result<Arguments, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut cursor = ArgumentCursor::new(arguments.into_iter().map(Into::into).collect());

    cursor.expect("--exact-cc")?;
    let exact_cc = parse_cc(&cursor.value("--exact-cc")?)?;
    let expected_for_cc = expected_routes_for_cc(exact_cc).ok_or_else(|| {
        format!(
            "unsupported TF32 qualification CC {}.{}",
            exact_cc.0, exact_cc.1
        )
    })?;
    expect_value(&mut cursor, "--family", "all-admitted")?;
    cursor.expect("--all-routes")?;
    cursor.expect("--all-m-tiles")?;
    expect_value(&mut cursor, "--ops", "nn,tn,nt")?;
    expect_value(&mut cursor, "--precision", "f32")?;
    expect_value(&mut cursor, "--k-cases", K_CASES)?;
    expect_value(&mut cursor, "--tail-cases", TAIL_CASES)?;

    cursor.expect("--repeat")?;
    let repeat = parse_u32(&cursor.value("--repeat")?, "--repeat")?;
    if repeat == 0 {
        return Err("qualification argument --repeat must be positive".to_owned());
    }
    cursor.expect("--expected-routes")?;
    let expected_routes = parse_usize(&cursor.value("--expected-routes")?, "--expected-routes")?;
    if expected_routes != expected_for_cc {
        return Err(format!(
            "qualification CC {}.{} requires {expected_for_cc} routes, got {expected_routes}",
            exact_cc.0, exact_cc.1
        ));
    }

    cursor.expect("--k0-every-symbol")?;
    cursor.expect("--k0-null-inputs")?;
    cursor.expect("--driver-abi")?;
    cursor.expect("--driver-abi-live-query")?;
    cursor.expect("--driver-abi-proof-output")?;
    let driver_abi_proof_output = PathBuf::from(cursor.value("--driver-abi-proof-output")?);
    cursor.expect("--artifact-output")?;
    let artifact_output = PathBuf::from(cursor.value("--artifact-output")?);
    distinct_output_paths(&artifact_output, &driver_abi_proof_output)?;
    expect_value(&mut cursor, "--report", "json")?;

    let suite = if cursor.is_finished() {
        SuiteArg::Full
    } else {
        expect_value(&mut cursor, "--suite", "sanitizer")?;
        SuiteArg::Sanitizer
    };
    cursor.finish()?;

    Ok(Arguments {
        exact_cc,
        expected_routes,
        repeat,
        artifact_output,
        driver_abi_proof_output,
        suite,
    })
}

struct StagedOutput {
    path: Option<PathBuf>,
}

impl StagedOutput {
    fn path(&self) -> &Path {
        self.path
            .as_deref()
            .expect("staged qualification output is still present")
    }

    fn remove(&mut self) -> Result<(), String> {
        let path = self
            .path
            .as_ref()
            .expect("staged qualification output is still present");
        std::fs::remove_file(path)
            .map_err(|error| format!("remove staged qualification output {path:?}: {error}"))?;
        self.path = None;
        Ok(())
    }
}

impl Drop for StagedOutput {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = std::fs::remove_file(path);
        }
    }
}

fn stage_output(destination: &Path, bytes: &[u8], label: &str) -> Result<StagedOutput, String> {
    let parent = destination
        .parent()
        .expect("resolved qualification output has a parent");
    let name = destination
        .file_name()
        .expect("resolved qualification output has a file name");
    for _ in 0..128 {
        let serial = NEXT_OUTPUT_TEMP.fetch_add(1, Ordering::Relaxed);
        let mut temporary_name = OsString::from(".");
        temporary_name.push(name);
        temporary_name.push(format!(
            ".mamba-qualification.{}.{serial}.tmp",
            std::process::id()
        ));
        let temporary = parent.join(temporary_name);
        let mut file = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => file,
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("stage {label} at {temporary:?}: {error}")),
        };
        let write_result = file.write_all(bytes).and_then(|()| file.sync_all());
        drop(file);
        if let Err(error) = write_result {
            let _ = std::fs::remove_file(&temporary);
            return Err(format!("write and sync staged {label}: {error}"));
        }
        return Ok(StagedOutput {
            path: Some(temporary),
        });
    }
    Err(format!(
        "could not reserve a unique staged qualification output for {label}"
    ))
}

#[cfg(unix)]
fn sync_output_directory(directory: &Path) -> Result<(), String> {
    File::open(directory)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("sync qualification output directory {directory:?}: {error}"))
}

#[cfg(not(unix))]
fn sync_output_directory(_directory: &Path) -> Result<(), String> {
    Ok(())
}

fn sync_output_parents(first: &Path, second: &Path) -> Result<(), String> {
    let first_parent = first
        .parent()
        .expect("resolved qualification output has a parent");
    sync_output_directory(first_parent)?;
    let second_parent = second
        .parent()
        .expect("resolved qualification output has a parent");
    if second_parent != first_parent {
        sync_output_directory(second_parent)?;
    }
    Ok(())
}

fn rollback_created_outputs(paths: &[&Path]) -> Result<(), String> {
    let mut failure = None;
    for path in paths.iter().rev() {
        if let Err(error) = std::fs::remove_file(path) {
            failure
                .get_or_insert_with(|| format!("roll back qualification output {path:?}: {error}"));
        }
    }
    match failure {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn publication_error(error: String, rollback: Result<(), String>) -> String {
    match rollback {
        Ok(()) => error,
        Err(rollback) => format!("{error}; {rollback}"),
    }
}

fn rollback_publication(paths: &[&Path], first: &Path, second: &Path) -> Result<(), String> {
    let removal = rollback_created_outputs(paths);
    let synchronization = sync_output_parents(first, second);
    match (removal, synchronization) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(removal), Err(synchronization)) => Err(format!("{removal}; {synchronization}")),
    }
}

fn publish_outputs(
    artifact_path: &Path,
    artifact: &[u8],
    driver_abi_path: &Path,
    driver_abi: &[u8],
) -> Result<(), String> {
    let (artifact_path, driver_abi_path) = distinct_output_paths(artifact_path, driver_abi_path)?;
    let mut staged_artifact = stage_output(&artifact_path, artifact, "qualification artifact")?;
    let mut staged_driver_abi = stage_output(&driver_abi_path, driver_abi, "Driver ABI proof")?;

    std::fs::hard_link(staged_artifact.path(), &artifact_path).map_err(|error| {
        format!("publish qualification artifact without overwriting {artifact_path:?}: {error}")
    })?;
    if let Err(error) = std::fs::hard_link(staged_driver_abi.path(), &driver_abi_path) {
        let rollback = rollback_publication(&[&artifact_path], &artifact_path, &driver_abi_path);
        return Err(publication_error(
            format!("publish Driver ABI proof without overwriting {driver_abi_path:?}: {error}"),
            rollback,
        ));
    }
    if let Err(error) = sync_output_parents(&artifact_path, &driver_abi_path) {
        let rollback = rollback_publication(
            &[&artifact_path, &driver_abi_path],
            &artifact_path,
            &driver_abi_path,
        );
        return Err(publication_error(error, rollback));
    }
    if let Err(error) = staged_artifact
        .remove()
        .and_then(|()| staged_driver_abi.remove())
    {
        let rollback = rollback_publication(
            &[&artifact_path, &driver_abi_path],
            &artifact_path,
            &driver_abi_path,
        );
        return Err(publication_error(error, rollback));
    }
    if let Err(error) = sync_output_parents(&artifact_path, &driver_abi_path) {
        let rollback = rollback_publication(
            &[&artifact_path, &driver_abi_path],
            &artifact_path,
            &driver_abi_path,
        );
        return Err(publication_error(error, rollback));
    }
    Ok(())
}

#[cfg(not(test))]
fn environment_arguments() -> Result<Vec<String>, String> {
    std::env::args_os()
        .skip(1)
        .map(|argument| {
            argument
                .into_string()
                .map_err(|_| "qualification arguments must be valid UTF-8".to_owned())
        })
        .collect()
}

#[cfg(not(test))]
fn run() -> Result<(), String> {
    use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
        Tf32QualificationConfig, Tf32QualificationSuite, run_tf32_qualification,
    };

    let arguments = parse_arguments(environment_arguments()?)?;
    let device = GpuDevice::new(0)?;
    if device.compute_capability != arguments.exact_cc {
        return Err(format!(
            "qualification requires CC {}.{}, found {}.{}",
            arguments.exact_cc.0,
            arguments.exact_cc.1,
            device.compute_capability.0,
            device.compute_capability.1
        ));
    }
    let context = GpuCtx::new(&device)?;
    let suite = match arguments.suite {
        SuiteArg::Full => Tf32QualificationSuite::Full,
        SuiteArg::Sanitizer => Tf32QualificationSuite::Sanitizer,
    };
    let output = run_tf32_qualification(
        &context,
        Tf32QualificationConfig {
            exact_cc: arguments.exact_cc,
            expected_routes: arguments.expected_routes,
            repeat: arguments.repeat,
            suite,
        },
    )?;
    if output.driver_abi_proof.is_empty() {
        return Err("live Driver ABI proof is empty".into());
    }

    publish_outputs(
        &arguments.artifact_output,
        &output.artifact,
        &arguments.driver_abi_proof_output,
        output.driver_abi_proof.as_bytes(),
    )?;
    std::io::stdout()
        .lock()
        .write_all(output.report_json.as_bytes())
        .map_err(|error| format!("write qualification JSON: {error}"))
}

#[cfg(not(test))]
fn main() {
    if let Err(error) = run() {
        eprintln!("gemm-bi-tf32-qualification: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_route_counts_include_every_sm120_route() {
        assert_eq!(expected_routes_for_cc((12, 0)), Some(36));
        assert_eq!(expected_routes_for_cc((12, 1)), Some(36));
    }

    fn contract_arguments() -> Vec<String> {
        [
            "--exact-cc",
            "8.9",
            "--family",
            "all-admitted",
            "--all-routes",
            "--all-m-tiles",
            "--ops",
            "nn,tn,nt",
            "--precision",
            "f32",
            "--k-cases",
            "0,1,7,8,9,15,16,17,24,31,32,33,65,97,129,257",
            "--tail-cases",
            "1,7,8,9,15,16,17,31,32,33,63,64,65,127,128,129",
            "--repeat",
            "100",
            "--expected-routes",
            "18",
            "--k0-every-symbol",
            "--k0-null-inputs",
            "--driver-abi",
            "--driver-abi-live-query",
            "--driver-abi-proof-output",
            "/tmp/driver-abi.tsv",
            "--artifact-output",
            "/tmp/artifact.bin",
            "--report",
            "json",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    }

    #[test]
    fn parses_the_tracked_qualification_cli() {
        let parsed = parse_arguments(contract_arguments()).expect("parse tracked CLI");
        assert_eq!(parsed.exact_cc, (8, 9));
        assert_eq!(parsed.expected_routes, 18);
        assert_eq!(parsed.repeat, 100);
        assert_eq!(parsed.artifact_output, PathBuf::from("/tmp/artifact.bin"));
        assert_eq!(
            parsed.driver_abi_proof_output,
            PathBuf::from("/tmp/driver-abi.tsv")
        );
        assert_eq!(parsed.suite, SuiteArg::Full);
    }

    #[test]
    fn accepts_only_the_optional_sanitizer_suffix() {
        let mut sanitizer = contract_arguments();
        sanitizer.extend(["--suite".to_owned(), "sanitizer".to_owned()]);
        assert_eq!(
            parse_arguments(sanitizer)
                .expect("parse sanitizer CLI")
                .suite,
            SuiteArg::Sanitizer
        );

        let mut invalid = contract_arguments();
        invalid.extend(["--suite".to_owned(), "full".to_owned()]);
        assert!(parse_arguments(invalid).is_err());
    }

    #[test]
    fn rejects_the_removed_precision_preservation_promise() {
        let mut arguments = contract_arguments();
        let insertion = arguments
            .iter()
            .position(|argument| argument == "--k-cases")
            .expect("K cases flag");
        arguments.splice(
            insertion..insertion,
            [
                "--preserve-existing-precisions".to_owned(),
                "bf16,f16".to_owned(),
            ],
        );

        assert!(parse_arguments(arguments).is_err());
    }

    #[test]
    fn rejects_colliding_output_paths() {
        let mut arguments = contract_arguments();
        let driver = arguments
            .iter()
            .position(|argument| argument == "--driver-abi-proof-output")
            .expect("Driver ABI output flag")
            + 1;
        let artifact = arguments
            .iter()
            .position(|argument| argument == "--artifact-output")
            .expect("artifact output flag")
            + 1;
        arguments[artifact] = arguments[driver].clone();

        assert!(parse_arguments(arguments).is_err());
    }

    #[test]
    fn publishes_both_outputs_without_leaking_staging_files() {
        let directory = tempfile::tempdir().expect("create publication directory");
        let artifact = directory.path().join("artifact.bin");
        let driver_abi = directory.path().join("driver-abi.tsv");

        publish_outputs(&artifact, b"artifact", &driver_abi, b"driver-abi")
            .expect("publish qualification outputs");

        assert_eq!(std::fs::read(&artifact).unwrap(), b"artifact");
        assert_eq!(std::fs::read(&driver_abi).unwrap(), b"driver-abi");
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 2);
    }

    #[test]
    fn preexisting_artifact_is_untouched_and_staging_is_cleaned() {
        let directory = tempfile::tempdir().expect("create publication directory");
        let artifact = directory.path().join("artifact.bin");
        let driver_abi = directory.path().join("driver-abi.tsv");
        std::fs::write(&artifact, b"unrelated").unwrap();

        assert!(publish_outputs(&artifact, b"artifact", &driver_abi, b"driver-abi").is_err());
        assert_eq!(std::fs::read(&artifact).unwrap(), b"unrelated");
        assert!(!driver_abi.exists());
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
    }

    #[test]
    fn preexisting_driver_proof_rolls_back_artifact_and_cleans_staging() {
        let directory = tempfile::tempdir().expect("create publication directory");
        let artifact = directory.path().join("artifact.bin");
        let driver_abi = directory.path().join("driver-abi.tsv");
        std::fs::write(&driver_abi, b"unrelated").unwrap();

        assert!(publish_outputs(&artifact, b"artifact", &driver_abi, b"driver-abi").is_err());
        assert!(!artifact.exists());
        assert_eq!(std::fs::read(&driver_abi).unwrap(), b"unrelated");
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
    }

    #[test]
    fn rejects_cli_drift_and_route_count_mismatches() {
        for invalid in [
            Vec::new(),
            contract_arguments()[1..].to_vec(),
            {
                let mut arguments = contract_arguments();
                arguments[3] = "portable".to_owned();
                arguments
            },
            {
                let mut arguments = contract_arguments();
                arguments[19] = "21".to_owned();
                arguments
            },
            {
                let mut arguments = contract_arguments();
                arguments.push("--extra".to_owned());
                arguments
            },
        ] {
            assert!(parse_arguments(invalid).is_err());
        }
    }
}
