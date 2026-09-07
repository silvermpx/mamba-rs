#!/usr/bin/env python3
"""Build and run the frozen Task3 gates on one lower CUDA toolkit.

The sole GPU executor invokes this remotely for 12.8 and 13.0.  It imports
the already-reviewed 13.2 once-21 validator instead of maintaining another
copy of the arithmetic/record-closure checks.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
import time


ROOT = Path("/root/mamba-ada-exact-toolkit-auto-20260907")
ENV_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
ONCE21_RUNNER = Path(
    "/root/evidence-ada-triad-finalist-integration-20260907/"
    "task3-once21-cuda132/run.py"
)
EVIDENCE_ROOT = Path("/root/evidence-ada-triad-finalist-integration-20260907")
GENERATION = "triad-nt-padded36-two-arm1"
EXPECTED_SOURCES = {
    "src/mamba_ssm/gpu/gemm_bi_triad/qualification.rs":
        "0e3f222ea55e9888e2e712a01d02e4060f03de457d4df1948c58dcce47ae5f74",
    "tests/gemm_bi_performance_matrix.rs":
        "3a2dc3aaba9b60ea4f2295260ae41782356eaece48acc8424d44f06a53c43b2d",
    "tests/gemm_bi_tf32_cohort_binding.rs":
        "9a54bcb0d569db72176895577a4dc54f27017ab20be5a0f9085dd02d1c98323d",
    "tests/support/fixed_full_mantissa.rs":
        "234d4d87dd479780a3cacee34e5f7f1e868e0465536caba3866a46a16a39cb22",
    "src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs":
        "7f32111aeda43cfd069c32ccbf1c77b31bb598db805f422d02bc3b295fb2e550",
}
TESTS = {
    "bits": (
        "gemm_bi_tf32_cohort_binding",
        "sm89_nt_compact_finalist_forced_matches_portable_rna_bits",
    ),
    "resource": (
        "mamba_rs",
        "mamba_ssm::gpu::gemm_bi_triad::qualification::tests::"
        "sm89_nt_compact_finalist_resources_k0_and_live_revisions",
    ),
    "once21": (
        "gemm_bi_performance_matrix",
        "sm89_nt_finalist_once21::gemm_bi_sm89_nt_finalist_current_fast_once21",
    ),
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def cache_hashes(cache: Path) -> dict[str, str]:
    return {path.name: sha(path) for path in sorted(cache.iterdir()) if path.is_file()}


def source_manifest() -> dict[str, object]:
    paths = [ROOT / "Cargo.toml", ROOT / "Cargo.lock"]
    for directory in ("src", "kernels", "tests"):
        paths.extend(
            path
            for path in (ROOT / directory).rglob("*")
            if path.is_file()
            and not path.name.startswith("._")
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        )
    sources = {str(path.relative_to(ROOT)): sha(path) for path in sorted(set(paths))}
    mismatch = {
        relative: {"expected": expected, "actual": sources.get(relative)}
        for relative, expected in EXPECTED_SOURCES.items()
        if sources.get(relative) != expected
    }
    if mismatch:
        raise RuntimeError(f"frozen source mismatch: {mismatch}")
    return {
        "schema": "MambaTriadAdaFinalistTask3LowerSourcesV1",
        "expected_frozen": EXPECTED_SOURCES,
        "count": len(sources),
        "sources": sources,
    }


def parse_artifacts(build_log: Path) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for line in build_log.read_text().splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = message.get("target", {}).get("name")
        if (
            message.get("reason") == "compiler-artifact"
            and name in {target for target, _ in TESTS.values()}
            and message.get("executable")
        ):
            artifacts[name] = Path(message["executable"])
    return artifacts


def list_tests(
    evidence: Path,
    artifacts: dict[str, Path],
    environment: dict[str, str],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for label, (target, test) in TESTS.items():
        binary = artifacts[target]
        listed = subprocess.run(
            [str(binary), "--list"],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        (evidence / f"list-{label}.log").write_text(listed.stdout + listed.stderr)
        names = {
            line.removesuffix(": test")
            for line in listed.stdout.splitlines()
            if line.endswith(": test")
        }
        if listed.returncode != 0 or test not in names:
            raise RuntimeError(f"{label} exact test is not listed")
        result[label] = {
            "target": target,
            "test": test,
            "binary": str(binary),
            "binary_sha256": sha(binary),
            "listed_count": len(names),
        }
    return result


def drain_quiet(once21, environment: dict[str, str]) -> tuple[list[object], bool]:
    samples = []
    consecutive = 0
    for _ in range(120):
        sample = once21.telemetry("DRAIN", environment)
        samples.append(sample)
        consecutive = consecutive + 1 if once21.quiet(sample) else 0
        if consecutive >= 5:
            return samples, True
        time.sleep(1)
    return samples, False


def validate_bits(records: list[dict[str, object]]) -> dict[str, object]:
    rows = [record for record in records if record.get("kind") == "sm89_nt_finalist_bits"]
    cases = {str(record.get("case")) for record in rows}
    expected = {
        "d768_in",
        "d768_out",
        "prism",
        "tail_alpha1",
        "prefix_alpha1",
        "tail_alpha_neg075",
        "prefix_alpha_neg075",
    }
    if len(rows) != 7 or cases != expected:
        raise RuntimeError(f"forced-bit record closure changed: {cases}")
    if any(record.get("launches_per_arm") != 4 for record in rows):
        raise RuntimeError("forced-bit launch closure changed")
    return {"rows": len(rows), "cases": sorted(cases)}


def validate_resource(records: list[dict[str, object]]) -> dict[str, object]:
    if [record.get("kind") for record in records] != [
        "sm89_nt_finalist_resources",
        "sm89_nt_finalist_k0_revisions",
    ]:
        raise RuntimeError("resource/K0 record closure changed")
    resource, revisions = records
    required = {
        "local_bytes": 0,
        "static_shared_bytes": 0,
        "dynamic_shared_bytes": 49_152,
        "launch_threads": 256,
        "max_threads": 256,
        "occupancy": 2,
        "required_occupancy": 2,
    }
    for key, expected in required.items():
        if resource.get(key) != expected:
            raise RuntimeError(f"resource field {key} changed: {resource.get(key)}")
    if (
        revisions.get("finalist_revision") != 1
        or revisions.get("portable_revision") != 45
        or revisions.get("shared_route_revision") != 45
    ):
        raise RuntimeError("live revision closure changed")
    return {"resource": resource, "revisions": revisions}


def run_one(
    label: str,
    evidence: Path,
    binary: Path,
    test: str,
    environment: dict[str, str],
    once21,
) -> dict[str, object]:
    pre = once21.telemetry("PRE", environment)
    write_json(evidence / f"{label}-pre.json", pre)
    if not once21.quiet(pre):
        raise RuntimeError(f"{label} PRE is not strict quiet/no-apps")
    args = [str(binary), test, "--exact", "--ignored", "--nocapture"]
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with (evidence / f"{label}.log").open("x") as output:
        run = subprocess.run(
            args,
            cwd=ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    release = once21.telemetry("RELEASE", environment)
    write_json(evidence / f"{label}-release.json", release)
    drain, quiet = drain_quiet(once21, environment)
    write_json(
        evidence / f"{label}-drain.json",
        {
            "schema": "MambaTriadAdaFinalistDrainV1",
            "required_consecutive_quiet": 5,
            "samples": drain,
            "complete": quiet,
        },
    )
    output = (evidence / f"{label}.log").read_text()
    executed_one = "1 passed" in output and "0 passed" not in output
    records = [
        json.loads(line)
        for line in output.splitlines()
        if line.startswith("{") and line.endswith("}")
    ]
    if run.returncode != 0 or not executed_one or not quiet:
        raise RuntimeError(
            f"{label} failed: exit={run.returncode} executed_one={executed_one} drain={quiet}"
        )
    if label == "bits":
        analysis = validate_bits(records)
    elif label == "resource":
        analysis = validate_resource(records)
    else:
        analysis = once21.validate_records(records)
    return {
        "args": args,
        "started_utc": started,
        "exit": run.returncode,
        "executed_one": executed_one,
        "analysis": analysis,
        "pre_utc": pre["utc"],
        "release_utc": release["utc"],
        "release_was_quiet": once21.quiet(release),
        "drain_complete": quiet,
        "drain_samples": len(drain),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("toolkit", choices=("12.8", "13.0"))
    parser.add_argument(
        "--expected-cache-json",
        type=Path,
        required=True,
        help="JSON object mapping every pre-run private-cache filename to SHA256",
    )
    args = parser.parse_args()
    suffix = args.toolkit.replace(".", "")
    evidence = EVIDENCE_ROOT / f"task3-lower-batch-cuda{suffix}"
    evidence.mkdir(parents=True, exist_ok=False)
    env_runner = load_module("task3_env_runner", ENV_RUNNER)
    once21 = load_module("task3_once21_validator", ONCE21_RUNNER)
    context = env_runner.env_for(args.toolkit, GENERATION)
    environment = context["env"]
    cache = Path(context["cache"])
    expected_cache = json.loads(args.expected_cache_json.read_text())
    initial_cache = cache_hashes(cache)
    if stat.S_IMODE(cache.stat().st_mode) != 0o700 or initial_cache != expected_cache:
        raise RuntimeError("private cache does not match the supplied pre-run hash map")

    manifest = source_manifest()
    write_json(evidence / "source-manifest.json", manifest)
    build_pre = once21.telemetry("BUILD_PRE", environment)
    write_json(evidence / "build-pre.json", build_pre)
    if not once21.quiet(build_pre):
        raise RuntimeError("build PRE is not strict quiet/no-apps")
    build_args = [
        "cargo",
        "test",
        "--release",
        "--features",
        str(context["feature"]),
        "--no-run",
        "--lib",
        "--test",
        "gemm_bi_tf32_cohort_binding",
        "--test",
        "gemm_bi_performance_matrix",
        "--message-format=json-render-diagnostics",
    ]
    with (evidence / "build.log").open("x") as output:
        build = subprocess.run(
            build_args,
            cwd=ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    artifacts = parse_artifacts(evidence / "build.log")
    expected_targets = {target for target, _ in TESTS.values()}
    if build.returncode != 0 or set(artifacts) != expected_targets:
        raise RuntimeError(f"build/artifact closure failed: {build.returncode}/{artifacts}")
    listed = list_tests(evidence, artifacts, environment)
    after_build = cache_hashes(cache)
    if after_build != initial_cache:
        raise RuntimeError("host build/list changed the runtime kernel cache")

    results: dict[str, object] = {}
    cache_after_each: dict[str, dict[str, str]] = {}
    for label in ("bits", "resource", "once21"):
        target, test = TESTS[label]
        results[label] = run_one(
            label,
            evidence,
            artifacts[target],
            test,
            environment,
            once21,
        )
        cache_after_each[label] = cache_hashes(cache)
    final_cache = cache_hashes(cache)
    if any(final_cache.get(name) != digest for name, digest in initial_cache.items()):
        raise RuntimeError("pre-existing cache artifact changed")
    if cache_after_each["resource"] != cache_after_each["bits"] or final_cache != cache_after_each["bits"]:
        raise RuntimeError("resource or timing gate changed the post-bit cache")
    new_cache = sorted(set(final_cache) - set(initial_cache))
    if len(new_cache) > 1:
        raise RuntimeError(f"more than one runtime kernel-cache artifact appeared: {new_cache}")

    receipt = {
        "schema": "MambaTriadAdaFinalistTask3LowerBatchReceiptV1",
        "toolkit": context["toolkit"],
        "feature": context["feature"],
        "generation": GENERATION,
        "target": str(context["target"]),
        "cache": str(cache),
        "cache_mode": oct(stat.S_IMODE(cache.stat().st_mode)),
        "cache_reused_not_cold": True,
        "cache_before": initial_cache,
        "cache_after": final_cache,
        "new_cache_files": new_cache,
        "build_args": build_args,
        "build_exit": build.returncode,
        "source_manifest_sha256": sha(evidence / "source-manifest.json"),
        "artifacts": listed,
        "runs": results,
    }
    write_json(evidence / "command.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    print("WRAPPER_COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
