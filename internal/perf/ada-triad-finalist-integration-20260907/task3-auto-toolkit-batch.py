#!/usr/bin/env python3
"""Run the frozen Task3 post-AUTO closure on one Ada CUDA toolkit."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
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
TESTS = [
    (
        "lib",
        "mamba_ssm::gpu::gemm_bi_triad::dispatch::tf32_tests::"
        "sm89_finalist_measured_cohorts_select_and_decline_to_prior_routes",
        False,
    ),
    (
        "lib",
        "mamba_ssm::gpu::gemm_bi_triad::dispatch::tf32_tests::"
        "sm89_finalist_is_forced_only_and_uses_only_its_own_pointer_binding",
        False,
    ),
    ("cohort", "sm89_finalist_admitted_cell_filter_is_strict", False),
    (
        "cohort",
        "sm89_nt_compact_finalist_actual_auto_symbols_graphs_and_bits",
        True,
    ),
]
EXPECTED_AUTO_CELLS = {"d768_in", "d768_out", "prism"}
EXPECTED_SYMBOL = "gemm_bi_nt_sm89_mma_tf32_compact8_v1_m128n64_bk32_s2"


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


def hashes(path: Path) -> dict[str, str]:
    return {item.name: sha(item) for item in sorted(path.iterdir()) if item.is_file()}


def source_manifest(expected: dict[str, str]) -> dict[str, object]:
    paths = [ROOT / "Cargo.toml", ROOT / "Cargo.lock"]
    for directory in ("src", "kernels", "tests"):
        paths.extend(
            item
            for item in (ROOT / directory).rglob("*")
            if item.is_file()
            and not item.name.startswith("._")
            and "__pycache__" not in item.parts
            and item.suffix != ".pyc"
        )
    sources = {str(item.relative_to(ROOT)): sha(item) for item in sorted(set(paths))}
    mismatch = {
        relative: {"expected": digest, "actual": sources.get(relative)}
        for relative, digest in expected.items()
        if sources.get(relative) != digest
    }
    if mismatch:
        raise RuntimeError(f"frozen source mismatch: {mismatch}")
    return {
        "schema": "MambaTriadAdaFinalistTask3AutoSourcesV2",
        "count": len(sources),
        "expected_frozen": expected,
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
            and name in ("mamba_rs", "gemm_bi_tf32_cohort_binding")
            and message.get("executable")
        ):
            artifacts["lib" if name == "mamba_rs" else "cohort"] = Path(
                message["executable"]
            )
    return artifacts


def validate_auto_output(text: str) -> dict[str, object]:
    records = [
        json.loads(line)
        for line in text.splitlines()
        if line.startswith("{") and line.endswith("}")
    ]
    rows = [
        record
        for record in records
        if record.get("kind") == "sm89_nt_finalist_actual_auto_bits"
    ]
    skips = [
        record
        for record in records
        if record.get("kind") == "sm89_nt_finalist_actual_auto_skip"
    ]
    cells = {str(record.get("case")) for record in rows}
    if len(rows) != 3 or cells != EXPECTED_AUTO_CELLS or skips:
        raise RuntimeError(f"actual-AUTO record closure changed: rows={len(rows)}, cells={cells}")
    for record in rows:
        if record.get("symbol") != EXPECTED_SYMBOL or record.get("launches_per_arm") != 4:
            raise RuntimeError("actual-AUTO symbol/launch closure changed")
    return {"rows": 3, "cells": sorted(cells), "skips": 0}


def validate_nonignored_lib_output(text: str) -> dict[str, int]:
    matches = re.findall(
        r"test result: ok\. (\d+) passed; 0 failed; (\d+) ignored; "
        r"0 measured; 0 filtered out",
        text,
    )
    if len(matches) != 1:
        raise RuntimeError(f"nonignored library summary changed: {matches}")
    passed, ignored = (int(value) for value in matches[0])
    if passed == 0:
        raise RuntimeError("nonignored library batch executed zero tests")
    return {"passed": passed, "ignored": ignored}


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("toolkit", choices=("12.8", "13.0", "13.2"))
    parser.add_argument(
        "--expected-sources-json",
        required=True,
        type=Path,
        help="JSON object mapping frozen repo-relative source paths to SHA256",
    )
    parser.add_argument(
        "--expected-cache-json",
        required=True,
        type=Path,
        help="JSON object mapping all four private-cache filenames to SHA256",
    )
    args = parser.parse_args()
    suffix = args.toolkit.replace(".", "")
    evidence = EVIDENCE_ROOT / f"task3-auto-batch-cuda{suffix}"
    evidence.mkdir(parents=True, exist_ok=False)
    env_runner = load_module("task3_auto_env", ENV_RUNNER)
    once21 = load_module("task3_auto_telemetry", ONCE21_RUNNER)
    context = env_runner.env_for(args.toolkit, GENERATION)
    environment = context["env"]
    cache = Path(context["cache"])
    expected_sources = json.loads(args.expected_sources_json.read_text())
    expected_cache = json.loads(args.expected_cache_json.read_text())
    if not isinstance(expected_sources, dict) or not expected_sources:
        raise RuntimeError("expected source pins must be a nonempty JSON object")
    if not isinstance(expected_cache, dict) or len(expected_cache) != 4:
        raise RuntimeError("expected cache manifest must contain exactly four artifacts")
    if stat.S_IMODE(cache.stat().st_mode) != 0o700 or hashes(cache) != expected_cache:
        raise RuntimeError("private four-artifact cache binding changed")
    write_json(evidence / "source-manifest.json", source_manifest(expected_sources))

    pre = once21.telemetry("BUILD_PRE", environment)
    write_json(evidence / "build-pre.json", pre)
    if not once21.quiet(pre):
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
    if build.returncode != 0 or set(artifacts) != {"lib", "cohort"}:
        raise RuntimeError(f"build closure failed: {build.returncode}/{artifacts}")

    lists: dict[str, object] = {}
    for label, binary in artifacts.items():
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
        expected = {test for target, test, _ in TESTS if target == label}
        if listed.returncode != 0 or not expected.issubset(names):
            raise RuntimeError(f"{label} exact test list changed")
        lists[label] = {
            "binary": str(binary),
            "binary_sha256": sha(binary),
            "count": len(names),
            "expected": sorted(expected),
        }

    regression_pre = once21.telemetry("PRE", environment)
    write_json(evidence / "lib-nonignored-pre.json", regression_pre)
    if not once21.quiet(regression_pre):
        raise RuntimeError("nonignored library PRE is not strict quiet/no-apps")
    regression_args = [str(artifacts["lib"]), "--test-threads=1"]
    with (evidence / "lib-nonignored.log").open("x") as output:
        regression = subprocess.run(
            regression_args,
            cwd=ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    regression_text = (evidence / "lib-nonignored.log").read_text()
    regression_analysis = validate_nonignored_lib_output(regression_text)
    regression_release = once21.telemetry("RELEASE", environment)
    write_json(evidence / "lib-nonignored-release.json", regression_release)
    if regression.returncode != 0:
        raise RuntimeError(f"nonignored library batch failed: {regression.returncode}")

    results = []
    for index, (target, test, ignored) in enumerate(TESTS, 1):
        run_pre = once21.telemetry("PRE", environment)
        write_json(evidence / f"test-{index}-pre.json", run_pre)
        if not once21.quiet(run_pre):
            raise RuntimeError(f"test {index} PRE is not strict quiet/no-apps")
        run_args = [str(artifacts[target]), test, "--exact"]
        if ignored:
            run_args.append("--ignored")
        run_args.append("--nocapture")
        with (evidence / f"test-{index}.log").open("x") as output:
            run = subprocess.run(
                run_args,
                cwd=ROOT,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        text = (evidence / f"test-{index}.log").read_text()
        executed_one = "1 passed" in text and "0 passed" not in text
        release = once21.telemetry("RELEASE", environment)
        write_json(evidence / f"test-{index}-release.json", release)
        if run.returncode != 0 or not executed_one:
            raise RuntimeError(
                f"test {index} failed: exit={run.returncode}, executed_one={executed_one}"
            )
        analysis = validate_auto_output(text) if index == 4 else {"executed_one": True}
        results.append(
            {
                "target": target,
                "test": test,
                "ignored": ignored,
                "args": run_args,
                "exit": run.returncode,
                "executed_one": executed_one,
                "analysis": analysis,
                "pre_utc": run_pre["utc"],
                "release_utc": release["utc"],
                "release_was_quiet": once21.quiet(release),
            }
        )

    drain, quiet = drain_quiet(once21, environment)
    write_json(
        evidence / "drain.json",
        {
            "schema": "MambaTriadAdaFinalistDrainV1",
            "required_consecutive_quiet": 5,
            "samples": drain,
            "complete": quiet,
        },
    )
    if not quiet or hashes(cache) != expected_cache:
        raise RuntimeError("drain/cache closure failed")
    receipt = {
        "schema": "MambaTriadAdaFinalistTask3AutoReceiptV2",
        "toolkit": context["toolkit"],
        "feature": context["feature"],
        "generation": GENERATION,
        "build_args": build_args,
        "build_exit": build.returncode,
        "source_manifest_sha256": sha(evidence / "source-manifest.json"),
        "artifacts": lists,
        "nonignored_library": {
            "args": regression_args,
            "exit": regression.returncode,
            "analysis": regression_analysis,
            "pre_utc": regression_pre["utc"],
            "release_utc": regression_release["utc"],
            "release_was_quiet": once21.quiet(regression_release),
        },
        "runs": results,
        "cache_artifacts": hashes(cache),
        "drain_complete": True,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(evidence / "command.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    print("WRAPPER_COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
