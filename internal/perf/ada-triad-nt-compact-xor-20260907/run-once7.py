#!/usr/bin/env python3
"""Run the single authorized CUDA13.2 compact-XOR discovery test once."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time


TASK8_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
ROOT = Path("/root/mamba-ada-exact-toolkit-auto-20260907")
EVIDENCE = Path("/root/evidence-ada-triad-nt-compact-xor-20260907")
BUILD = EVIDENCE / "build-cuda132"
RUN = EVIDENCE / "once7-cuda132-valid1"
GENERATION = "triad-nt-compact-xor1"
TEST = "cuda_suite::ada_tf32_nt_compact_xor_discovery_once7"
EXPECTED_BINARY_SHA = "6f083ae8be2d7910d31129473c7abd03bc6258c5fa77831d026000ecc4002a66"
EXPECTED_SOURCE_MANIFEST_SHA = "cf249699558d2e8ac7c5af17b5a66f2febf4903bf2957ae725b333e4d89f0a37"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def telemetry(
    phase: str, context: dict[str, object], *, require_quiet: bool
) -> dict[str, object]:
    data: dict[str, object] = {
        "phase": phase,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment_constructor": str(TASK8_RUNNER) + "::env_for",
    }
    environment = context["env"]
    assert isinstance(environment, dict)
    for kind, query in (
        ("gpu", "--query-gpu=uuid,name,compute_cap,utilization.gpu,utilization.memory"),
        ("apps", "--query-compute-apps=pid,gpu_uuid,process_name"),
    ):
        result = subprocess.run(
            ["/usr/bin/nvidia-smi", query, "--format=csv,noheader"],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        data[kind] = result.stdout
        data[kind + "_stderr"] = result.stderr
        data[kind + "_exit"] = result.returncode
    fields = [field.strip() for field in str(data["gpu"]).strip().split(",")]
    identity_ok = (
        data["gpu_exit"] == 0
        and data["apps_exit"] == 0
        and fields[:3]
        == [
            "GPU-d1edd7be-e88d-aed6-047d-622163306f0e",
            "NVIDIA RTX 6000 Ada Generation",
            "8.9",
        ]
        and not str(data["apps"]).strip()
    )
    data["identity_no_apps"] = identity_ok
    data["quiet"] = identity_ok and fields[3:] == ["0 %", "0 %"]
    write_json(RUN / (phase.lower() + ".json"), data)
    if not identity_ok or (require_quiet and not data["quiet"]):
        raise RuntimeError(f"{phase} telemetry gate failed: {data}")
    return data


def current_sources(bound: dict[str, object]) -> dict[str, str]:
    sources = bound["sources"]
    assert isinstance(sources, dict)
    return {relative: sha(ROOT / relative) for relative in sources}


def library_hashes(binary: Path) -> dict[str, str]:
    ldd = subprocess.run(
        ["ldd", str(binary)], text=True, capture_output=True, check=True
    ).stdout
    libraries: dict[str, str] = {}
    for line in ldd.splitlines():
        fields = line.strip().split()
        candidates = [field for field in fields if field.startswith("/")]
        if candidates:
            path = Path(candidates[0])
            if path.is_file():
                libraries[str(path)] = sha(path)
    return libraries


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("task8_run", TASK8_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Task8 environment constructor")
    task8_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task8_run)
    context = task8_run.env_for("13.2", GENERATION)
    cache = Path(context["cache"])
    if stat.S_IMODE(cache.stat().st_mode) != 0o700:
        raise RuntimeError("compact-XOR cache is not private 0700")

    source_manifest_path = BUILD / "source-manifest.json"
    if sha(source_manifest_path) != EXPECTED_SOURCE_MANIFEST_SHA:
        raise RuntimeError("build source manifest digest changed")
    source_manifest = json.loads(source_manifest_path.read_text())
    if current_sources(source_manifest) != source_manifest["sources"]:
        raise RuntimeError("remote source differs from frozen build manifest before run")
    command = json.loads((BUILD / "command.json").read_text())
    if command["exit"] != 0 or len(command["binaries"]) != 1:
        raise RuntimeError("build receipt is not the single successful frozen binary")
    binary = Path(next(iter(command["binaries"])))
    if sha(binary) != EXPECTED_BINARY_SHA:
        raise RuntimeError("compact-XOR test binary digest changed")

    listing = subprocess.run(
        [str(binary), "--list"],
        cwd=ROOT,
        env=context["env"],
        text=True,
        capture_output=True,
        check=False,
    )
    (RUN / "test-list.log").write_text(listing.stdout + listing.stderr)
    exact_names = {
        line.removesuffix(": test")
        for line in listing.stdout.splitlines()
        if line.endswith(": test")
    }
    if listing.returncode != 0 or TEST not in exact_names:
        raise RuntimeError(f"authoritative libtest listing does not contain {TEST}")

    pre = telemetry("PRE", context, require_quiet=True)
    args = [str(binary), TEST, "--ignored", "--exact", "--nocapture"]
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with (RUN / "test.log").open("x") as output:
        result = subprocess.run(
            args,
            cwd=ROOT,
            env=context["env"],
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    test_log = (RUN / "test.log").read_text()
    executed_one = "test result: ok. 1 passed; 0 failed; 0 ignored;" in test_log
    schema_counts = {
        "resource": test_log.count('"schema":"MambaBiTf32NtCompactXorResourceV1"'),
        "screen": test_log.count('"schema":"MambaBiTf32NtCompactXorScreenV1"'),
        "decision": test_log.count('"schema":"MambaBiTf32NtCompactXorDecisionV1"'),
    }
    complete_success = executed_one and schema_counts == {
        "resource": 1,
        "screen": 4,
        "decision": 1,
    }
    release = telemetry("RELEASE", context, require_quiet=False)
    time.sleep(5)
    drain = telemetry("DRAIN", context, require_quiet=True)
    if current_sources(source_manifest) != source_manifest["sources"]:
        raise RuntimeError("remote source differs from frozen build manifest after run")
    cache_artifacts = {
        str(path): sha(path)
        for path in sorted(cache.rglob("*"))
        if path.is_file() and not path.name.startswith("._")
    }
    binding = {
        "schema": "MambaTriadNtCompactXorArtifactBindingV1",
        "head": source_manifest["head"],
        "toolkit": context["toolkit"],
        "feature": context["feature"],
        "generation": GENERATION,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA,
        "source_count": source_manifest["count"],
        "binary": str(binary),
        "binary_sha256": EXPECTED_BINARY_SHA,
        "cache": str(cache),
        "cache_mode": oct(stat.S_IMODE(cache.stat().st_mode)),
        "cache_artifacts": cache_artifacts,
        "libraries": library_hashes(binary),
        "environment_constructor": str(TASK8_RUNNER) + "::env_for",
    }
    write_json(RUN / "artifact-binding.json", binding)
    receipt = {
        "schema": "MambaTriadNtCompactXorRunReceiptV1",
        "args": args,
        "started_utc": started,
        "exit": result.returncode,
        "executed_exactly_one_test": executed_one,
        "schema_counts": schema_counts,
        "complete_success": complete_success,
        "pre_utc": pre["utc"],
        "release_utc": release["utc"],
        "release_quiet": release["quiet"],
        "drain_utc": drain["utc"],
        "drain_quiet": drain["quiet"],
        "artifact_binding_sha256": sha(RUN / "artifact-binding.json"),
        "test_log_sha256": sha(RUN / "test.log"),
    }
    write_json(RUN / "command.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    print("WRAPPER_COMPLETE")
    return result.returncode if result.returncode != 0 or complete_success else 97


if __name__ == "__main__":
    sys.exit(main())
