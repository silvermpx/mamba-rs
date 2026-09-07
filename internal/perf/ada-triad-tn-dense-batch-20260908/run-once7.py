#!/usr/bin/env python3
"""Run the three reviewed TN dense S3 discovery cells serially."""

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
import time


ROOT = Path("/root/mamba-ada-exact-toolkit-auto-20260907")
EVIDENCE = Path("/root/evidence-ada-triad-tn-dense-batch-20260908")
BUILD = EVIDENCE / "build-cuda132"
SOURCE_MANIFEST = BUILD / "source-manifest.json"
ENV_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
TELEMETRY_RUNNER = Path(
    "/root/evidence-ada-triad-nt-padded36-two-arm-20260907/run-arm.py"
)
GENERATION = "triad-nt-padded36-two-arm1"
BINARY = Path(
    "/root/target-ada-exact-toolkit-auto-triad-nt-padded36-two-arm1-"
    "cuda132-20260907/release/deps/gemm_bi_tf32_nt_compact_xor-72f43542cd9e6a2e"
)
BINARY_SHA = "8b5fb74056b37ff94dd4881891c8a16c29ef86d1a7023444bb8dbd1b866385e1"
ARMS = [
    (
        "d768-in",
        "tn_dense_s3_d768_in",
        "cuda_suite::ada_tf32_tn_dense_s3_d768_in_discovery_once7",
    ),
    (
        "d768-out",
        "tn_dense_s3_d768_out",
        "cuda_suite::ada_tf32_tn_dense_s3_d768_out_discovery_once7",
    ),
    (
        "prism",
        "tn_dense_s3_prism",
        "cuda_suite::ada_tf32_tn_dense_s3_prism_discovery_once7",
    ),
]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(directory: Path) -> dict[str, str]:
    return {path.name: sha(path) for path in sorted(directory.iterdir()) if path.is_file()}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    env_runner = load("tn_dense_env", ENV_RUNNER)
    telemetry_runner = load("tn_dense_telemetry", TELEMETRY_RUNNER)
    context = env_runner.env_for("13.2", GENERATION)
    environment = context["env"]
    cache = Path(context["cache"])
    if stat.S_IMODE(cache.stat().st_mode) != 0o700:
        raise RuntimeError("private cache mode changed")
    if sha(BINARY) != BINARY_SHA:
        raise RuntimeError("test binary changed")
    cache_before = hashes(cache)
    manifest_sha = sha(SOURCE_MANIFEST)

    listing = subprocess.run(
        [str(BINARY), "--list"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    list_text = listing.stdout + listing.stderr
    (EVIDENCE / "test-list.log").write_text(list_text)
    listed = {
        line.removesuffix(": test")
        for line in list_text.splitlines()
        if line.endswith(": test")
    }
    expected = {test for _, _, test in ARMS}
    if listing.returncode != 0 or not expected.issubset(listed):
        raise RuntimeError("authoritative list is missing an exact dense test")

    statuses = []
    for label, variant, test in ARMS:
        run = EVIDENCE / f"once7-{label}-cuda132"
        run.mkdir(parents=True, exist_ok=False)
        pre = telemetry_runner.telemetry("PRE", context, run, True)
        args = [str(BINARY), test, "--ignored", "--exact", "--nocapture"]
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with (run / "test.log").open("x") as output:
            result = subprocess.run(
                args,
                cwd=ROOT,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        release = telemetry_runner.telemetry("RELEASE", context, run, False)
        time.sleep(5)
        drain = telemetry_runner.telemetry("DRAIN", context, run, True)
        text = (run / "test.log").read_text()
        executed_one = "test result: ok. 1 passed; 0 failed; 0 ignored;" in text
        schema_counts = {
            "resource": text.count('"schema":"MambaBiTf32TnDiscoveryResourceV1"'),
            "screen": text.count('"schema":"MambaBiTf32TnDenseDiscoveryScreenV1"'),
            "decision": text.count('"schema":"MambaBiTf32TnDenseDiscoveryDecisionV1"'),
            "variant": text.count(f'"variant":"{variant}"'),
        }
        complete = executed_one and schema_counts == {
            "resource": 1,
            "screen": 4,
            "decision": 1,
            "variant": 6,
        }
        receipt = {
            "schema": "MambaBiTf32TnDenseBatchRunReceiptV1",
            "arm": label,
            "variant": variant,
            "args": args,
            "started_utc": started,
            "exit": result.returncode,
            "executed_exactly_one_test": executed_one,
            "schema_counts": schema_counts,
            "complete_success": complete,
            "pre_utc": pre["utc"],
            "release_utc": release["utc"],
            "release_quiet": release["quiet"],
            "drain_utc": drain["utc"],
            "drain_quiet": drain["quiet"],
            "source_manifest_sha256": manifest_sha,
            "binary_sha256": BINARY_SHA,
            "test_log_sha256": sha(run / "test.log"),
            "cache_before": cache_before,
            "cache_after": hashes(cache),
            "candidate_ptx_persisted": False,
            "environment_constructor": str(ENV_RUNNER) + "::env_for",
            "telemetry_runner_sha256": sha(TELEMETRY_RUNNER),
        }
        receipt["cache_stable"] = receipt["cache_after"] == cache_before
        write_json(run / "command.json", receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)
        statuses.append(result.returncode if result.returncode != 0 or complete else 97)

    print("WRAPPER_COMPLETE", flush=True)
    return next((status for status in statuses if status != 0), 0)


if __name__ == "__main__":
    sys.exit(main())
