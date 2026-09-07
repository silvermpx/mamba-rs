#!/usr/bin/env python3
"""Run exactly one reviewed padded36 NT discovery arm."""

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
import time

TASK8_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
ROOT = Path("/root/mamba-ada-exact-toolkit-auto-20260907")
EVIDENCE = Path("/root/evidence-ada-triad-nt-padded36-two-arm-20260907")
BUILD = EVIDENCE / "build-repair-cuda132"
GENERATION = "triad-nt-padded36-two-arm1"
BINARY = Path(
    "/root/target-ada-exact-toolkit-auto-triad-nt-padded36-two-arm1-cuda132-20260907/"
    "release/deps/gemm_bi_tf32_nt_compact_xor-72f43542cd9e6a2e"
)
EXPECTED_BINARY_SHA = "34a67271406e49a62a1baba8765a640bb26b2c0777b339f4545ece40cd25713e"
EXPECTED_SOURCE_MANIFEST_SHA = "db00c029ed47bd6a41a3a3a096f0a101c41d4128dcbb8d2773ae3b8982ba22dc"
ARMS = {
    "copy-plan": (
        "padded_copy_plan",
        "cuda_suite::ada_tf32_nt_padded_copy_plan_discovery_once7",
    ),
    "ldmatrix": (
        "padded_ldmatrix",
        "cuda_suite::ada_tf32_nt_padded_ldmatrix_discovery_once7",
    ),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def telemetry(
    phase: str, context: dict[str, object], run: Path, require_quiet: bool
) -> dict[str, object]:
    environment = context["env"]
    assert isinstance(environment, dict)
    data: dict[str, object] = {
        "phase": phase,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment_constructor": str(TASK8_RUNNER) + "::env_for",
    }
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
    identity = (
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
    data["identity_no_apps"] = identity
    data["quiet"] = identity and fields[3:] == ["0 %", "0 %"]
    write_json(run / (phase.lower() + ".json"), data)
    if not identity or (require_quiet and not data["quiet"]):
        raise RuntimeError(f"{phase} telemetry gate failed: {data}")
    return data


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ARMS:
        raise RuntimeError("usage: run-arm.py copy-plan|ldmatrix")
    arm = sys.argv[1]
    variant, test = ARMS[arm]
    run = EVIDENCE / f"once7-{arm}-cuda132"
    run.mkdir(parents=True, exist_ok=False)

    spec = importlib.util.spec_from_file_location("task8_run", TASK8_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load environment constructor")
    task8_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task8_run)
    context = task8_run.env_for("13.2", GENERATION)
    cache = Path(context["cache"])
    if stat.S_IMODE(cache.stat().st_mode) != 0o700:
        raise RuntimeError("cache is not private 0700")
    if sha(BUILD / "source-manifest.json") != EXPECTED_SOURCE_MANIFEST_SHA:
        raise RuntimeError("repair source manifest digest changed")
    manifest = json.loads((BUILD / "source-manifest.json").read_text())
    current = {relative: sha(ROOT / relative) for relative in manifest["sources"]}
    if current != manifest["sources"]:
        raise RuntimeError("remote source differs from repaired source manifest")
    if sha(BINARY) != EXPECTED_BINARY_SHA:
        raise RuntimeError("repaired test binary digest changed")
    listed = {
        line.removesuffix(": test")
        for line in (BUILD / "test-list.log").read_text().splitlines()
        if line.endswith(": test")
    }
    if test not in listed:
        raise RuntimeError(f"authoritative list is missing {test}")

    pre = telemetry("PRE", context, run, True)
    args = [str(BINARY), test, "--ignored", "--exact", "--nocapture"]
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with (run / "test.log").open("x") as output:
        result = subprocess.run(
            args,
            cwd=ROOT,
            env=context["env"],
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    release = telemetry("RELEASE", context, run, False)
    time.sleep(5)
    drain = telemetry("DRAIN", context, run, True)
    text = (run / "test.log").read_text()
    executed_one = "test result: ok. 1 passed; 0 failed; 0 ignored;" in text
    schema_counts = {
        "resource": text.count('"schema":"MambaBiTf32NtDiscoveryResourceV1"'),
        "screen": text.count('"schema":"MambaBiTf32NtDiscoveryScreenV1"'),
        "decision": text.count('"schema":"MambaBiTf32NtDiscoveryDecisionV1"'),
        "variant": text.count(f'"variant":"{variant}"'),
    }
    complete = executed_one and schema_counts == {
        "resource": 1,
        "screen": 4,
        "decision": 1,
        "variant": 6,
    }
    receipt = {
        "schema": "MambaTriadNtPadded36ArmRunReceiptV1",
        "arm": arm,
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
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA,
        "binary_sha256": EXPECTED_BINARY_SHA,
        "test_log_sha256": sha(run / "test.log"),
        "candidate_ptx_persisted": False,
    }
    write_json(run / "command.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    print("WRAPPER_COMPLETE")
    return result.returncode if result.returncode != 0 or complete else 97


if __name__ == "__main__":
    sys.exit(main())
