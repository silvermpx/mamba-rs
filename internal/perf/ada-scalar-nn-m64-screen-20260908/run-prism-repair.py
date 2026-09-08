#!/usr/bin/env python3
"""Run only the reviewed Ada scalar-NN Prism repair screen."""

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
OUT = Path("/root/evidence-ada-scalar-nn-m64-prism-repair-20260908")
ENV_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
TELEMETRY_RUNNER = Path("/root/evidence-ada-triad-nt-padded36-two-arm-20260907/run-arm.py")
GENERATION = "triad-nt-padded36-two-arm1"
SOURCE = "tests/gemm_bi_scalar_nn_m64n64_qualification.rs"
SOURCE_SHA = "6a7354d8b322d4fb4b83301b41cc3f15f1f7d59632f8aa93b212718865494cfd"
TEST = "cuda_qualification::ada_exact_nn_m64n64_prism_repair_discovery_once7"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hashes(directory: Path) -> dict[str, str]:
    return {path.name: sha(path) for path in sorted(directory.iterdir()) if path.is_file()}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    OUT.mkdir(mode=0o700, parents=True, exist_ok=False)
    env_runner = load("scalar_nn_repair_env", ENV_RUNNER)
    telemetry_runner = load("scalar_nn_repair_telemetry", TELEMETRY_RUNNER)
    context = env_runner.env_for("13.2", GENERATION)
    env = context["env"]
    cache = Path(context["cache"])
    if stat.S_IMODE(cache.stat().st_mode) != 0o700:
        raise RuntimeError("private cache mode changed")
    cache_before = hashes(cache)
    if sha(ROOT / SOURCE) != SOURCE_SHA:
        raise RuntimeError("frozen NN repair source changed")

    build_args = [
        "cargo", "test", "--release", "--features", context["feature"],
        "--test", "gemm_bi_scalar_nn_m64n64_qualification", "--no-run",
    ]
    with (OUT / "build.log").open("x") as output:
        build = subprocess.run(build_args, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    matches = re.findall(r"Executable .* \((/root/[^)]+)\)", (OUT / "build.log").read_text())
    if build.returncode != 0 or len(matches) != 1:
        raise RuntimeError(f"focused build failed/ambiguous: exit={build.returncode} matches={matches}")
    binary = Path(matches[0])
    listing = subprocess.run([str(binary), "--list"], cwd=ROOT, env=env, text=True, capture_output=True)
    list_text = listing.stdout + listing.stderr
    (OUT / "test-list.log").write_text(list_text)
    if listing.returncode != 0 or f"{TEST}: test" not in list_text.splitlines():
        raise RuntimeError("exact repair test absent from authoritative list")

    run = OUT / "once7-cuda132"
    run.mkdir()
    pre = telemetry_runner.telemetry("PRE", context, run, True)
    args = [str(binary), TEST, "--ignored", "--exact", "--nocapture"]
    with (run / "test.log").open("x") as output:
        result = subprocess.run(args, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    release = telemetry_runner.telemetry("RELEASE", context, run, False)
    time.sleep(5)
    drain = telemetry_runner.telemetry("DRAIN", context, run, True)
    text = (run / "test.log").read_text()
    executed_one = "test result: ok. 1 passed; 0 failed; 0 ignored;" in text
    counts = {
        "resource": text.count('"schema":"MambaBiScalarNnM64AdaDiscoveryResourceV1"'),
        "screen": text.count('"schema":"MambaBiScalarNnM64AdaDiscoveryScreenV1"'),
        "decision": text.count('"schema":"MambaBiScalarNnM64AdaDiscoveryDecisionV1"'),
        "batch": text.count('"schema":"MambaBiScalarNnM64AdaDiscoveryBatchV1"'),
    }
    complete = executed_one and counts == {"resource": 3, "screen": 4, "decision": 1, "batch": 1}
    receipt = {
        "schema": "MambaBiScalarNnM64AdaPrismRepairReceiptV1",
        "toolkit": "13.2", "feature": context["feature"], "generation": GENERATION,
        "source": SOURCE, "source_sha256": SOURCE_SHA,
        "build_args": build_args, "build_exit": build.returncode,
        "binary": str(binary), "binary_sha256": sha(binary),
        "test": TEST, "test_listed": True, "args": args, "exit": result.returncode,
        "executed_exactly_one_test": executed_one, "schema_counts": counts,
        "complete_success": complete,
        "pre_utc": pre["utc"], "release_utc": release["utc"], "release_quiet": release["quiet"],
        "drain_utc": drain["utc"], "drain_quiet": drain["quiet"],
        "cache_before": cache_before, "cache_after": hashes(cache),
    }
    receipt["cache_stable"] = receipt["cache_before"] == receipt["cache_after"]
    write_json(OUT / "command.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return result.returncode if result.returncode != 0 or complete else 97


if __name__ == "__main__":
    sys.exit(main())
