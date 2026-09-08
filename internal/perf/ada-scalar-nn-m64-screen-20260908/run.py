#!/usr/bin/env python3
"""Build, list, and run the reviewed Ada scalar-NN M64 three-cell screen."""

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
OUT = Path("/root/evidence-ada-scalar-nn-m64-screen-20260908")
ENV_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
TELEMETRY_RUNNER = Path(
    "/root/evidence-ada-triad-nt-padded36-two-arm-20260907/run-arm.py"
)
BASE_MANIFEST = Path(
    "/root/evidence-ada-triad-tn-dense-batch-20260908/build-cuda132/source-manifest.json"
)
GENERATION = "triad-nt-padded36-two-arm1"
SOURCE = "tests/gemm_bi_scalar_nn_m64n64_qualification.rs"
SOURCE_SHA = "71dcd7099c76138bde0a6010e1b8221d8c71ce8ac4c302281b9eb20e26498475"
TEST = "cuda_qualification::ada_exact_nn_m64n64_three_cell_discovery_once7"


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
    env_runner = load("scalar_nn_env", ENV_RUNNER)
    telemetry_runner = load("scalar_nn_telemetry", TELEMETRY_RUNNER)
    context = env_runner.env_for("13.2", GENERATION)
    env = context["env"]
    cache = Path(context["cache"])
    if stat.S_IMODE(cache.stat().st_mode) != 0o700:
        raise RuntimeError("private cache mode changed")
    cache_before = hashes(cache)

    base = json.loads(BASE_MANIFEST.read_text())
    sources = {relative: sha(ROOT / relative) for relative in sorted(base["sources"])}
    if sources[SOURCE] != SOURCE_SHA:
        raise RuntimeError("frozen NN source changed")
    manifest = {
        "schema": "MambaBiScalarNnM64AdaDiscoverySourcesV1",
        "base_head": "0e1d0ce4bedd77914f99815ee21da0bbab85c513",
        "count": len(sources),
        "excluded_local_wip": base.get("excluded_local_wip", []),
        "sources": sources,
    }
    write_json(OUT / "source-manifest.json", manifest)

    build_args = [
        "cargo", "test", "--release", "--features", context["feature"],
        "--test", "gemm_bi_scalar_nn_m64n64_qualification", "--no-run",
    ]
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with (OUT / "build.log").open("x") as output:
        build = subprocess.run(
            build_args, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT
        )
    build_text = (OUT / "build.log").read_text()
    matches = re.findall(r"Executable .* \((/root/[^)]+)\)", build_text)
    if build.returncode != 0 or len(matches) != 1:
        raise RuntimeError(f"focused build failed/ambiguous: exit={build.returncode} matches={matches}")
    binary = Path(matches[0])
    binary_sha = sha(binary)
    listing = subprocess.run(
        [str(binary), "--list"], cwd=ROOT, env=env, text=True, capture_output=True
    )
    list_text = listing.stdout + listing.stderr
    (OUT / "test-list.log").write_text(list_text)
    listed = {
        line.removesuffix(": test")
        for line in list_text.splitlines()
        if line.endswith(": test")
    }
    if listing.returncode != 0 or TEST not in listed:
        raise RuntimeError("exact NN test absent from authoritative list")

    run = OUT / "once7-cuda132"
    run.mkdir()
    pre = telemetry_runner.telemetry("PRE", context, run, True)
    args = [str(binary), TEST, "--ignored", "--exact", "--nocapture"]
    with (run / "test.log").open("x") as output:
        result = subprocess.run(
            args, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT
        )
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
    complete = executed_one and counts == {"resource": 5, "screen": 12, "decision": 3, "batch": 1}
    receipt = {
        "schema": "MambaBiScalarNnM64AdaDiscoveryReceiptV1",
        "toolkit": "13.2",
        "feature": context["feature"],
        "generation": GENERATION,
        "build_args": build_args,
        "build_started_utc": started,
        "build_exit": build.returncode,
        "binary": str(binary),
        "binary_sha256": binary_sha,
        "test": TEST,
        "test_listed": True,
        "args": args,
        "exit": result.returncode,
        "executed_exactly_one_test": executed_one,
        "schema_counts": counts,
        "complete_success": complete,
        "source_manifest_sha256": sha(OUT / "source-manifest.json"),
        "pre_utc": pre["utc"],
        "release_utc": release["utc"],
        "release_quiet": release["quiet"],
        "drain_utc": drain["utc"],
        "drain_quiet": drain["quiet"],
        "cache_before": cache_before,
        "cache_after": hashes(cache),
    }
    receipt["cache_stable"] = receipt["cache_before"] == receipt["cache_after"]
    write_json(OUT / "command.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    print("WRAPPER_COMPLETE")
    return result.returncode if result.returncode != 0 or complete else 97


if __name__ == "__main__":
    sys.exit(main())
