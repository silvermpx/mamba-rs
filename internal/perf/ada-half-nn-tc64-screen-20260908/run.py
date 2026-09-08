#!/usr/bin/env python3
"""Build and run the reviewed Ada half NN TC64-vs-TC128 screen."""

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
OUT = Path("/root/evidence-ada-half-nn-tc64-screen-20260908")
ENV_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
TELEMETRY_RUNNER = Path("/root/evidence-ada-triad-nt-padded36-two-arm-20260907/run-arm.py")
GENERATION = "triad-nt-padded36-two-arm1"
SOURCES = {
    "tests/gemm_bi_typed_parity.rs": "982a1b43f355f420ba435fc3e18a4e09d4cd1e0dbfbbce5441429eb32824303d",
    "tests/support/triad_half_tile_screen.rs": "157ef344572deb55153ec4aadebcb754deac8d9f4db7492e55586986c6f04774",
}
TEST = "ada_half_nn_d768_in_tc64_vs_tc128_discovery_once7"


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


def main() -> int:
    OUT.mkdir(mode=0o700, parents=True, exist_ok=False)
    env_runner = load("half_nn_env", ENV_RUNNER)
    telemetry_runner = load("half_nn_telemetry", TELEMETRY_RUNNER)
    context = env_runner.env_for("13.2", GENERATION)
    env = context["env"]
    cache = Path(context["cache"])
    if stat.S_IMODE(cache.stat().st_mode) != 0o700:
        raise RuntimeError("private cache mode changed")
    for relative, expected in SOURCES.items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"frozen source changed: {relative}")
    cache_before = hashes(cache)
    build_args = ["cargo", "test", "--release", "--features", context["feature"],
                  "--test", "gemm_bi_typed_parity", "--no-run"]
    with (OUT / "build.log").open("x") as output:
        build = subprocess.run(build_args, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    matches = re.findall(r"Executable .* \((/root/[^)]+)\)", (OUT / "build.log").read_text())
    if build.returncode != 0 or len(matches) != 1:
        raise RuntimeError(f"focused build failed/ambiguous: exit={build.returncode} matches={matches}")
    binary = Path(matches[0])
    listing = subprocess.run([str(binary), "--list"], cwd=ROOT, env=env, text=True, capture_output=True)
    list_text = listing.stdout + listing.stderr
    (OUT / "test-list.log").write_text(list_text)
    listed = {line.removesuffix(": test") for line in list_text.splitlines() if line.endswith(": test")}
    if listing.returncode != 0 or TEST not in listed:
        raise RuntimeError(f"exact half test absent from authoritative list: {sorted(listed)}")

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
    counts = {
        "resource": text.count('"schema":"MambaBiHalfNnTileAdaDiscoveryResourceV1"'),
        "bits": text.count('"schema":"MambaBiHalfNnTileAdaDiscoveryBitsV1"'),
        "screen": text.count('"schema":"MambaBiHalfNnTileAdaDiscoveryScreenV1"'),
        "decision": text.count('"schema":"MambaBiHalfNnTileAdaDiscoveryDecisionV1"'),
    }
    executed_one = ("test result: ok. 1 passed; 0 failed; 0 ignored;" in text
                    or "test result: FAILED. 0 passed; 1 failed; 0 ignored;" in text)
    schemas_complete = counts == {"resource": 4, "bits": 16, "screen": 8, "decision": 2}
    valid_advance = result.returncode == 0 and schemas_complete and text.count('"decision":"advance_to_full_qualification"') == 2
    valid_stop = result.returncode != 0 and schemas_complete and '"decision":"stop_no_retry"' in text
    complete = executed_one and (valid_advance or valid_stop)
    receipt = {
        "schema": "MambaBiHalfNnTileAdaDiscoveryReceiptV1", "toolkit": "13.2",
        "feature": context["feature"], "generation": GENERATION, "sources": SOURCES,
        "build_args": build_args, "build_exit": build.returncode,
        "binary": str(binary), "binary_sha256": sha(binary), "test": TEST,
        "test_listed": True, "args": args, "exit": result.returncode,
        "executed_exactly_one_test": executed_one, "schema_counts": counts,
        "complete_valid_outcome": complete, "valid_advance": valid_advance, "valid_stop": valid_stop,
        "pre_utc": pre["utc"], "release_utc": release["utc"], "release_quiet": release["quiet"],
        "drain_utc": drain["utc"], "drain_quiet": drain["quiet"],
        "cache_before": cache_before, "cache_after": hashes(cache),
    }
    receipt["cache_stable"] = receipt["cache_before"] == receipt["cache_after"]
    (OUT / "command.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0 if complete else 97


if __name__ == "__main__":
    sys.exit(main())
