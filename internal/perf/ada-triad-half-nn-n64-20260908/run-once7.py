#!/usr/bin/env python3
"""Build and run the frozen CUDA 13.2 half-NN N64 two-shape screen."""

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
OUT = Path("/root/evidence-ada-triad-half-nn-n64-20260908-attempt2")
ENV_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
TELEMETRY = Path("/root/evidence-ada-triad-nt-padded36-two-arm-20260907/run-arm.py")
SOURCES = {
    "tests/gemm_bi_typed_parity.rs":
        "291e71c2d185b0923187e1ac4dd58e9449886059310d51fd269f339a3f424348",
    "tests/support/triad_half_nn_n64_source.rs":
        "fc552af4f9a64eb0ea5ab965396d369729f4bbfcc24ddd7f3f994937abf8fb5c",
}
TEST = "ada_half_nn_n64_two_shape_vs_current_and_fast_discovery_once7"


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
    return {p.name: sha(p) for p in sorted(directory.iterdir()) if p.is_file()}


def main() -> int:
    OUT.mkdir(mode=0o700, parents=True, exist_ok=False)
    context = load("half_nn_n64_env", ENV_RUNNER).env_for(
        "13.2", "triad-nt-padded36-two-arm1"
    )
    telemetry = load("half_nn_n64_telemetry", TELEMETRY)
    env = dict(context["env"])
    cache = Path(context["cache"])
    if stat.S_IMODE(cache.stat().st_mode) != 0o700:
        raise RuntimeError("private cache mode changed")
    actual_sources = {name: sha(ROOT / name) for name in SOURCES}
    if actual_sources != SOURCES:
        raise RuntimeError(f"frozen source mismatch: {actual_sources!r}")
    before = hashes(cache)
    build_args = [
        "cargo", "test", "--release", "--features", "cuda", "--test",
        "gemm_bi_typed_parity", "--no-run",
    ]
    with (OUT / "build.log").open("x") as output:
        build = subprocess.run(
            build_args, cwd=ROOT, env=env, stdout=output,
            stderr=subprocess.STDOUT, check=False,
        )
    build_text = (OUT / "build.log").read_text()
    matches = re.findall(r"Executable .* \((/root/[^)]+)\)", build_text)
    if build.returncode != 0 or len(matches) != 1:
        raise RuntimeError(
            f"focused build failed/ambiguous: exit={build.returncode} matches={matches!r}"
        )
    binary = Path(matches[0])
    listing = subprocess.run(
        [str(binary), "--list"], cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    (OUT / "test-list.log").write_text(listing.stdout)
    listed = {
        line.removesuffix(": test")
        for line in listing.stdout.splitlines()
        if line.endswith(": test")
    }
    if listing.returncode != 0 or TEST not in listed:
        raise RuntimeError("exact N64 test absent from authoritative list")
    run = OUT / "once7-cuda132"
    run.mkdir()
    pre = telemetry.telemetry("PRE", context, run, True)
    args = [str(binary), TEST, "--ignored", "--exact", "--nocapture"]
    with (run / "test.log").open("x") as output:
        result = subprocess.run(
            args, cwd=ROOT, env=env, stdout=output,
            stderr=subprocess.STDOUT, check=False,
        )
    release = telemetry.telemetry("RELEASE", context, run, False)
    time.sleep(5)
    drain = telemetry.telemetry("DRAIN", context, run, True)
    text = (run / "test.log").read_text()
    counts = {
        "candidate_resource": text.count('"schema":"MambaBiHalfNnN64ResourceV1"'),
        "current_resource": text.count('"schema":"MambaBiHalfNnTileAdaDiscoveryResourceV1"'),
        "screen": text.count('"schema":"MambaBiHalfNnN64AlignedScreenV1"'),
        "decision": text.count('"schema":"MambaBiHalfNnN64AlignedDecisionV1"'),
    }
    expected = {
        "candidate_resource": 4, "current_resource": 4,
        "screen": 32, "decision": 4,
    }
    exact_one = "test result: ok. 1 passed; 0 failed; 0 ignored;" in text
    after = hashes(cache)
    complete = result.returncode == 0 and exact_one and counts == expected and before == after
    receipt = {
        "schema": "MambaBiHalfNnN64RunReceiptV1",
        "sources": SOURCES,
        "build_args": build_args,
        "build_exit": build.returncode,
        "binary": str(binary),
        "binary_sha256": sha(binary),
        "test": TEST,
        "args": args,
        "exit": result.returncode,
        "executed_exactly_one_test": exact_one,
        "schema_counts": counts,
        "complete_success": complete,
        "pre_utc": pre["utc"],
        "release_utc": release["utc"],
        "release_quiet": release["quiet"],
        "drain_utc": drain["utc"],
        "drain_quiet": drain["quiet"],
        "test_log_sha256": sha(run / "test.log"),
        "cache_before": before,
        "cache_after": after,
        "cache_stable": before == after,
    }
    (run / "command.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return result.returncode if result.returncode != 0 or complete else 97


if __name__ == "__main__":
    sys.exit(main())
