#!/usr/bin/env python3
"""Build and run the frozen CUDA 13.2 half-NT compact BK64/S2 screen."""

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
OUT = Path("/root/evidence-ada-triad-half-nt-compact-20260908")
ENV_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
TELEMETRY = Path("/root/evidence-ada-triad-nt-padded36-two-arm-20260907/run-arm.py")
SOURCES = {
    "tests/gemm_bi_typed_parity.rs":
        "c2db48361dc8ee90d511bf7dff04e1d416f5b24ab9045b6ab1e40bb37f2e665e",
    "tests/support/triad_half_nt_compact_source.rs":
        "637dc875d69a9fd08282f2035d97f4254529f651810058880e65d0b3355c4b56",
}
TEST = "ada_half_nt_d768_out_compact_bk64_s2_vs_current_and_fast_discovery_once7"


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
    context = load("half_nt_compact_env", ENV_RUNNER).env_for(
        "13.2", "triad-nt-padded36-two-arm1"
    )
    telemetry = load("half_nt_compact_telemetry", TELEMETRY)
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
        raise RuntimeError("exact NT compact test absent from authoritative list")
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
        "resource": text.count('"schema":"MambaBiHalfNtCompactResourceV1"'),
        "bits": text.count('"schema":"MambaBiHalfNtCompactBitsV1"'),
        "screen": text.count('"schema":"MambaBiHalfNtCompactFastScreenV1"'),
        "decision": text.count('"schema":"MambaBiHalfNtCompactFastDecisionV1"'),
    }
    expected = {"resource": 2, "bits": 16, "screen": 8, "decision": 2}
    exact_one = "test result: ok. 1 passed; 0 failed; 0 ignored;" in text
    after = hashes(cache)
    complete = result.returncode == 0 and exact_one and counts == expected and before == after
    receipt = {
        "schema": "MambaBiHalfNtCompactRunReceiptV1",
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
