#!/usr/bin/env python3
"""Run the frozen BF16 d768-in singleton from the combined e77e build."""

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
import time


ROOT = Path("/root/mamba-ada-exact-toolkit-auto-20260907")
BUILD = Path("/root/evidence-ada-triad-half-tn-compact-20260908-attempt2")
OUT = Path("/root/evidence-ada-triad-half-nn-s3-bf16in-20260908")
ENV_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
TELEMETRY = Path("/root/evidence-ada-triad-nt-padded36-two-arm-20260907/run-arm.py")
TEST = "ada_half_nn_fixed_s3_aligned_bf16_d768_in_confirmation_once7"


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
    prior = json.loads((BUILD / "once7-cuda132" / "command.json").read_text())
    binary = Path(prior["binary"])
    binary_sha = prior["binary_sha256"]
    if sha(binary) != binary_sha:
        raise RuntimeError("combined e77e binary changed")
    listed = {
        line.removesuffix(": test")
        for line in (BUILD / "test-list.log").read_text().splitlines()
        if line.endswith(": test")
    }
    if TEST not in listed:
        raise RuntimeError("exact BF16 singleton absent from authoritative list")
    context = load("half_nn_bf16_env", ENV_RUNNER).env_for(
        "13.2", "triad-nt-padded36-two-arm1"
    )
    telemetry = load("half_nn_bf16_telemetry", TELEMETRY)
    env = dict(context["env"])
    cache = Path(context["cache"])
    if stat.S_IMODE(cache.stat().st_mode) != 0o700:
        raise RuntimeError("private cache mode changed")
    before = hashes(cache)
    run = OUT / "once7-cuda132"
    run.mkdir()
    pre = telemetry.telemetry("PRE", context, run, True)
    args = [str(binary), TEST, "--ignored", "--exact", "--nocapture"]
    with (run / "test.log").open("x") as output:
        result = subprocess.run(args, cwd=ROOT, env=env, stdout=output,
                                stderr=subprocess.STDOUT, check=False)
    release = telemetry.telemetry("RELEASE", context, run, False)
    time.sleep(5)
    drain = telemetry.telemetry("DRAIN", context, run, True)
    text = (run / "test.log").read_text()
    counts = {
        "resource": text.count('"schema":"MambaBiHalfNnTileAdaDiscoveryResourceV1"'),
        "screen": text.count('"schema":"MambaBiHalfNnS3AlignedScreenV1"'),
        "decision": text.count('"schema":"MambaBiHalfNnS3AlignedDecisionV1"'),
    }
    expected = {"resource": 2, "screen": 8, "decision": 1}
    exact_one = "test result: ok. 1 passed; 0 failed; 0 ignored;" in text
    after = hashes(cache)
    complete = result.returncode == 0 and exact_one and counts == expected and before == after
    receipt = {
        "schema": "MambaBiHalfNnS3Bf16InRunReceiptV1",
        "source_sha256": "e77e1961d03d7f26d19ffd1a5fe6514127ead50871768f44ad4eab6d70d160b6",
        "binary": str(binary), "binary_sha256": binary_sha, "test": TEST,
        "args": args, "exit": result.returncode,
        "executed_exactly_one_test": exact_one, "schema_counts": counts,
        "complete_success": complete, "pre_utc": pre["utc"],
        "release_utc": release["utc"], "release_quiet": release["quiet"],
        "drain_utc": drain["utc"], "drain_quiet": drain["quiet"],
        "test_log_sha256": sha(run / "test.log"), "cache_before": before,
        "cache_after": after, "cache_stable": before == after,
    }
    (run / "command.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return result.returncode if result.returncode != 0 or complete else 97


if __name__ == "__main__":
    sys.exit(main())
