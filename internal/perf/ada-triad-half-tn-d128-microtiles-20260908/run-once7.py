#!/usr/bin/env python3
"""Run the reviewed CUDA 13.2 half-TN two-microtile once7 screen."""

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
import time


ROOT = Path("/root/mamba-ada-exact-toolkit-auto-20260907")
EVIDENCE = Path("/root/evidence-ada-triad-half-tn-d128-microtiles-20260908")
RUN = EVIDENCE / "once7-cuda132-fastgraph-repair"
ENV_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
TELEMETRY_RUNNER = Path(
    "/root/evidence-ada-triad-nt-padded36-two-arm-20260907/run-arm.py"
)
BINARY = Path(
    "/root/target-ada-exact-toolkit-auto-triad-nt-padded36-two-arm1-cuda132-20260907/"
    "release/deps/gemm_bi_typed_parity-94f76dadc2c09cc4"
)
BINARY_SHA = "bfbbbae1cba6594c59ec908595231f4b412f0b564066827861fc3492f3f97a1f"
TEST = "ada_half_tn_d128_microtiles_vs_current_and_fast_discovery_once7"


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
    return {
        path.name: sha(path)
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=False)
    env_runner = load("half_tn_env", ENV_RUNNER)
    telemetry_runner = load("half_tn_telemetry", TELEMETRY_RUNNER)
    context = env_runner.env_for("13.2", "triad-nt-padded36-two-arm1")
    env = dict(context["env"])
    cache = Path(context["cache"])
    if stat.S_IMODE(cache.stat().st_mode) != 0o700:
        raise RuntimeError("private cache mode changed")
    cache_before = hashes(cache)
    if sha(BINARY) != BINARY_SHA:
        raise RuntimeError("test binary changed")
    listed = {
        line.removesuffix(": test")
        for line in (EVIDENCE / "test-list5.log").read_text().splitlines()
        if line.endswith(": test")
    }
    if TEST not in listed:
        raise RuntimeError("exact test absent from authoritative list")

    pre = telemetry_runner.telemetry("PRE", context, RUN, True)
    args = [str(BINARY), TEST, "--ignored", "--exact", "--nocapture"]
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with (RUN / "test.log").open("x") as output:
        result = subprocess.run(
            args,
            cwd=ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    release = telemetry_runner.telemetry("RELEASE", context, RUN, False)
    time.sleep(5)
    drain = telemetry_runner.telemetry("DRAIN", context, RUN, True)
    text = (RUN / "test.log").read_text()
    executed_one = "test result: ok. 1 passed; 0 failed; 0 ignored;" in text
    counts = {
        "resource": text.count('"schema":"MambaBiHalfTnMicrotileResourceV1"'),
        "bits": text.count('"schema":"MambaBiHalfTnMicrotileBitsV1"'),
        "screen": text.count('"schema":"MambaBiHalfTnMicrotileScreenV1"'),
        "decision": text.count('"schema":"MambaBiHalfTnMicrotileDecisionV1"'),
    }
    expected = {"resource": 8, "bits": 64, "screen": 32, "decision": 8}
    cache_after = hashes(cache)
    complete = executed_one and counts == expected and cache_after == cache_before
    receipt = {
        "schema": "MambaBiHalfTnMicrotileRunReceiptV1",
        "args": args,
        "started_utc": started,
        "exit": result.returncode,
        "executed_exactly_one_test": executed_one,
        "schema_counts": counts,
        "complete_success": complete,
        "pre_utc": pre["utc"],
        "release_utc": release["utc"],
        "release_quiet": release["quiet"],
        "drain_utc": drain["utc"],
        "drain_quiet": drain["quiet"],
        "binary_sha256": BINARY_SHA,
        "test_log_sha256": sha(RUN / "test.log"),
        "cache_before": cache_before,
        "cache_after": cache_after,
        "cache_stable": cache_after == cache_before,
        "environment_constructor": str(ENV_RUNNER) + "::env_for",
        "telemetry_runner_sha256": sha(TELEMETRY_RUNNER),
    }
    (RUN / "command.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return result.returncode if result.returncode != 0 or complete else 97


if __name__ == "__main__":
    sys.exit(main())
