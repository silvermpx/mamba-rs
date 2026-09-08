#!/usr/bin/env python3
"""Run the CUDA 13.2 Half NN Fast denominator diagnostic."""

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
import time

ROOT = Path("/root/mamba-ada-exact-toolkit-auto-20260907")
EVIDENCE = Path("/root/evidence-ada-triad-half-nn-fast-diag-20260908")
RUN = EVIDENCE / "cuda132"
ENV_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
TELEMETRY_RUNNER = Path("/root/evidence-ada-triad-nt-padded36-two-arm-20260907/run-arm.py")
BINARY = Path(
    "/root/target-ada-exact-toolkit-auto-triad-nt-padded36-two-arm1-cuda132-20260907/"
    "release/deps/gemm_bi_typed_parity-94f76dadc2c09cc4"
)
BINARY_SHA = "817b1aa56421a53a70e6fa221127ccde114374139d4a5a4beab6ba305d50428f"
TEST = "ada_half_nn_d768_out_fast_denominator_diagnostic"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(directory):
    return {p.name: sha(p) for p in sorted(directory.iterdir()) if p.is_file()}


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    env_runner = load("diag_env", ENV_RUNNER)
    telemetry = load("diag_telemetry", TELEMETRY_RUNNER)
    context = env_runner.env_for("13.2", "triad-nt-padded36-two-arm1")
    env = dict(context["env"])
    cache = Path(context["cache"])
    if stat.S_IMODE(cache.stat().st_mode) != 0o700 or sha(BINARY) != BINARY_SHA:
        raise RuntimeError("private cache mode or binary binding changed")
    before = hashes(cache)
    listed = {
        line.removesuffix(": test")
        for line in (EVIDENCE / "test-list.log").read_text().splitlines()
        if line.endswith(": test")
    }
    if TEST not in listed:
        raise RuntimeError("exact diagnostic absent from authoritative list")
    pre = telemetry.telemetry("PRE", context, RUN, True)
    args = [str(BINARY), TEST, "--ignored", "--exact", "--nocapture"]
    with (RUN / "test.log").open("x") as output:
        result = subprocess.run(args, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    release = telemetry.telemetry("RELEASE", context, RUN, False)
    time.sleep(5)
    drain = telemetry.telemetry("DRAIN", context, RUN, True)
    text = (RUN / "test.log").read_text()
    exact_one = "test result: ok. 1 passed; 0 failed; 0 ignored;" in text
    records = text.count('"schema":"MambaBiHalfNnFastDenominatorDiagnosticV1"')
    after = hashes(cache)
    complete = exact_one and records == 4 and before == after
    receipt = {
        "schema": "MambaBiHalfNnFastDenominatorDiagnosticReceiptV1",
        "args": args,
        "exit": result.returncode,
        "executed_exactly_one_test": exact_one,
        "diagnostic_records": records,
        "complete_success": complete,
        "pre_utc": pre["utc"],
        "release_utc": release["utc"],
        "release_quiet": release["quiet"],
        "drain_utc": drain["utc"],
        "drain_quiet": drain["quiet"],
        "binary_sha256": BINARY_SHA,
        "test_log_sha256": sha(RUN / "test.log"),
        "cache_before": before,
        "cache_after": after,
        "cache_stable": before == after,
    }
    (RUN / "command.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return result.returncode if result.returncode != 0 or complete else 97


if __name__ == "__main__":
    sys.exit(main())
