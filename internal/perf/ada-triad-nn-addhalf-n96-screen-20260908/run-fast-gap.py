#!/usr/bin/env python3
"""Run the reviewed CUDA 13.2 N96-versus-Fast once7 screen."""

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
import time


ROOT = Path("/root/mamba-ada-exact-toolkit-auto-20260907")
EVIDENCE = Path(
    "/root/evidence-ada-triad-nn-addhalf-n96-screen-20260908/fast-gap-cuda132"
)
RUN = EVIDENCE / "once7"
ENV_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
TELEMETRY_RUNNER = Path(
    "/root/evidence-ada-triad-nt-padded36-two-arm-20260907/run-arm.py"
)
GENERATION = "triad-nt-padded36-two-arm1"
BINARY = Path(
    "/root/target-ada-exact-toolkit-auto-triad-nt-padded36-two-arm1-cuda132-20260907/"
    "release/deps/gemm_bi_fixed_tf32_n96_discovery-77fa7e0a95565ad1"
)
BINARY_SHA = "9b4a71cf392db73377e89c817b5418824ec8338848a8959d34efa8755420cf8d"
TEST = "triad_nn_add_half_screen::triad_nn_d768_out_add_half_n96_fast_gap_once7"


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


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=False)
    env_runner = load("n96_fast_gap_env", ENV_RUNNER)
    telemetry_runner = load("n96_fast_gap_telemetry", TELEMETRY_RUNNER)
    context = env_runner.env_for("13.2", GENERATION)
    environment = dict(context["env"])
    environment["MAMBA_TRIAD_NN_N96_DISCOVERY"] = "1"
    cache = Path(context["cache"])
    if stat.S_IMODE(cache.stat().st_mode) != 0o700:
        raise RuntimeError("private cache mode changed")
    cache_before = hashes(cache)
    if sha(BINARY) != BINARY_SHA:
        raise RuntimeError("test binary changed")
    listed = {
        line.removesuffix(": test")
        for line in (EVIDENCE / "test-list.log").read_text().splitlines()
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
            env=environment,
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
        "physical": text.count('"schema":"MambaBiTriadNnN96FastGapPhysicalV1"'),
        "binding": text.count('"schema":"MambaBiTriadNnN96FastGapBindingV1"'),
        "bits": text.count('"schema":"MambaBiTriadNnN96FastGapBitsV1"'),
        "timing": text.count('"schema":"MambaBiTriadNnN96FastGapTimingV1"'),
        "decision": text.count('"schema":"MambaBiTriadNnN96FastGapDecisionV1"'),
        "pairwise_bits": text.count("TRIAD_NN_N96_BITS pair="),
    }
    expected = {
        "physical": 1,
        "binding": 1,
        "bits": 1,
        "timing": 8,
        "decision": 1,
        "pairwise_bits": 3,
    }
    cache_after = hashes(cache)
    complete = executed_one and counts == expected and cache_after == cache_before
    receipt = {
        "schema": "MambaBiTriadNnN96FastGapRunReceiptV1",
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
    write_json(RUN / "command.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    print("WRAPPER_COMPLETE")
    return result.returncode if result.returncode != 0 or complete else 97


if __name__ == "__main__":
    sys.exit(main())
