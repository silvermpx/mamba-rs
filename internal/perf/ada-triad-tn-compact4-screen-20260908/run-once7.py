#!/usr/bin/env python3
"""Run the one reviewed TN compact-four-warp discovery arm."""

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
import time


ROOT = Path("/root/mamba-ada-exact-toolkit-auto-20260907")
EVIDENCE = Path("/root/evidence-ada-triad-tn-compact4-screen-20260908")
RUN = EVIDENCE / "once7-cuda132"
SOURCE_MANIFEST = EVIDENCE / "source-manifest.json"
ENV_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
TELEMETRY_RUNNER = Path(
    "/root/evidence-ada-triad-nt-padded36-two-arm-20260907/run-arm.py"
)
BASE_MANIFEST = Path(
    "/root/evidence-ada-triad-finalist-integration-20260907/"
    "task3-auto-batch-cuda132/source-manifest.json"
)
GENERATION = "triad-nt-padded36-two-arm1"
BINARY = Path(
    "/root/target-ada-half-auto-task3a-cuda132-20260907/release/deps/"
    "gemm_bi_tf32_nt_compact_xor-72f43542cd9e6a2e"
)
BINARY_SHA = "91bfbe972b96a99b9ca9a6b29e26dfc72264d28233cfee955e5a8c8ff2f729ae"
TEST = "cuda_suite::ada_tf32_tn_prism_compact_four_warp_s2_discovery_once7"
VARIANT = "tn_compact_four_warp_s2_prism"
FROZEN = {
    "tests/gemm_bi_tf32_nt_compact_xor.rs": "8b921bbb65a0310bccf14c8e575d95fce1fed1e706c049bea9f9e8a2904a2e47",
    "tests/support/triad_tn_compact_source.rs": "cfcbe3146b513fdf51ec82e95059d535df0a8fa598f583adfdd41c0bcebd1924",
    "tests/gemm_bi_tf32_tn_compact_xor.cuh": "0e9712b436f38f65ea3d035ced5bd83669dbad784f60a6496e9a34f39172a495",
}


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
    return {path.name: sha(path) for path in sorted(directory.iterdir()) if path.is_file()}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=False)
    env_runner = load("tn_compact4_env", ENV_RUNNER)
    telemetry_runner = load("tn_compact4_telemetry", TELEMETRY_RUNNER)
    context = env_runner.env_for("13.2", GENERATION)
    environment = context["env"]
    cache = Path(context["cache"])
    if stat.S_IMODE(cache.stat().st_mode) != 0o700:
        raise RuntimeError("private cache mode changed")
    cache_before = hashes(cache)
    base = json.loads(BASE_MANIFEST.read_text())
    paths = set(base["sources"])
    paths.update(FROZEN)
    sources = {relative: sha(ROOT / relative) for relative in sorted(paths)}
    for relative, expected in FROZEN.items():
        if sources.get(relative) != expected:
            raise RuntimeError(f"frozen source changed: {relative}")
    manifest = {
        "schema": "MambaBiTf32TnCompact4DiscoverySourcesV1",
        "base_head": "070a4f54",
        "count": len(sources),
        "excluded_local_wip": [
            "tests/gemm_bi_sm120_tf32_selector_qualification.rs",
            "tests/support/triad_discovery_samples.rs",
        ],
        "sources": sources,
    }
    write_json(SOURCE_MANIFEST, manifest)
    if sha(BINARY) != BINARY_SHA:
        raise RuntimeError("test binary changed")
    listed = {
        line.removesuffix(": test")
        for line in (EVIDENCE / "list2.log").read_text().splitlines()
        if line.endswith(": test")
    }
    if TEST not in listed:
        raise RuntimeError("exact test is absent from authoritative list")

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
    schema_counts = {
        "resource": text.count('"schema":"MambaBiTf32TnDiscoveryResourceV1"'),
        "screen": text.count('"schema":"MambaBiTf32TnCompact4DiscoveryScreenV1"'),
        "decision": text.count('"schema":"MambaBiTf32TnCompact4DiscoveryDecisionV1"'),
        "variant": text.count(f'"variant":"{VARIANT}"'),
    }
    complete = executed_one and schema_counts == {
        "resource": 1,
        "screen": 4,
        "decision": 1,
        "variant": 6,
    }
    cache_after = hashes(cache)
    receipt = {
        "schema": "MambaBiTf32TnCompact4DiscoveryRunReceiptV1",
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
        "source_manifest_sha256": sha(SOURCE_MANIFEST),
        "binary_sha256": BINARY_SHA,
        "test_log_sha256": sha(RUN / "test.log"),
        "cache_before": cache_before,
        "cache_after": cache_after,
        "cache_stable": cache_after == cache_before,
        "candidate_ptx_persisted": False,
        "environment_constructor": str(ENV_RUNNER) + "::env_for",
        "telemetry_runner_sha256": sha(TELEMETRY_RUNNER),
    }
    write_json(RUN / "command.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    print("WRAPPER_COMPLETE")
    return result.returncode if result.returncode != 0 or complete else 97


if __name__ == "__main__":
    sys.exit(main())
