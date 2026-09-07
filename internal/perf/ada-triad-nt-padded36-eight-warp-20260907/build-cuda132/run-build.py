#!/usr/bin/env python3
"""Incremental build/list receipt for the bounded padded36 eight-warp arm."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

TASK8_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
ROOT = Path("/root/mamba-ada-exact-toolkit-auto-20260907")
EVIDENCE = Path(
    "/root/evidence-ada-triad-nt-padded36-eight-warp-20260907/build-cuda132"
)
GENERATION = "triad-nt-padded36-two-arm1"
HEAD = "279263019f67131c4ada92b8e574d8570998118e"
RS_SHA = "989acbcf79c07607dc08125e2e8d5141761416dbc7422e44e3e05cde072f32c5"
TEST = "cuda_suite::ada_tf32_nt_padded_eight_warp_discovery_once7"
EXPECTED_CACHE = {
    "mamba-kernels-v1-3809af034718aa08dfdfc1b6baeb407715436dc7db87ef39825b033c2fd72541.bin": "5851ebd49595a0b2ef665073289e52f903eec3266deadfc5ad5d9f2096ce6d16",
    "mamba-kernels-v1-822ebf0dda7dcf9c4f5c5a2213884dde42ab45f5cf238479a81856042c302b7c.bin": "5035921fc8e078e5454c9c5392f9e9ee6ca6a3de54c1e08cb54bba432fa37546",
    "mamba-kernels-v1-e4f76515e441adc28862775000fce218e2d2cca61b47ce484e15eb1144c40fb3.bin": "f0be052fc160b9e5f4b6a897878781327968c0015d551128415a94b85e6fc0aa",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def telemetry(phase: str, context: dict[str, object]) -> dict[str, object]:
    environment = context["env"]
    assert isinstance(environment, dict)
    data: dict[str, object] = {
        "phase": phase,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment_constructor": str(TASK8_RUNNER) + "::env_for",
    }
    for kind, query in (
        ("gpu", "--query-gpu=uuid,name,compute_cap,utilization.gpu,utilization.memory"),
        ("apps", "--query-compute-apps=pid,gpu_uuid,process_name"),
    ):
        result = subprocess.run(
            ["/usr/bin/nvidia-smi", query, "--format=csv,noheader"],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        data[kind] = result.stdout
        data[kind + "_stderr"] = result.stderr
        data[kind + "_exit"] = result.returncode
    fields = [field.strip() for field in str(data["gpu"]).strip().split(",")]
    write_json(EVIDENCE / (phase.lower() + ".json"), data)
    if (
        data["gpu_exit"] != 0
        or data["apps_exit"] != 0
        or fields[:3]
        != [
            "GPU-d1edd7be-e88d-aed6-047d-622163306f0e",
            "NVIDIA RTX 6000 Ada Generation",
            "8.9",
        ]
        or fields[3:] != ["0 %", "0 %"]
        or str(data["apps"]).strip()
    ):
        raise RuntimeError(f"{phase} is not strict quiet/no-apps: {data}")
    return data


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("task8_run", TASK8_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load environment constructor")
    task8_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task8_run)
    context = task8_run.env_for("13.2", GENERATION)
    cache = Path(context["cache"])
    if stat.S_IMODE(cache.stat().st_mode) != 0o700:
        raise RuntimeError("cache is not private 0700")
    cache_hashes = {
        path.name: sha(path) for path in sorted(cache.iterdir()) if path.is_file()
    }
    if cache_hashes != EXPECTED_CACHE:
        raise RuntimeError("preseeded production cache inventory changed")
    paths = (EVIDENCE / "source-paths.txt").read_text().splitlines()
    sources = {relative: sha(ROOT / relative) for relative in paths}
    if len(paths) != 372 or sources["tests/gemm_bi_tf32_nt_compact_xor.rs"] != RS_SHA:
        raise RuntimeError("eight-warp source inventory changed")
    manifest = {
        "schema": "MambaTriadNtPadded36EightWarpBuildSourcesV1",
        "head": HEAD,
        "count": len(sources),
        "excluded_local_wip": [
            "tests/gemm_bi_sm120_tf32_selector_qualification.rs",
            "tests/support/triad_discovery_samples.rs",
        ],
        "sources": sources,
    }
    write_json(EVIDENCE / "source-manifest.json", manifest)
    pre = telemetry("PRE", context)
    args = [
        "cargo",
        "test",
        "--release",
        "--features",
        context["feature"],
        "--test",
        "gemm_bi_tf32_nt_compact_xor",
        "--no-run",
    ]
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with (EVIDENCE / "build.log").open("x") as output:
        result = subprocess.run(
            args,
            cwd=ROOT,
            env=context["env"],
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    release = telemetry("RELEASE", context)
    binaries = [
        path
        for path in (Path(context["target"]) / "release/deps").glob(
            "gemm_bi_tf32_nt_compact_xor-*"
        )
        if path.is_file() and os.access(path, os.X_OK) and "." not in path.name
    ]
    listing = subprocess.run(
        [str(binaries[0]), "--list"],
        cwd=ROOT,
        env=context["env"],
        text=True,
        capture_output=True,
        check=False,
    ) if result.returncode == 0 and len(binaries) == 1 else None
    list_text = "" if listing is None else listing.stdout + listing.stderr
    (EVIDENCE / "test-list.log").write_text(list_text)
    names = {
        line.removesuffix(": test")
        for line in list_text.splitlines()
        if line.endswith(": test")
    }
    listed = listing is not None and listing.returncode == 0 and TEST in names
    receipt = {
        "schema": "MambaTriadNtPadded36EightWarpBuildReceiptV1",
        "head": HEAD,
        "toolkit": context["toolkit"],
        "feature": context["feature"],
        "generation": GENERATION,
        "target": str(context["target"]),
        "cache": str(cache),
        "cache_mode": oct(stat.S_IMODE(cache.stat().st_mode)),
        "cold_cache": False,
        "cache_artifacts": cache_hashes,
        "args": args,
        "started_utc": started,
        "exit": result.returncode,
        "expected_test": TEST,
        "expected_test_listed": listed,
        "pre_utc": pre["utc"],
        "release_utc": release["utc"],
        "source_manifest_sha256": sha(EVIDENCE / "source-manifest.json"),
        "binaries": {str(path): sha(path) for path in sorted(binaries)},
    }
    write_json(EVIDENCE / "command.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    print("WRAPPER_COMPLETE")
    return result.returncode if result.returncode != 0 or listed else 97


if __name__ == "__main__":
    sys.exit(main())
