#!/usr/bin/env python3
"""Build-only CUDA13.2 receipt for the bounded NT compact-XOR discovery."""

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
EVIDENCE = Path("/root/evidence-ada-triad-nt-compact-xor-20260907/build-cuda132")
GENERATION = "triad-nt-compact-xor1"
EXPECTED_HEAD = "311e3435607bdb36c0c45c290c77e82f0439edb1"
EXPECTED_FROZEN = {
    "tests/gemm_bi_tf32_nt_compact_xor.rs": "f9feece07feda4b9bac2679fe4c340a617050e8f82edb276a3174e30329a5be4",
    "tests/gemm_bi_tf32_nt_compact_xor.cu": "0c3907f60f4f6f47abafdc3d476bb1956172c443acfd10b11481fd1121117b76",
    "src/mamba_ssm/gpu/gemm_bi_triad/qualification.rs": "5a84ce112bc50d264c8c019fe7aae1f4e40b98949e438add3de92d40e7fab7e9",
}
EXCLUDED_WIP = {
    "tests/gemm_bi_sm120_tf32_selector_qualification.rs",
    "tests/support/triad_discovery_samples.rs",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def telemetry(phase: str, context: dict[str, object]) -> dict[str, object]:
    data: dict[str, object] = {
        "phase": phase,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment_constructor": str(TASK8_RUNNER) + "::env_for",
    }
    environment = context["env"]
    assert isinstance(environment, dict)
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
    write_json(EVIDENCE / (phase.lower() + ".json"), data)
    return data


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("task8_run", TASK8_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Task8 environment constructor")
    task8_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task8_run)
    context = task8_run.env_for("13.2", GENERATION)
    cache = Path(context["cache"])
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    if stat.S_IMODE(cache.stat().st_mode) != 0o700:
        raise RuntimeError(f"private cache mode is not 0700: {cache}")

    paths = (EVIDENCE / "source-paths.txt").read_text().splitlines()
    if len(paths) != len(set(paths)) or paths != sorted(paths):
        raise RuntimeError("source path inventory is not sorted and unique")
    if EXCLUDED_WIP.intersection(paths):
        raise RuntimeError("excluded local WIP entered the build source inventory")
    sources = {relative: sha(ROOT / relative) for relative in paths}
    for relative, expected in EXPECTED_FROZEN.items():
        if sources.get(relative) != expected:
            raise RuntimeError(
                f"frozen source mismatch {relative}: {sources.get(relative)} != {expected}"
            )
    source_manifest = {
        "schema": "MambaTriadNtCompactXorBuildSourcesV1",
        "head": EXPECTED_HEAD,
        "serialization": "JSON object sources has lexicographically sorted relative-path keys",
        "count": len(sources),
        "excluded_local_wip": sorted(EXCLUDED_WIP),
        "sources": sources,
    }
    write_json(EVIDENCE / "source-manifest.json", source_manifest)

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
    receipt = {
        "schema": "MambaTriadNtCompactXorBuildReceiptV1",
        "head": EXPECTED_HEAD,
        "toolkit": context["toolkit"],
        "feature": context["feature"],
        "generation": GENERATION,
        "target": str(context["target"]),
        "cache": str(cache),
        "cache_mode": oct(stat.S_IMODE(cache.stat().st_mode)),
        "args": args,
        "started_utc": started,
        "exit": result.returncode,
        "pre_utc": pre["utc"],
        "release_utc": release["utc"],
        "source_manifest_sha256": sha(EVIDENCE / "source-manifest.json"),
        "binaries": {str(path): sha(path) for path in sorted(binaries)},
    }
    write_json(EVIDENCE / "command.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    print("WRAPPER_COMPLETE")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
