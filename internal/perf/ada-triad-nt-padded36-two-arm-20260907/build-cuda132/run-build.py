#!/usr/bin/env python3
"""Build-only CUDA13.2 receipt for the two padded36 NT discoveries."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time


TASK8_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
ROOT = Path("/root/mamba-ada-exact-toolkit-auto-20260907")
EVIDENCE = Path("/root/evidence-ada-triad-nt-padded36-two-arm-20260907/build-cuda132")
GENERATION = "triad-nt-padded36-two-arm1"
EXPECTED_HEAD = "32485efda68e2ad64e30ad5750407e0ba38055bc"
EXPECTED_FROZEN = {
    "tests/gemm_bi_tf32_nt_compact_xor.rs": "e92648d277340262cfd446bb769c11947981ac18ab1cd6f4e6427d15bb000a37",
    "tests/gemm_bi_tf32_nt_compact_xor.cu": "0c3907f60f4f6f47abafdc3d476bb1956172c443acfd10b11481fd1121117b76",
    "tests/gemm_bi_tf32_nt_padded_copy_plan.cuh": "d7d1bc3850cd6eaeaf1e591b098ba23009d6515f8b1ac55713a9cefdafe86a93",
    "tests/gemm_bi_tf32_nt_padded_ldmatrix.cuh": "f59b762af62f26693d8dd889763d27696f0663ac1a694cbf334ee425ee8ea63b",
    "tests/gemm_bi_tf32_nt_padded_ldmatrix_host.cpp": "acdc18d8bb391548d9e43f4b964f75d910126439f6ec96b62b32299698f228d9",
}
EXCLUDED_WIP = {
    "tests/gemm_bi_sm120_tf32_selector_qualification.rs",
    "tests/support/triad_discovery_samples.rs",
}
EXPECTED_TESTS = {
    "cuda_suite::ada_tf32_nt_padded_copy_plan_discovery_once7",
    "cuda_suite::ada_tf32_nt_padded_ldmatrix_discovery_once7",
}
PRESEED_SOURCE = Path(
    "/root/mamba-kcache-ada-exact-toolkit-auto-triad-nt-compact-xor1-cuda132-20260907"
)
PRESEED = {
    "mamba-kernels-v1-3809af034718aa08dfdfc1b6baeb407715436dc7db87ef39825b033c2fd72541.bin": "5851ebd49595a0b2ef665073289e52f903eec3266deadfc5ad5d9f2096ce6d16",
    "mamba-kernels-v1-822ebf0dda7dcf9c4f5c5a2213884dde42ab45f5cf238479a81856042c302b7c.bin": "5035921fc8e078e5454c9c5392f9e9ee6ca6a3de54c1e08cb54bba432fa37546",
    "mamba-kernels-v1-e4f76515e441adc28862775000fce218e2d2cca61b47ce484e15eb1144c40fb3.bin": "f0be052fc160b9e5f4b6a897878781327968c0015d551128415a94b85e6fc0aa",
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
            raise RuntimeError(f"frozen source mismatch {relative}")
    source_manifest = {
        "schema": "MambaTriadNtPadded36TwoArmBuildSourcesV1",
        "head": EXPECTED_HEAD,
        "count": len(sources),
        "excluded_local_wip": sorted(EXCLUDED_WIP),
        "sources": sources,
    }
    write_json(EVIDENCE / "source-manifest.json", source_manifest)

    preseed = []
    for name, expected in PRESEED.items():
        source = PRESEED_SOURCE / name
        destination = cache / name
        if sha(source) != expected:
            raise RuntimeError(f"preseed source hash changed: {source}")
        resumed_after_wrapper_failure = destination.exists()
        if not resumed_after_wrapper_failure:
            shutil.copy2(source, destination)
            os.chmod(destination, 0o600)
        if sha(destination) != expected:
            raise RuntimeError(f"preseed destination hash mismatch: {destination}")
        preseed.append(
            {
                "source": str(source),
                "destination": str(destination),
                "sha256": expected,
                "source_key_filename_equal": source.name == destination.name,
                "resumed_after_wrapper_failure": resumed_after_wrapper_failure,
            }
        )
    write_json(
        EVIDENCE / "cache-preseed.json",
        {
            "schema": "MambaTriadNtPadded36CachePreseedV1",
            "cold_cache": False,
            "source_cache_mutated": False,
            "artifacts": preseed,
        },
    )

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
    list_exit = None
    actual_tests: set[str] = set()
    if result.returncode == 0 and len(binaries) == 1:
        listing = subprocess.run(
            [str(binaries[0]), "--list"],
            cwd=ROOT,
            env=context["env"],
            text=True,
            capture_output=True,
            check=False,
        )
        (EVIDENCE / "test-list.log").write_text(listing.stdout + listing.stderr)
        list_exit = listing.returncode
        actual_tests = {
            line.removesuffix(": test")
            for line in listing.stdout.splitlines()
            if line.endswith(": test")
        }
    listed = list_exit == 0 and EXPECTED_TESTS.issubset(actual_tests)
    receipt = {
        "schema": "MambaTriadNtPadded36TwoArmBuildReceiptV1",
        "head": EXPECTED_HEAD,
        "toolkit": context["toolkit"],
        "feature": context["feature"],
        "generation": GENERATION,
        "target": str(context["target"]),
        "cache": str(cache),
        "cache_mode": oct(stat.S_IMODE(cache.stat().st_mode)),
        "cold_cache": False,
        "args": args,
        "started_utc": started,
        "exit": result.returncode,
        "list_exit": list_exit,
        "expected_tests": sorted(EXPECTED_TESTS),
        "expected_tests_listed": listed,
        "pre_utc": pre["utc"],
        "release_utc": release["utc"],
        "source_manifest_sha256": sha(EVIDENCE / "source-manifest.json"),
        "cache_preseed_sha256": sha(EVIDENCE / "cache-preseed.json"),
        "binaries": {str(path): sha(path) for path in sorted(binaries)},
    }
    write_json(EVIDENCE / "command.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    print("WRAPPER_COMPLETE")
    return result.returncode if result.returncode != 0 or listed else 97


if __name__ == "__main__":
    sys.exit(main())
