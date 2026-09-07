#!/usr/bin/env python3
"""Run the frozen Task3 SM89 finalist resource/K0/revision gate once."""

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
import time


ENV_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
ROOT = Path("/root/mamba-ada-exact-toolkit-auto-20260907")
EVIDENCE = Path(
    "/root/evidence-ada-triad-finalist-integration-20260907/task3-resource-k0-cuda132"
)
GENERATION = "triad-nt-padded36-two-arm1"
BINARY = Path(
    "/root/target-ada-exact-toolkit-auto-triad-nt-padded36-two-arm1-cuda132-20260907/"
    "release/deps/mamba_rs-029e0500e5246408"
)
BINARY_SHA = "efa0c96e0c3399f12dd402db516f9a8433db8b99455205d259299d11a13f9643"
TEST = "mamba_ssm::gpu::gemm_bi_triad::qualification::tests::sm89_nt_compact_finalist_resources_k0_and_live_revisions"
SOURCE_MANIFEST_SHA = "fd80555ed9ac4f66a936f7adbb6749d94bc3f546b8ee52c82eec6582b67eacda"
EXPECTED_CACHE = {
    "mamba-kernels-v1-3809af034718aa08dfdfc1b6baeb407715436dc7db87ef39825b033c2fd72541.bin": "5851ebd49595a0b2ef665073289e52f903eec3266deadfc5ad5d9f2096ce6d16",
    "mamba-kernels-v1-822ebf0dda7dcf9c4f5c5a2213884dde42ab45f5cf238479a81856042c302b7c.bin": "5035921fc8e078e5454c9c5392f9e9ee6ca6a3de54c1e08cb54bba432fa37546",
    "mamba-kernels-v1-e4f76515e441adc28862775000fce218e2d2cca61b47ce484e15eb1144c40fb3.bin": "f0be052fc160b9e5f4b6a897878781327968c0015d551128415a94b85e6fc0aa",
    "mamba-kernels-v1-e61681f002f8ecdd61677fc89a8f0faeac5ce494db503fcc97fd344633ec2d65.bin": "1da42f2fbc44dfd9093959831296d86de2414a6d9bb042bc1640730c68d36ac5",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_hashes(cache: Path) -> dict[str, str]:
    return {path.name: sha(path) for path in sorted(cache.iterdir()) if path.is_file()}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def telemetry(phase: str, environment: dict[str, str]) -> dict[str, object]:
    value: dict[str, object] = {
        "phase": phase,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment_constructor": str(ENV_RUNNER) + "::env_for",
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
        value[kind] = result.stdout
        value[kind + "_stderr"] = result.stderr
        value[kind + "_exit"] = result.returncode
    return value


def quiet(value: dict[str, object]) -> bool:
    fields = [field.strip() for field in str(value["gpu"]).strip().split(",")]
    return (
        value["gpu_exit"] == 0
        and value["apps_exit"] == 0
        and fields[:3]
        == [
            "GPU-d1edd7be-e88d-aed6-047d-622163306f0e",
            "NVIDIA RTX 6000 Ada Generation",
            "8.9",
        ]
        and fields[3:] == ["0 %", "0 %"]
        and not str(value["apps"]).strip()
    )


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("env_runner", ENV_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load environment constructor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = module.env_for("13.2", GENERATION)
    environment = context["env"]
    cache = Path(context["cache"])
    if sha(BINARY) != BINARY_SHA:
        raise RuntimeError("resource binary changed")
    if stat.S_IMODE(cache.stat().st_mode) != 0o700 or cache_hashes(cache) != EXPECTED_CACHE:
        raise RuntimeError("resource cache binding changed")
    listing = subprocess.run(
        [str(BINARY), "--list"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    (EVIDENCE / "test-list.log").write_text(listing.stdout + listing.stderr)
    names = {
        line.removesuffix(": test")
        for line in listing.stdout.splitlines()
        if line.endswith(": test")
    }
    if listing.returncode != 0 or TEST not in names:
        raise RuntimeError("exact resource test is not listed")
    pre = telemetry("PRE", environment)
    write_json(EVIDENCE / "pre.json", pre)
    if not quiet(pre):
        raise RuntimeError(f"resource PRE is not strict quiet/no-apps: {pre}")
    args = [str(BINARY), TEST, "--exact", "--ignored", "--nocapture"]
    with (EVIDENCE / "test.log").open("x") as output:
        run = subprocess.run(
            args,
            cwd=ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    release = telemetry("RELEASE", environment)
    write_json(EVIDENCE / "release.json", release)
    drain = release
    if not quiet(drain):
        for _ in range(60):
            time.sleep(1)
            drain = telemetry("DRAIN", environment)
            if quiet(drain):
                break
    write_json(EVIDENCE / "drain.json", drain)
    output = (EVIDENCE / "test.log").read_text()
    executed_one = "1 passed" in output and "0 passed" not in output
    records = [
        json.loads(line)
        for line in output.splitlines()
        if line.startswith("{") and line.endswith("}")
    ]
    kinds = [record.get("kind") for record in records]
    receipt = {
        "schema": "MambaTriadAdaFinalistTask3ResourceK0ReceiptV1",
        "toolkit": context["toolkit"],
        "feature": context["feature"],
        "generation": GENERATION,
        "binary": str(BINARY),
        "binary_sha256": BINARY_SHA,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA,
        "cache": str(cache),
        "cache_mode": oct(stat.S_IMODE(cache.stat().st_mode)),
        "cache_artifacts": cache_hashes(cache),
        "args": args,
        "exit": run.returncode,
        "executed_one": executed_one,
        "record_kinds": kinds,
        "pre_utc": pre["utc"],
        "release_utc": release["utc"],
        "release_was_quiet": quiet(release),
        "drain_utc": drain["utc"],
        "drain_quiet": quiet(drain),
    }
    write_json(EVIDENCE / "command.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    print("WRAPPER_COMPLETE")
    ok = (
        run.returncode == 0
        and executed_one
        and kinds == ["sm89_nt_finalist_resources", "sm89_nt_finalist_k0_revisions"]
        and cache_hashes(cache) == EXPECTED_CACHE
        and quiet(drain)
    )
    return 0 if ok else 97


if __name__ == "__main__":
    sys.exit(main())
