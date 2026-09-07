#!/usr/bin/env python3
"""Capture release/cache state after the preserved checkpoint1 wrapper failure."""

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import time


RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
OUT = Path(
    "/root/evidence-ada-triad-finalist-integration-20260907/"
    "task2-build-cuda132-checkpoint1/release-supplement.json"
)
EXPECTED = {
    "mamba-kernels-v1-3809af034718aa08dfdfc1b6baeb407715436dc7db87ef39825b033c2fd72541.bin": "5851ebd49595a0b2ef665073289e52f903eec3266deadfc5ad5d9f2096ce6d16",
    "mamba-kernels-v1-822ebf0dda7dcf9c4f5c5a2213884dde42ab45f5cf238479a81856042c302b7c.bin": "5035921fc8e078e5454c9c5392f9e9ee6ca6a3de54c1e08cb54bba432fa37546",
    "mamba-kernels-v1-e4f76515e441adc28862775000fce218e2d2cca61b47ce484e15eb1144c40fb3.bin": "f0be052fc160b9e5f4b6a897878781327968c0015d551128415a94b85e6fc0aa",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


spec = importlib.util.spec_from_file_location("task8_run", RUNNER)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load environment constructor")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
context = module.env_for("13.2", "triad-nt-padded36-two-arm1")
environment = context["env"]
cache = Path(context["cache"])
artifacts = {
    path.name: sha(path) for path in sorted(cache.iterdir()) if path.is_file()
}
queries = {}
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
    queries[kind] = result.stdout
    queries[kind + "_stderr"] = result.stderr
    queries[kind + "_exit"] = result.returncode
value = {
    "schema": "MambaTriadAdaFinalistTask2FailedBuildReleaseSupplementV1",
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "environment_constructor": str(RUNNER) + "::env_for",
    "cache": str(cache),
    "cache_mode": oct(stat.S_IMODE(cache.stat().st_mode)),
    "cache_artifacts": artifacts,
    "cache_unchanged_old_three": artifacts == EXPECTED,
    "telemetry": queries,
}
OUT.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
print(json.dumps(value, sort_keys=True))
if artifacts != EXPECTED or stat.S_IMODE(cache.stat().st_mode) != 0o700:
    raise SystemExit(97)
