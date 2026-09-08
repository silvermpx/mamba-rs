#!/usr/bin/env python3
"""Build/run the frozen Ada TF32 TN Prism single-barrier screen."""

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

ROOT = Path("/root/mamba-ada-exact-toolkit-auto-20260907")
OUT = Path("/root/evidence-ada-triad-tf32-tn-single-barrier-20260908")
ENV_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
TELEMETRY = Path("/root/evidence-ada-triad-nt-padded36-two-arm-20260907/run-arm.py")
SOURCES = {
    "tests/gemm_bi_tf32_nt_compact_xor.rs":
        "2587c6b634505710835c3822d642b2e53fa24c8e58ce975cc7f2df81b7cf4570",
    "tests/support/triad_tn_single_barrier_source.rs":
        "0c6e772d1d58c51011f94ea93a0fbad3986409b05462ba860e8b48d8167b574a",
}
TEST = "cuda_suite::ada_tf32_tn_prism_single_barrier_s3_discovery_once7"


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
    context = load("tf32_tn_barrier_env", ENV_RUNNER).env_for(
        "13.2", "triad-nt-padded36-two-arm1"
    )
    telemetry = load("tf32_tn_barrier_telemetry", TELEMETRY)
    env = dict(context["env"])
    cache = Path(context["cache"])
    if stat.S_IMODE(cache.stat().st_mode) != 0o700:
        raise RuntimeError("private cache mode changed")
    actual_sources = {name: sha(ROOT / name) for name in SOURCES}
    if actual_sources != SOURCES:
        raise RuntimeError(f"frozen source mismatch: {actual_sources!r}")
    before = hashes(cache)
    build_args = ["cargo", "test", "--release", "--features", "cuda",
                  "--test", "gemm_bi_tf32_nt_compact_xor", "--no-run"]
    with (OUT / "build.log").open("x") as output:
        build = subprocess.run(build_args, cwd=ROOT, env=env, stdout=output,
                               stderr=subprocess.STDOUT, check=False)
    build_text = (OUT / "build.log").read_text()
    matches = re.findall(r"Executable .* \((/root/[^)]+)\)", build_text)
    if build.returncode != 0 or len(matches) != 1:
        raise RuntimeError(f"focused build failed/ambiguous: {build.returncode=} {matches=}")
    binary = Path(matches[0])
    listing = subprocess.run([str(binary), "--list"], cwd=ROOT, env=env,
                             text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, check=False)
    (OUT / "test-list.log").write_text(listing.stdout)
    listed = {line.removesuffix(": test") for line in listing.stdout.splitlines()
              if line.endswith(": test")}
    if listing.returncode != 0 or TEST not in listed:
        raise RuntimeError("exact single-barrier test absent from authoritative list")
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
        "resource": text.count('"schema":"MambaBiTf32TnDiscoveryResourceV1"'),
        "auto_screen": text.count('"schema":"MambaBiTf32TnSingleBarrierDiscoveryScreenV1"'),
        "fast_screen": text.count('"schema":"MambaBiTf32TnSingleBarrierFastScreenV1"'),
        "decision": text.count('"schema":"MambaBiTf32TnSingleBarrierDiscoveryDecisionV1"'),
    }
    expected = {"resource": 1, "auto_screen": 4, "fast_screen": 4, "decision": 1}
    exact_one = "test result: ok. 1 passed; 0 failed; 0 ignored;" in text
    after = hashes(cache)
    complete = result.returncode == 0 and exact_one and counts == expected and before == after
    receipt = {
        "schema": "MambaBiTf32TnSingleBarrierRunReceiptV1", "sources": SOURCES,
        "build_args": build_args, "build_exit": build.returncode,
        "binary": str(binary), "binary_sha256": sha(binary), "test": TEST,
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
