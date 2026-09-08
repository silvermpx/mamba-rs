#!/usr/bin/env python3
"""Build and run the reviewed Ada scalar-NT d768-out short screen."""

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
OUT = Path("/root/evidence-ada-scalar-nt-d768-out-screen-20260908")
ENV_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
TELEMETRY_RUNNER = Path("/root/evidence-ada-triad-nt-padded36-two-arm-20260907/run-arm.py")
GENERATION = "triad-nt-padded36-two-arm1"
SOURCE = "tests/gemm_bi_scalar_nt_d768_out_m64n64_tournament.rs"
SOURCE_SHA = "064b9c6e575f8f56f7a6cbfff31ca91481fae2ef6d6f8dace811a2e3dc6a8c3d"
TEST = "cuda_tournament::ada_d768_out_transpose16_m64n64_discovery_once7"


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
    return {path.name: sha(path) for path in sorted(directory.iterdir()) if path.is_file()}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    OUT.mkdir(mode=0o700, parents=True, exist_ok=False)
    env_runner = load("scalar_nt_env", ENV_RUNNER)
    telemetry_runner = load("scalar_nt_telemetry", TELEMETRY_RUNNER)
    context = env_runner.env_for("13.2", GENERATION)
    env = context["env"]
    cache = Path(context["cache"])
    if stat.S_IMODE(cache.stat().st_mode) != 0o700 or sha(ROOT / SOURCE) != SOURCE_SHA:
        raise RuntimeError("private cache/source binding changed")
    cache_before = hashes(cache)
    build_args = ["cargo", "test", "--release", "--features", context["feature"],
                  "--test", "gemm_bi_scalar_nt_d768_out_m64n64_tournament", "--no-run"]
    with (OUT / "build.log").open("x") as output:
        build = subprocess.run(build_args, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    matches = re.findall(r"Executable .* \((/root/[^)]+)\)", (OUT / "build.log").read_text())
    if build.returncode != 0 or len(matches) != 1:
        raise RuntimeError(f"focused build failed/ambiguous: exit={build.returncode} matches={matches}")
    binary = Path(matches[0])
    listing = subprocess.run([str(binary), "--list"], cwd=ROOT, env=env, text=True, capture_output=True)
    list_text = listing.stdout + listing.stderr
    (OUT / "test-list.log").write_text(list_text)
    if listing.returncode != 0 or f"{TEST}: test" not in list_text.splitlines():
        raise RuntimeError("exact NT test absent from authoritative list")

    run = OUT / "once7-cuda132"
    run.mkdir()
    pre = telemetry_runner.telemetry("PRE", context, run, True)
    args = [str(binary), TEST, "--ignored", "--exact", "--nocapture"]
    with (run / "test.log").open("x") as output:
        result = subprocess.run(args, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    release = telemetry_runner.telemetry("RELEASE", context, run, False)
    time.sleep(5)
    drain = telemetry_runner.telemetry("DRAIN", context, run, True)
    text = (run / "test.log").read_text()
    counts = {
        "resource": text.count('"schema":"MambaBiScalarNtAdaDiscoveryResourceV1"'),
        "auto_identity": text.count('"schema":"MambaBiScalarNtAdaAutoIdentityV1"'),
        "candidate_identity": text.count('"schema":"MambaBiScalarNtAdaCandidateIdentityV1"'),
        "bits": text.count('"schema":"MambaBiScalarNtAdaBitsV1"'),
        "screen": text.count('"schema":"MambaBiScalarNtAdaDiscoveryScreenV1"'),
        "decision": text.count('"schema":"MambaBiScalarNtAdaDiscoveryDecisionV1"'),
    }
    executed_one = ("test result: ok. 1 passed; 0 failed; 0 ignored;" in text
                    or "test result: FAILED. 0 passed; 1 failed; 0 ignored;" in text)
    schemas_complete = counts == {"resource": 3, "auto_identity": 1, "candidate_identity": 1,
                                  "bits": 8, "screen": 4, "decision": 1}
    valid_stop = result.returncode != 0 and schemas_complete and '"decision":"stop_no_retry"' in text
    valid_advance = result.returncode == 0 and schemas_complete and '"decision":"advance_to_full_qualification"' in text
    complete = executed_one and (valid_stop or valid_advance)
    receipt = {
        "schema": "MambaBiScalarNtAdaDiscoveryReceiptV1", "toolkit": "13.2",
        "feature": context["feature"], "generation": GENERATION,
        "source": SOURCE, "source_sha256": SOURCE_SHA,
        "build_args": build_args, "build_exit": build.returncode,
        "binary": str(binary), "binary_sha256": sha(binary), "test": TEST,
        "test_listed": True, "args": args, "exit": result.returncode,
        "executed_exactly_one_test": executed_one, "schema_counts": counts,
        "complete_valid_outcome": complete, "valid_stop": valid_stop, "valid_advance": valid_advance,
        "pre_utc": pre["utc"], "release_utc": release["utc"], "release_quiet": release["quiet"],
        "drain_utc": drain["utc"], "drain_quiet": drain["quiet"],
        "cache_before": cache_before, "cache_after": hashes(cache),
    }
    receipt["cache_stable"] = receipt["cache_before"] == receipt["cache_after"]
    write_json(OUT / "command.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if complete else 97


if __name__ == "__main__":
    sys.exit(main())
