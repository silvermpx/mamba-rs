#!/usr/bin/env python3
"""Build once and run the frozen NT128 and aligned F16 singleton screens."""

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
OUT = Path("/root/evidence-ada-triad-half-nt128-f16in-20260908")
ENV_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
TELEMETRY = Path("/root/evidence-ada-triad-nt-padded36-two-arm-20260907/run-arm.py")
SOURCES = {
    "tests/gemm_bi_typed_parity.rs":
        "0ccb32d6bde913ba6d55a511442271c14a1ae77448a93b1f6e29fb7c4353b45f",
    "tests/support/triad_half_tile_screen.rs":
        "759a1bbd4233d389c56b5489991ade0cc009173d3d8c6355d1f160a03f402859",
}
TESTS = [
    (
        "nt128",
        "cuda_suite::ada_half_nt_d768_out_loaded_tc128_vs_current_and_fast_discovery_once7",
        {
            "resource": ("MambaBiHalfNtLoadedTc128ResourceV1", 2),
            "bits": ("MambaBiHalfNtLoadedTc128BitsV1", 16),
            "screen": ("MambaBiHalfNtLoadedTc128ScreenV1", 16),
            "decision": ("MambaBiHalfNtLoadedTc128DecisionV1", 4),
        },
    ),
    (
        "f16-d768-in",
        "cuda_suite::ada_half_nn_fixed_s3_aligned_f16_d768_in_confirmation_once7",
        {
            "resource": ("MambaBiHalfNnTileAdaDiscoveryResourceV1", 2),
            "screen": ("MambaBiHalfNnS3AlignedScreenV1", 8),
            "decision": ("MambaBiHalfNnS3AlignedDecisionV1", 1),
        },
    ),
]


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


def run_one(label, test, schemas, binary, binary_sha, context, telemetry):
    run = OUT / f"once7-cuda132-{label}"
    run.mkdir()
    cache = Path(context["cache"])
    before = hashes(cache)
    pre = telemetry.telemetry("PRE", context, run, True)
    args = [str(binary), test, "--ignored", "--exact", "--nocapture"]
    with (run / "test.log").open("x") as output:
        result = subprocess.run(args, cwd=ROOT, env=dict(context["env"]),
                                stdout=output, stderr=subprocess.STDOUT, check=False)
    release = telemetry.telemetry("RELEASE", context, run, False)
    time.sleep(5)
    drain = telemetry.telemetry("DRAIN", context, run, True)
    text = (run / "test.log").read_text()
    counts = {name: text.count(f'"schema":"{schema}"')
              for name, (schema, _) in schemas.items()}
    expected = {name: count for name, (_, count) in schemas.items()}
    exact_one = "test result: ok. 1 passed; 0 failed; 0 ignored;" in text
    after = hashes(cache)
    complete = result.returncode == 0 and exact_one and counts == expected and before == after
    receipt = {
        "schema": "MambaBiHalfShortScreenRunReceiptV1", "label": label,
        "sources": SOURCES, "binary": str(binary), "binary_sha256": binary_sha,
        "test": test, "args": args, "exit": result.returncode,
        "executed_exactly_one_test": exact_one, "schema_counts": counts,
        "complete_success": complete, "pre_utc": pre["utc"],
        "release_utc": release["utc"], "release_quiet": release["quiet"],
        "drain_utc": drain["utc"], "drain_quiet": drain["quiet"],
        "test_log_sha256": sha(run / "test.log"), "cache_before": before,
        "cache_after": after, "cache_stable": before == after,
    }
    (run / "command.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return result.returncode if result.returncode != 0 or complete else 97


def main() -> int:
    OUT.mkdir(mode=0o700, parents=True, exist_ok=False)
    context = load("half_two_env", ENV_RUNNER).env_for(
        "13.2", "triad-nt-padded36-two-arm1"
    )
    telemetry = load("half_two_telemetry", TELEMETRY)
    cache = Path(context["cache"])
    if stat.S_IMODE(cache.stat().st_mode) != 0o700:
        raise RuntimeError("private cache mode changed")
    actual_sources = {name: sha(ROOT / name) for name in SOURCES}
    if actual_sources != SOURCES:
        raise RuntimeError(f"frozen source mismatch: {actual_sources!r}")
    build_args = ["cargo", "test", "--release", "--features", "cuda",
                  "--test", "gemm_bi_typed_parity", "--no-run"]
    with (OUT / "build.log").open("x") as output:
        build = subprocess.run(build_args, cwd=ROOT, env=dict(context["env"]),
                               stdout=output, stderr=subprocess.STDOUT, check=False)
    build_text = (OUT / "build.log").read_text()
    matches = re.findall(r"Executable .* \((/root/[^)]+)\)", build_text)
    if build.returncode != 0 or len(matches) != 1:
        raise RuntimeError(f"focused build failed/ambiguous: {build.returncode=} {matches=}")
    binary = Path(matches[0])
    binary_sha = sha(binary)
    listing = subprocess.run([str(binary), "--list"], cwd=ROOT,
                             env=dict(context["env"]), text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (OUT / "test-list.log").write_text(listing.stdout)
    listed = {line.removesuffix(": test") for line in listing.stdout.splitlines()
              if line.endswith(": test")}
    wanted = {test for _, test, _ in TESTS}
    if listing.returncode != 0 or not wanted.issubset(listed):
        raise RuntimeError(f"exact tests absent from authoritative list: {wanted - listed}")
    for label, test, schemas in TESTS:
        status = run_one(label, test, schemas, binary, binary_sha, context, telemetry)
        if status != 0:
            return status
    return 0


if __name__ == "__main__":
    sys.exit(main())
