#!/usr/bin/env python3
"""Run the two authoritative bare-name tests from the sealed 0ccb build."""

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
BUILD = Path("/root/evidence-ada-triad-half-nt128-f16in-20260908")
OUT = Path("/root/evidence-ada-triad-half-nt128-f16in-20260908-attempt2")
ENV_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
TELEMETRY = Path("/root/evidence-ada-triad-nt-padded36-two-arm-20260907/run-arm.py")
TESTS = [
    ("nt128", "ada_half_nt_d768_out_loaded_tc128_vs_current_and_fast_discovery_once7",
     {"resource": ("MambaBiHalfNtLoadedTc128ResourceV1", 2),
      "bits": ("MambaBiHalfNtLoadedTc128BitsV1", 16),
      "screen": ("MambaBiHalfNtLoadedTc128ScreenV1", 16),
      "decision": ("MambaBiHalfNtLoadedTc128DecisionV1", 4)}),
    ("f16-d768-in", "ada_half_nn_fixed_s3_aligned_f16_d768_in_confirmation_once7",
     {"resource": ("MambaBiHalfNnTileAdaDiscoveryResourceV1", 2),
      "screen": ("MambaBiHalfNnS3AlignedScreenV1", 8),
      "decision": ("MambaBiHalfNnS3AlignedDecisionV1", 1)}),
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


def main() -> int:
    OUT.mkdir(mode=0o700, parents=True, exist_ok=False)
    matches = re.findall(r"Executable .* \((/root/[^)]+)\)",
                         (BUILD / "build.log").read_text())
    if len(matches) != 1:
        raise RuntimeError(f"sealed build binary ambiguous: {matches!r}")
    binary = Path(matches[0])
    binary_sha = sha(binary)
    listed = {line.removesuffix(": test")
              for line in (BUILD / "test-list.log").read_text().splitlines()
              if line.endswith(": test")}
    wanted = {test for _, test, _ in TESTS}
    if not wanted.issubset(listed):
        raise RuntimeError(f"authoritative bare tests absent: {wanted - listed}")
    context = load("half_two_env", ENV_RUNNER).env_for(
        "13.2", "triad-nt-padded36-two-arm1"
    )
    telemetry = load("half_two_telemetry", TELEMETRY)
    cache = Path(context["cache"])
    if stat.S_IMODE(cache.stat().st_mode) != 0o700:
        raise RuntimeError("private cache mode changed")
    for label, test, schemas in TESTS:
        run = OUT / f"once7-cuda132-{label}"
        run.mkdir()
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
            "source_sha256": "0ccb32d6bde913ba6d55a511442271c14a1ae77448a93b1f6e29fb7c4353b45f",
            "binary": str(binary), "binary_sha256": binary_sha,
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
        if result.returncode != 0 or not complete:
            return result.returncode if result.returncode != 0 else 97
    return 0


if __name__ == "__main__":
    sys.exit(main())
