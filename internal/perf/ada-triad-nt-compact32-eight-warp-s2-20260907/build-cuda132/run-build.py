#!/usr/bin/env python3
"""Bind the validated incremental build/list runner to compact32 S2 discovery."""

import importlib.util
from pathlib import Path
import sys

BASE = Path(
    "/root/evidence-ada-triad-nt-padded36-two-arm-20260907/"
    "build-repair-cuda132/run-build.py"
)
EVIDENCE = Path(
    "/root/evidence-ada-triad-nt-compact32-eight-warp-s2-20260907/build-cuda132"
)
TEST = "cuda_suite::ada_tf32_nt_compact_eight_warp_s2_discovery_once7"

if len(sys.argv) != 3:
    raise RuntimeError("usage: run-build.py HEAD RUST_SOURCE_SHA256")

spec = importlib.util.spec_from_file_location("validated_build_runner", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load validated incremental build runner")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.EVIDENCE = EVIDENCE
runner.EXPECTED_HEAD = sys.argv[1]
runner.EXPECTED_RS = sys.argv[2]
runner.EXPECTED_TESTS = {TEST}

original_write_json = runner.write_json


def write_json(path: Path, value: object) -> None:
    if isinstance(value, dict) and path.name == "source-manifest.json":
        value["schema"] = "MambaTriadNtCompact32EightWarpS2BuildSourcesV1"
        value["supersedes_source_manifest_sha256"] = (
            "860fb1ee864b847d7f1d7ddb641aed66be2fc7c229f24911bb14a4e93c9e948b"
        )
    if isinstance(value, dict) and path.name == "command.json":
        value["schema"] = "MambaTriadNtCompact32EightWarpS2BuildReceiptV1"
        value["expected_test"] = TEST
    original_write_json(path, value)


runner.write_json = write_json
sys.exit(runner.main())
