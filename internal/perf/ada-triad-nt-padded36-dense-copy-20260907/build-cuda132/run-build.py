#!/usr/bin/env python3
"""Bind the validated incremental build/list runner to dense-copy discovery."""

import importlib.util
from pathlib import Path
import sys

BASE = Path(
    "/root/evidence-ada-triad-nt-padded36-two-arm-20260907/"
    "build-repair-cuda132/run-build.py"
)
EVIDENCE = Path(
    "/root/evidence-ada-triad-nt-padded36-dense-copy-20260907/build-cuda132"
)
TEST = "cuda_suite::ada_tf32_nt_padded_dense_copy_discovery_once7"

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
        value["schema"] = "MambaTriadNtPadded36DenseCopyBuildSourcesV1"
        value["supersedes_source_manifest_sha256"] = (
            "94a6e459d47b1bced440b76607af444e7e7e6b9c6d0da27765a44c2d7aedd758"
        )
    if isinstance(value, dict) and path.name == "command.json":
        value["schema"] = "MambaTriadNtPadded36DenseCopyBuildReceiptV1"
        value["expected_test"] = TEST
    original_write_json(path, value)


runner.write_json = write_json
sys.exit(runner.main())
