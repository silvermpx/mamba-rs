#!/usr/bin/env python3
"""Bind the validated single-arm runner to dense-copy discovery."""

import importlib.util
from pathlib import Path
import sys

BASE = Path("/root/evidence-ada-triad-nt-padded36-two-arm-20260907/run-arm.py")
EVIDENCE = Path("/root/evidence-ada-triad-nt-padded36-dense-copy-20260907")
TEST = "cuda_suite::ada_tf32_nt_padded_dense_copy_discovery_once7"

if len(sys.argv) != 3:
    raise RuntimeError("usage: run-once7.py BINARY_SHA256 SOURCE_MANIFEST_SHA256")

spec = importlib.util.spec_from_file_location("validated_arm_runner", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load validated single-arm runner")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.EVIDENCE = EVIDENCE
runner.BUILD = EVIDENCE / "build-cuda132"
runner.EXPECTED_BINARY_SHA = sys.argv[1]
runner.EXPECTED_SOURCE_MANIFEST_SHA = sys.argv[2]
runner.ARMS = {"dense-copy": ("padded_dense_copy", TEST)}
sys.argv = [sys.argv[0], "dense-copy"]
sys.exit(runner.main())
