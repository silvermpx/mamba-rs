#!/usr/bin/env python3
"""Bind the validated padded36 arm runner to the reviewed eight-warp arm."""

import importlib.util
from pathlib import Path
import sys

BASE = Path("/root/evidence-ada-triad-nt-padded36-two-arm-20260907/run-arm.py")
EVIDENCE = Path("/root/evidence-ada-triad-nt-padded36-eight-warp-20260907")

spec = importlib.util.spec_from_file_location("validated_arm_runner", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load validated padded36 arm runner")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.EVIDENCE = EVIDENCE
runner.BUILD = EVIDENCE / "build-cuda132"
runner.EXPECTED_BINARY_SHA = (
    "11b99112c4de62e0e5644fb7711f61a944343286f1c4890a54cc4693d24b4869"
)
runner.EXPECTED_SOURCE_MANIFEST_SHA = (
    "94a6e459d47b1bced440b76607af444e7e7e6b9c6d0da27765a44c2d7aedd758"
)
runner.ARMS = {
    "eight-warp": (
        "padded_eight_warp",
        "cuda_suite::ada_tf32_nt_padded_eight_warp_discovery_once7",
    )
}
sys.argv = [sys.argv[0], "eight-warp"]
sys.exit(runner.main())
