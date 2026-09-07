#!/usr/bin/env python3
"""Bind the validated single-arm runner to one sibling discovery arm."""

import importlib.util
from pathlib import Path
import sys

BASE = Path("/root/evidence-ada-triad-nt-padded36-two-arm-20260907/run-arm.py")
EVIDENCE = Path("/root/evidence-ada-triad-nt-sibling-two-mechanism-20260907")
ARMS = {
    "padded-dense-d768-out": (
        "padded_dense_d768_out",
        "cuda_suite::ada_tf32_nt_padded_dense_d768_out_discovery_once7",
    ),
    "padded-dense-prism": (
        "padded_dense_prism",
        "cuda_suite::ada_tf32_nt_padded_dense_prism_discovery_once7",
    ),
    "compact-eight-warp-s2-d768-out": (
        "compact_eight_warp_s2_d768_out",
        "cuda_suite::ada_tf32_nt_compact_eight_warp_s2_d768_out_discovery_once7",
    ),
    "compact-eight-warp-s2-prism": (
        "compact_eight_warp_s2_prism",
        "cuda_suite::ada_tf32_nt_compact_eight_warp_s2_prism_discovery_once7",
    ),
}

if len(sys.argv) != 4 or sys.argv[1] not in ARMS:
    raise RuntimeError(
        "usage: run-once7.py ARM BINARY_SHA256 SOURCE_MANIFEST_SHA256"
    )

arm = sys.argv[1]
spec = importlib.util.spec_from_file_location("validated_arm_runner", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load validated single-arm runner")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.EVIDENCE = EVIDENCE
runner.BUILD = EVIDENCE / "build-cuda132"
runner.EXPECTED_BINARY_SHA = sys.argv[2]
runner.EXPECTED_SOURCE_MANIFEST_SHA = sys.argv[3]
runner.ARMS = ARMS
sys.argv = [sys.argv[0], arm]
sys.exit(runner.main())
