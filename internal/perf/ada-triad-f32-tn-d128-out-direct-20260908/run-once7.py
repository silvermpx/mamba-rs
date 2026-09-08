#!/usr/bin/env python3
"""Bind the validated direct-fold runner to the d128-out two-arm screen."""

import importlib.util
from pathlib import Path
import sys


BASE = Path(
    "/root/evidence-ada-triad-f32-tn-d128-direct-20260908/"
    "more-waves/run-more-waves.py"
)
EVIDENCE = Path("/root/evidence-ada-triad-f32-tn-d128-out-direct-20260908")

spec = importlib.util.spec_from_file_location("validated_direct_runner", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {BASE}")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.EVIDENCE = EVIDENCE
runner.RUN = EVIDENCE / "once7-cuda132"
runner.BINARY_SHA = "98df620c423deb12ca149e71a3a0c9123ce8f119952956e2b9053f37e6d56286"
runner.TEST = "cuda_tournament::ada_d128_out_direct_fold_two_arm_once7"
sys.exit(runner.main())
