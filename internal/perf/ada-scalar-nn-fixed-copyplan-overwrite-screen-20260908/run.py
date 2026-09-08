#!/usr/bin/env python3
"""Run the reviewed overwrite-only FixedCopyPlan Ada scalar-NN screen."""

import importlib.util
from pathlib import Path

BASE = Path("/root/run-scalar-nn-base.py")
spec = importlib.util.spec_from_file_location("scalar_nn_base", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {BASE}")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.OUT = Path("/root/evidence-ada-scalar-nn-fixed-copyplan-overwrite-screen-20260908")
runner.SOURCE_SHA = "6729c227c1148f175158d3aeec77c29bc43092f632851e47fa66cf671f269ba5"
runner.TEST = "cuda_qualification::ada_exact_nn_fixed_copyplan_overwrite_three_cell_discovery_once7"

if __name__ == "__main__":
    raise SystemExit(runner.main())
