#!/usr/bin/env python3
"""Run the reviewed FixedCopyPlan-only Ada scalar-NN three-cell screen."""

import importlib.util
from pathlib import Path


BASE = Path(__file__).with_name("run.py")
spec = importlib.util.spec_from_file_location("scalar_nn_base", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {BASE}")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.OUT = Path("/root/evidence-ada-scalar-nn-fixed-copyplan-screen-20260908")
runner.SOURCE_SHA = "6473d6023b68e8b88bec9a9e3f07108d6b4bc65b79bc4671d6f6bd754f26fede"
runner.TEST = "cuda_qualification::ada_exact_nn_fixed_copyplan_three_cell_discovery_once7"

if __name__ == "__main__":
    raise SystemExit(runner.main())
