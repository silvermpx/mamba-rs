#!/usr/bin/env python3
"""Run the target-gate-repaired Ada scalar-NT d768-out screen."""

import importlib.util
from pathlib import Path

BASE = Path(__file__).with_name("run.py")
spec = importlib.util.spec_from_file_location("scalar_nt_base", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {BASE}")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.OUT = Path("/root/evidence-ada-scalar-nt-d768-out-target-repair-20260908")
runner.SOURCE_SHA = "36b2c804be31aea16feb585dfa8ced01751bfa417014b12a2063632224697047"

if __name__ == "__main__":
    raise SystemExit(runner.main())
