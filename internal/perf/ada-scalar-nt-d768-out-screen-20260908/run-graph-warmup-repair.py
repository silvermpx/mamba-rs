#!/usr/bin/env python3
"""Run the graph-warmup-repaired Ada scalar-NT d768-out screen."""

import importlib.util
from pathlib import Path

BASE = Path(__file__).with_name("run.py")
spec = importlib.util.spec_from_file_location("scalar_nt_base", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {BASE}")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.OUT = Path("/root/evidence-ada-scalar-nt-d768-out-graph-warmup-repair-20260908")
runner.SOURCE_SHA = "a1f20ca1aed43010f5aae8740164a6a80b45d70e9b01e09d83fbb68ed811e1a7"

if __name__ == "__main__":
    raise SystemExit(runner.main())
