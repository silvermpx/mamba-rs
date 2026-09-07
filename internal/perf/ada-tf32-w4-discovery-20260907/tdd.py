#!/usr/bin/env python3
"""Build the isolated four-warp TF32 discovery test with CUDA 13.2."""

import importlib.util
from pathlib import Path
import re
import sys


TASK8 = Path("/root/evidence-ada-exact-toolkit-auto-20260907")
EVIDENCE = Path("/root/evidence-ada-tf32-w4-discovery-20260907")


def main():
    if len(sys.argv) != 2 or re.fullmatch(r"[a-z0-9-]+", sys.argv[1]) is None:
        raise SystemExit("usage: tdd.py LABEL")
    spec = importlib.util.spec_from_file_location("task8_runner", TASK8 / "run.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load accepted Task8 runner")
    task8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task8)
    context = task8.env_for("13.2", "final2")
    EVIDENCE.mkdir(mode=0o700, parents=True, exist_ok=True)
    code = task8.command(
        [
            "cargo",
            "test",
            "--release",
            "--features",
            context["feature"],
            "--test",
            "gemm_bi_fixed_tf32_w4_discovery",
            "--",
            "--nocapture",
            "--test-threads=1",
        ],
        EVIDENCE / f"{sys.argv[1]}.log",
        context,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
