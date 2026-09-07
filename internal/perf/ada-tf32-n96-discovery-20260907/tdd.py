#!/usr/bin/env python3
"""Build the isolated N96 discovery test with the accepted CUDA13.2 env."""

import importlib.util
from pathlib import Path
import re
import sys


TASK8 = Path("/root/evidence-ada-exact-toolkit-auto-20260907")
EVIDENCE = Path("/root/evidence-ada-tf32-n96-discovery-20260907")


def main():
    if (
        len(sys.argv) not in (2, 3)
        or re.fullmatch(r"[a-z0-9-]+", sys.argv[1]) is None
        or (len(sys.argv) == 3 and sys.argv[2] != "run")
    ):
        raise SystemExit("usage: tdd.py LABEL [run]")
    spec = importlib.util.spec_from_file_location("task8_runner", TASK8 / "run.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load accepted Task8 runner")
    task8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task8)
    context = task8.env_for("13.2", "final2")
    EVIDENCE.mkdir(mode=0o700, parents=True, exist_ok=True)
    args = [
            "cargo",
            "test",
            "--release",
            "--features",
            context["feature"],
            "--test",
            "gemm_bi_fixed_tf32_n96_discovery",
    ]
    if len(sys.argv) == 2:
        args.append("--no-run")
    else:
        args.extend(("--", "--nocapture", "--test-threads=1"))
    code = task8.command(
        args,
        EVIDENCE / f"{sys.argv[1]}.log",
        context,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
