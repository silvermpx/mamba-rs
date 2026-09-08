#!/usr/bin/env python3
"""Bind one fold-pipeline test to the validated direct-fold executor."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys


BASE = Path(
    "/root/evidence-ada-triad-f32-tn-d128-direct-20260908/"
    "more-waves/run-more-waves.py"
)
EVIDENCE = Path("/root/evidence-ada-triad-f32-tn-d128-fold-pipeline-20260908")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("label", choices=("d128-in", "d128-out"))
    parser.add_argument("test")
    args = parser.parse_args()
    build = json.loads((EVIDENCE / "build-receipt.json").read_text())
    spec = importlib.util.spec_from_file_location("validated_direct_runner", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE}")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.EVIDENCE = EVIDENCE
    runner.RUN = EVIDENCE / f"once7-cuda132-{args.label}"
    runner.BINARY = Path(build["binary"])
    runner.BINARY_SHA = build["binary_sha256"]
    runner.TEST = args.test
    return runner.main()


if __name__ == "__main__":
    sys.exit(main())
