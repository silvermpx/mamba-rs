#!/usr/bin/env python3
"""Parameterize the accepted short F32-TN runner for M16N16/128 threads."""

import importlib.util
from pathlib import Path
import sys


BASE = Path(
    "/root/evidence-ada-triad-f32-tn-d128-direct-20260908/"
    "more-waves/run-more-waves.py"
)


def main() -> int:
    spec = importlib.util.spec_from_file_location("accepted_more_waves_runner", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import accepted runner {BASE}")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.EVIDENCE = Path(
        "/root/evidence-ada-triad-f32-tn-d128-direct-20260908/more-warps"
    )
    runner.RUN = runner.EVIDENCE / "once7-cuda132"
    runner.BINARY_SHA = (
        "2c4eb49df61c80637cc18cb84498cc8c39651f76a9b44e07a57f3a5290b1c989"
    )
    runner.TEST = "cuda_tournament::ada_d128_direct_fold_more_warps_once7"
    return runner.main()


if __name__ == "__main__":
    sys.exit(main())
