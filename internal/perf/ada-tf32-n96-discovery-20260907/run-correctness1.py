#!/usr/bin/env python3
"""Run the bounded CUDA13.2 N96 discovery with strict lane telemetry."""

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


ROOT = Path("/root/mamba-ada-exact-toolkit-auto-20260907")
TASK8 = Path("/root/evidence-ada-exact-toolkit-auto-20260907")
EVIDENCE = Path("/root/evidence-ada-tf32-n96-discovery-20260907")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task8_runner():
    spec = importlib.util.spec_from_file_location("task8_runner", TASK8 / "run.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load accepted Task8 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def context_for(operation):
    task8 = task8_runner()
    context = task8.env_for("13.2", "final2")
    context = {**context, "env": context["env"].copy()}
    context["env"]["MAMBA_FIXED_N96_DISCOVERY"] = "1"
    if operation == "timing7":
        context["env"]["MAMBA_FIXED_N96_WINDOWS"] = "7"
    return task8, context


def main():
    if (
        len(sys.argv) != 3
        or sys.argv[1] not in ("correctness", "timing7")
        or re.fullmatch(r"[a-z0-9-]+", sys.argv[2]) is None
    ):
        raise SystemExit("usage: run.py correctness|timing7 ATTEMPT")
    operation, attempt = sys.argv[1:]
    task8, context = context_for(operation)
    directory = EVIDENCE / f"cuda132-{attempt}"
    directory.mkdir(mode=0o700, parents=True)
    binary = task8.find_binary(context, "gemm_bi_fixed_tf32_n96_discovery")
    source_paths = [
        ROOT / "tests/gemm_bi_fixed_tf32_n96_discovery.rs",
        ROOT / "tests/gemm_bi_fixed_tf32_n96_discovery.cu",
    ]
    binding = {
        "schema": "MambaBiFixedTf32N96DiscoveryBindingV1",
        "operation": operation,
        "toolkit": "13.2",
        "feature": context["feature"],
        "binary": str(binary),
        "binary_sha256": sha(binary),
        "sources": {str(path.relative_to(ROOT)): sha(path) for path in source_paths},
        "tools": {
            str(Path(__file__)): sha(__file__),
            str(TASK8 / "run.py"): sha(TASK8 / "run.py"),
        },
    }
    (directory / "binding.json").write_text(json.dumps(binding, indent=2) + "\n")
    test = (
        "n96_small_bits_graph_and_resources"
        if operation == "correctness"
        else "n96_e0_short_paired_discovery"
    )
    task8.telemetry("PRE", directory, context)
    code = task8.command(
        [str(binary), "--ignored", "--exact", test, "--nocapture", "--test-threads=1"],
        directory / "test.log",
        context,
    )
    task8.telemetry("POST", directory, context)
    task8.telemetry("RELEASE", directory, context)
    (directory / "result.json").write_text(
        json.dumps({"operation": operation, "attempt": attempt, "exit": code}, indent=2)
        + "\n"
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
