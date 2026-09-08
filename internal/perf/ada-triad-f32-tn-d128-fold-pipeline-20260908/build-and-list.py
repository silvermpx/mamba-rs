#!/usr/bin/env python3
"""Build and list the frozen Ada F32 TN fold-pipeline screen."""

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path("/root/mamba-ada-exact-toolkit-auto-20260907")
EVIDENCE = Path("/root/evidence-ada-triad-f32-tn-d128-fold-pipeline-20260908")
ENV_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
EXPECTED = {
    "tests/gemm_bi_tn_d128_fused_contract_tournament.rs":
        "bea57f42f30b947ed4dde4c7ebf13faf6ed7da9331c9e082f696c574b8008bd1",
    "tests/support/triad_tn_d128_direct_source.rs":
        "1243b969c16747388a07c46a2fe491a076421bb97b6a907b8c2aa0db6ea5f287",
}
TESTS = {
    "cuda_tournament::ada_d128_in_fold_pipeline_once7",
    "cuda_tournament::ada_d128_out_fold_pipeline_once7",
}


def load(path: Path):
    spec = importlib.util.spec_from_file_location("fold_pipeline_env", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=False)
    actual = {name: sha(ROOT / name) for name in EXPECTED}
    if actual != EXPECTED:
        raise RuntimeError(f"frozen source mismatch: {actual!r}")
    context = load(ENV_RUNNER).env_for("13.2", "triad-nt-padded36-two-arm1")
    env = dict(context["env"])
    args = [
        "cargo", "test", "--release", "--features", "cuda", "--test",
        "gemm_bi_tn_d128_fused_contract_tournament", "--no-run",
    ]
    started = time.monotonic()
    result = subprocess.run(
        args, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    (EVIDENCE / "build.log").write_text(result.stdout)
    (EVIDENCE / "build.exit").write_text(f"{result.returncode}\n")
    (EVIDENCE / "build.elapsed_seconds").write_text(
        f"{time.monotonic() - started:.6f}\n"
    )
    if result.returncode != 0:
        print(result.stdout)
        return result.returncode
    matches = re.findall(r"Executable .* \(([^)]+)\)", result.stdout)
    if len(matches) != 1:
        raise RuntimeError(f"expected one executable, got {matches!r}")
    binary = Path(matches[0])
    if not binary.is_absolute():
        binary = ROOT / binary
    listed = subprocess.run(
        [str(binary), "--list"], cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    (EVIDENCE / "test-list.log").write_text(listed.stdout)
    if listed.returncode != 0:
        return listed.returncode
    actual_tests = {
        line.removesuffix(": test")
        for line in listed.stdout.splitlines()
        if line.endswith(": test") and "fold_pipeline_once7" in line
    }
    if actual_tests != TESTS:
        raise RuntimeError(f"authoritative list mismatch: {actual_tests!r}")
    receipt = {
        "schema": "AdaTnFoldPipelineBuildReceiptV1",
        "args": args,
        "source_sha256": actual,
        "binary": str(binary),
        "binary_sha256": sha(binary),
        "tests": sorted(actual_tests),
        "environment_constructor": str(ENV_RUNNER) + "::env_for",
    }
    (EVIDENCE / "build-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
