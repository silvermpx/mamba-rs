#!/usr/bin/env python3
"""Bind the validated incremental builder to the three-cell TN dense batch."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


BASE = Path(
    "/root/evidence-ada-triad-nt-padded36-two-arm-20260907/"
    "build-repair-cuda132/run-build.py"
)
EVIDENCE = Path("/root/evidence-ada-triad-tn-dense-batch-20260908/build-cuda132")
PRIOR_MANIFEST = Path(
    "/root/evidence-ada-triad-tn-compact4-screen-20260908/source-manifest.json"
)
EXPECTED_HEAD = "0e1d0ce4bedd77914f99815ee21da0bbab85c513"
EXPECTED_RS = "2b3deae2581d5f117b43b520bac0d863cbf12336a50d0275a8f5182b4e38dd7d"
EXPECTED_HELPER = "30d7eefd26e1f4cac62217811e0e65c7bc649a39d5346228a814fab7ee833ba6"
EXPECTED_HEADER = "0a947ce30c5a51f110deacf1eeb1ff8e65cdee20c5f6d19708331e95a7de9b83"
EXPECTED_TESTS = {
    "cuda_suite::ada_tf32_tn_dense_s3_d768_in_discovery_once7",
    "cuda_suite::ada_tf32_tn_dense_s3_d768_out_discovery_once7",
    "cuda_suite::ada_tf32_tn_dense_s3_prism_discovery_once7",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


spec = importlib.util.spec_from_file_location("validated_build_runner", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load validated incremental build runner")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.EVIDENCE = EVIDENCE
runner.EXPECTED_HEAD = EXPECTED_HEAD
runner.EXPECTED_RS = EXPECTED_RS
runner.EXPECTED_TESTS = EXPECTED_TESTS
runner.EXPECTED_CACHE = {
    path.name: sha(path)
    for path in sorted(
        Path(
            "/root/mamba-kcache-ada-exact-toolkit-auto-"
            "triad-nt-padded36-two-arm1-cuda132-20260907"
        ).iterdir()
    )
    if path.is_file()
}

prior = json.loads(PRIOR_MANIFEST.read_text())
paths = set(prior["sources"])
paths.update(
    {
        "tests/support/triad_tn_dense_source.rs",
        "tests/gemm_bi_tf32_tn_dense_copy.cuh",
    }
)
EVIDENCE.mkdir(parents=True, exist_ok=True)
(EVIDENCE / "source-paths.txt").write_text("\n".join(sorted(paths)) + "\n")

root = runner.ROOT
if sha(root / "tests/support/triad_tn_dense_source.rs") != EXPECTED_HELPER:
    raise RuntimeError("dense source helper hash changed")
if sha(root / "tests/gemm_bi_tf32_tn_dense_copy.cuh") != EXPECTED_HEADER:
    raise RuntimeError("dense CUDA header hash changed")

original_write_json = runner.write_json


def write_json(path: Path, value: object) -> None:
    if isinstance(value, dict) and path.name == "source-manifest.json":
        value["schema"] = "MambaBiTf32TnDenseBatchBuildSourcesV1"
        value["excluded_local_wip"] = [
            "tests/gemm_bi_sm120_tf32_selector_qualification.rs",
            "tests/support/triad_discovery_samples.rs",
        ]
    if isinstance(value, dict) and path.name == "command.json":
        value["schema"] = "MambaBiTf32TnDenseBatchBuildReceiptV1"
        value["expected_tests"] = sorted(EXPECTED_TESTS)
        value["cache_reused_not_cold"] = True
    original_write_json(path, value)


runner.write_json = write_json
sys.exit(runner.main())
