#!/usr/bin/env python3
"""Run and validate the frozen Task3 13.2 finalist/current/Fast once-21 batch."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import stat
import subprocess
import sys
import time


ENV_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
ROOT = Path("/root/mamba-ada-exact-toolkit-auto-20260907")
EVIDENCE = Path(
    "/root/evidence-ada-triad-finalist-integration-20260907/task3-once21-cuda132"
)
GENERATION = "triad-nt-padded36-two-arm1"
BINARY = Path(
    "/root/target-ada-exact-toolkit-auto-triad-nt-padded36-two-arm1-cuda132-20260907/"
    "release/deps/gemm_bi_performance_matrix-15e772a3a3d63b19"
)
BINARY_SHA = "02014b3d1a8771abc120fc49c8742e9d32b103c13ce84f123f5c162debd3676b"
SOURCE_MANIFEST_SHA = "08b5d723232e93a8dc8acc7a252aa7bad2d3c6cde4a3d25c946756e6c62be1b3"
TEST = "sm89_nt_finalist_once21::gemm_bi_sm89_nt_finalist_current_fast_once21"
CELLS = {"d768_in", "d768_out", "prism"}
EXPECTED_CACHE = {
    "mamba-kernels-v1-3809af034718aa08dfdfc1b6baeb407715436dc7db87ef39825b033c2fd72541.bin": "5851ebd49595a0b2ef665073289e52f903eec3266deadfc5ad5d9f2096ce6d16",
    "mamba-kernels-v1-822ebf0dda7dcf9c4f5c5a2213884dde42ab45f5cf238479a81856042c302b7c.bin": "5035921fc8e078e5454c9c5392f9e9ee6ca6a3de54c1e08cb54bba432fa37546",
    "mamba-kernels-v1-e4f76515e441adc28862775000fce218e2d2cca61b47ce484e15eb1144c40fb3.bin": "f0be052fc160b9e5f4b6a897878781327968c0015d551128415a94b85e6fc0aa",
    "mamba-kernels-v1-e61681f002f8ecdd61677fc89a8f0faeac5ce494db503fcc97fd344633ec2d65.bin": "1da42f2fbc44dfd9093959831296d86de2414a6d9bb042bc1640730c68d36ac5",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def cache_hashes(cache: Path) -> dict[str, str]:
    return {path.name: sha(path) for path in sorted(cache.iterdir()) if path.is_file()}


def telemetry(phase: str, environment: dict[str, str]) -> dict[str, object]:
    value: dict[str, object] = {
        "phase": phase,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment_constructor": str(ENV_RUNNER) + "::env_for",
    }
    for kind, query in (
        (
            "gpu",
            "--query-gpu=uuid,name,compute_cap,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.current.sm",
        ),
        ("apps", "--query-compute-apps=pid,gpu_uuid,process_name"),
    ):
        result = subprocess.run(
            ["/usr/bin/nvidia-smi", query, "--format=csv,noheader"],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        value[kind] = result.stdout
        value[kind + "_stderr"] = result.stderr
        value[kind + "_exit"] = result.returncode
    return value


def quiet(value: dict[str, object]) -> bool:
    fields = [field.strip() for field in str(value["gpu"]).strip().split(",")]
    return (
        value["gpu_exit"] == 0
        and value["apps_exit"] == 0
        and fields[:3]
        == [
            "GPU-d1edd7be-e88d-aed6-047d-622163306f0e",
            "NVIDIA RTX 6000 Ada Generation",
            "8.9",
        ]
        and fields[3:5] == ["0 %", "0 %"]
        and not str(value["apps"]).strip()
    )


def validate_records(records: list[dict[str, object]]) -> dict[str, object]:
    rows = [record for record in records if record.get("kind") == "sm89_nt_finalist_once21"]
    decisions = [
        record for record in records if record.get("kind") == "sm89_nt_finalist_cell_decision"
    ]
    bindings = [
        record for record in records if record.get("kind") == "sm89_nt_finalist_binding"
    ]
    completions = [
        record for record in records if record.get("kind") == "sm89_nt_finalist_once21_complete"
    ]
    expected_keys = {
        (cell, comparison, path, order)
        for cell in CELLS
        for comparison in ("current", "fast")
        for path in ("eager", "graph")
        for order in ("ab", "ba")
    }
    actual_keys = {
        (record["cell"], record["comparison"], record["path"], record["order"])
        for record in rows
    }
    if len(rows) != 24 or actual_keys != expected_keys:
        raise RuntimeError(f"once21 row closure changed: rows={len(rows)} keys={actual_keys}")
    for record in rows:
        if record.get("windows") != 21:
            raise RuntimeError("once21 row does not have 21 windows")
        candidate = record.get("candidate_us")
        denominator = record.get("denominator_us")
        ratios = record.get("ratios")
        if not all(isinstance(values, list) and len(values) == 21 for values in (candidate, denominator, ratios)):
            raise RuntimeError("once21 sample-array closure changed")
        assert isinstance(candidate, list) and isinstance(denominator, list) and isinstance(ratios, list)
        for index, (candidate_us, denominator_us, ratio) in enumerate(zip(candidate, denominator, ratios)):
            if not all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0 for value in (candidate_us, denominator_us, ratio)):
                raise RuntimeError(f"invalid sample at {record['cell']}/{index}")
            expected = candidate_us / denominator_us
            if not math.isclose(ratio, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError(f"ratio arithmetic changed at {record['cell']}/{index}")
        if record.get("candidate_guards") != 3 or record.get("current_guards") != 3:
            raise RuntimeError("physical guard census changed")
        if record.get("fast_guard_inventory") != "three_exact_sized_buffers_no_redzones":
            raise RuntimeError("Fast guard inventory changed")
        candidate_physical = record.get("candidate_physical")
        current_physical = record.get("current_physical")
        candidate_route = record.get("candidate_route_identity")
        current_route = record.get("current_route_identity")
        if not isinstance(candidate_physical, dict) or candidate_physical.get("physical_launch_count") != 1:
            raise RuntimeError("candidate physical inventory changed")
        if candidate_physical.get("physical_symbol") != "gemm_bi_nt_sm89_mma_tf32_compact8_v1_m128n64_bk32_s2":
            raise RuntimeError("candidate symbol changed")
        if not isinstance(current_physical, dict) or not current_physical.get("physical_launch_count"):
            raise RuntimeError("current physical inventory is empty")
        if not isinstance(candidate_route, dict) or candidate_route.get("tuning_table_revision") != 45:
            raise RuntimeError("candidate shared route identity changed")
        if not isinstance(current_route, dict) or current_route.get("tuning_table_revision") != 45:
            raise RuntimeError("current shared route identity changed")
        if record["comparison"] == "fast" and not record.get("vendor_modes"):
            raise RuntimeError("Fast row omitted native modes")
        if record["comparison"] == "current" and record.get("vendor_modes") is not None:
            raise RuntimeError("current row incorrectly claims vendor modes")
    if len(decisions) != 3 or {record.get("cell") for record in decisions} != CELLS:
        raise RuntimeError("per-cell decision closure changed")
    if any(record.get("rows") != 8 for record in decisions):
        raise RuntimeError("per-cell row count changed")
    if len(bindings) != 3 or {record.get("cell") for record in bindings} != CELLS:
        raise RuntimeError("finalist binding closure changed")
    if len(completions) != 1 or completions[0].get("rows") != 24:
        raise RuntimeError("once21 completion closure changed")
    return {
        "rows": len(rows),
        "decisions": {record["cell"]: record["admit_against_current"] for record in decisions},
        "bindings": len(bindings),
        "completion": completions[0],
    }


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("env_runner", ENV_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load environment constructor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = module.env_for("13.2", GENERATION)
    environment = context["env"]
    cache = Path(context["cache"])
    if sha(BINARY) != BINARY_SHA:
        raise RuntimeError("once21 binary changed")
    if stat.S_IMODE(cache.stat().st_mode) != 0o700 or cache_hashes(cache) != EXPECTED_CACHE:
        raise RuntimeError("once21 cache binding changed")
    listing = subprocess.run(
        [str(BINARY), "--list"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    (EVIDENCE / "test-list.log").write_text(listing.stdout + listing.stderr)
    names = {
        line.removesuffix(": test")
        for line in listing.stdout.splitlines()
        if line.endswith(": test")
    }
    if listing.returncode != 0 or TEST not in names:
        raise RuntimeError("exact once21 test is not listed")
    pre = telemetry("PRE", environment)
    write_json(EVIDENCE / "pre.json", pre)
    if not quiet(pre):
        raise RuntimeError(f"once21 PRE is not strict quiet/no-apps: {pre}")
    args = [str(BINARY), TEST, "--exact", "--ignored", "--nocapture"]
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with (EVIDENCE / "test.log").open("x") as output:
        process = subprocess.Popen(
            args,
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        for line in process.stdout:
            output.write(line)
            output.flush()
            print(line, end="", flush=True)
        exit_code = process.wait()
    release = telemetry("RELEASE", environment)
    write_json(EVIDENCE / "release.json", release)
    drain_samples = []
    consecutive = 0
    for _ in range(120):
        sample = telemetry("DRAIN", environment)
        drain_samples.append(sample)
        consecutive = consecutive + 1 if quiet(sample) else 0
        if consecutive >= 5:
            break
        time.sleep(1)
    write_json(
        EVIDENCE / "drain.json",
        {
            "schema": "MambaTriadAdaFinalistDrainV1",
            "required_consecutive_quiet": 5,
            "samples": drain_samples,
            "complete": consecutive >= 5,
        },
    )
    output = (EVIDENCE / "test.log").read_text()
    executed_one = "1 passed" in output and "0 passed" not in output
    records = [
        json.loads(line)
        for line in output.splitlines()
        if line.startswith("{") and line.endswith("}")
    ]
    analysis = validate_records(records) if exit_code == 0 and executed_one else None
    receipt = {
        "schema": "MambaTriadAdaFinalistTask3Once21ReceiptV1",
        "toolkit": context["toolkit"],
        "feature": context["feature"],
        "generation": GENERATION,
        "binary": str(BINARY),
        "binary_sha256": BINARY_SHA,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA,
        "cache": str(cache),
        "cache_mode": oct(stat.S_IMODE(cache.stat().st_mode)),
        "cache_reused_not_cold": True,
        "cache_artifacts": cache_hashes(cache),
        "args": args,
        "started_utc": started,
        "exit": exit_code,
        "executed_one": executed_one,
        "analysis": analysis,
        "pre_utc": pre["utc"],
        "release_utc": release["utc"],
        "release_was_quiet": quiet(release),
        "drain_sample_count": len(drain_samples),
        "drain_complete": consecutive >= 5,
    }
    write_json(EVIDENCE / "command.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    print("WRAPPER_COMPLETE")
    ok = (
        exit_code == 0
        and executed_one
        and analysis is not None
        and cache_hashes(cache) == EXPECTED_CACHE
        and consecutive >= 5
    )
    return 0 if ok else 97


if __name__ == "__main__":
    sys.exit(main())
