#!/usr/bin/env python3
"""Independent Task8 post-AUTO44 validator using the frozen Task7 arithmetic gate."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


SCHEMA = "MambaBiFixedAdaExactPostAutoV1"
FAMILY = "f32_exact_post_auto"
EXACT = [f"hot_{cell}:{bias}" for cell in ("a", "b", "d", "e") for bias in (0, 1)]
ARMS = ["Legacy", "AUTO", "Fast"]
DIRECTIONS = ["AUTO/Legacy", "Legacy/Fast", "AUTO/Fast"]
PROMOTION_BASIS = {
    "task7_source_sha": "97f3d43fc315c96238f41f9f39bc15418518b5493e216fb9bc0a8f03a46e5bc7",
    "cuda128_screen_sha": "d43c360d844065a3691f933b0b743a18e8482d2d36de787ef0ed2392e30e53bf",
    "cuda128_confirm_sha": "eb3b0abee0e336c7c93f9c5eddec698dae8a3fc1d8363ca337047b835758ad96",
    "cuda130_screen_sha": "cae9db1864bd64700ee3033196eae4c6f8a0abd7a3a1cf09a407889cb7727682",
    "cuda130_confirm_sha": "0b27351512f3265b156a29aaf7fadcaf4a84870cac06e1c1b36ce3e358f20711",
    "task7_final_review_sha": "750e0d02b524229c7a987894eee214af5e33e57f75499779b7263352b33697ac",
    "task7_selected_manifest_sha": "822560b7978f641543418e5971033b872dd83198157d8456d20317998c4c58d7",
}


def _task7_path():
    local = Path(__file__).resolve().parent.parent / "ada-f32-tf32-toolkit-20260907" / "analyze.py"
    remote = Path("/root/evidence-ada-f32-tf32-toolkit-20260907/analyze.py")
    return local if local.exists() else remote


_SPEC = importlib.util.spec_from_file_location("frozen_task7_analyze", _task7_path())
task7 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(task7)

require = task7.require
sha = task7.sha
quantiles = task7.quantiles
admitted = task7.admitted


def read_jsonl(path):
    data = Path(path).read_bytes()
    lines = data.splitlines(keepends=True)
    require(lines and all(line.endswith(b"\n") for line in lines), "JSONL must be newline terminated")
    records = []
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except Exception as error:
            raise ValueError(f"invalid JSON line {index}: {error}") from error
        require(record.get("schema") == SCHEMA, f"wrong Task8 schema line {index}")
        records.append(record)
    complete = records[-1]
    require(complete.get("kind") == "complete", "completion must be last")
    require(complete.get("preceding_lines") == len(records) - 1, "preceding line count differs")
    require(
        complete.get("preceding_jsonl_sha256") == hashlib.sha256(b"".join(lines[:-1])).hexdigest(),
        "preceding JSONL digest differs",
    )
    return records


def _adapt_for_task7(records):
    adapted = copy.deepcopy(records)
    arm_map = {"Legacy": "actualAUTO", "AUTO": "candidate", "Fast": "Fast"}
    direction_map = {
        "AUTO/Legacy": "candidate/AUTO",
        "Legacy/Fast": "AUTO/Fast",
        "AUTO/Fast": "candidate/Fast",
    }
    stage = records[0]["stage"]
    task7_stage = "smoke1" if stage == "smoke1" else "confirm101"
    for record in adapted:
        record["schema"] = task7.SCHEMA
        if "family" in record:
            record["family"] = "f32_exact_fast"
        if "stage" in record:
            record["stage"] = task7_stage
        if record.get("kind") == "identity":
            record["tuning_revision"] = 43
            if task7_stage == "confirm101":
                record["screen_sha"] = "0" * 64
                record["screen_artifact_sha"] = record["fixed_artifact_digest"]
        elif record.get("kind") == "physical":
            graphs = record["graphs"]
            record["actual_auto"] = "Legacy"
            record["candidate"] = "F32Sm89N64CopyPlan"
            record["graphs"] = {
                "actualAUTO": graphs["Legacy"],
                "candidate": graphs["AUTO"],
                "Fast": graphs["Fast"],
            }
            record.pop("former_incumbent", None)
            record.pop("public_auto_enum_verified", None)
        elif record.get("kind") == "sample":
            record["arm"] = arm_map[record["arm"]]
        elif record.get("kind") in ("pair", "summary"):
            record["direction"] = direction_map[record["direction"]]
        elif record.get("kind") == "literal_decision":
            record["own_rule"] = (
                "candidate/AUTO p50 and p95 < 1 in all four path/start strata"
            )
    return adapted


def validate_records(records):
    require(records, "empty Task8 record set")
    require(all(record.get("schema") == SCHEMA for record in records), "foreign Task8 schema")
    identities = [record for record in records if record.get("kind") == "identity"]
    require(len(identities) == 1 and records[0] is identities[0], "identity must be unique and first")
    identity = identities[0]
    stage = identity.get("stage")
    windows = {"smoke1": 1, "post101": 101}.get(stage)
    require(identity.get("family") == FAMILY, "Task8 family must be exact post-AUTO")
    require(identity.get("windows") == windows, "Task8 stage/window mismatch")
    require(identity.get("toolkit") in ("12.8", "13.0"), "Task8 toolkit differs")
    require(identity.get("tuning_revision") == 44, "Task8 tuning revision differs")
    require(identity.get("promotion_basis") == PROMOTION_BASIS, "Task8 promotion basis differs")
    require(identity.get("screen_sha") is None, "Task8 post stage must not bind a Task7 screen")
    require(identity.get("screen_artifact_sha") is None, "Task8 post stage must not bind screen artifact")
    require(identity.get("literal_control") == ",".join(EXACT), "Task8 literal inventory/order differs")
    for record in records[1:-1]:
        require(record.get("family") == FAMILY and record.get("stage") == stage, "foreign Task8 family/stage")
    for record in (record for record in records if record.get("kind") == "physical"):
        require(record.get("former_incumbent") == "Legacy", "wrong former incumbent")
        require(record.get("actual_auto") == "F32Sm89N64CopyPlan", "wrong Task8 AUTO enum")
        require(record.get("public_auto_enum_verified") is True, "public AUTO enum was not verified")
        require(set(record.get("graphs", {})) == set(ARMS), "wrong Task8 physical arms")
    for record in (record for record in records if record.get("kind") == "sample"):
        require(record.get("arm") in ARMS, "wrong Task8 sample arm")
    for record in (
        record for record in records if record.get("kind") in ("pair", "summary")
    ):
        comparison = record.get("comparison")
        require(comparison in range(3), "wrong Task8 comparison")
        require(record.get("direction") == DIRECTIONS[comparison], "wrong Task8 direction")
    for record in (record for record in records if record.get("kind") == "literal_decision"):
        require(
            record.get("own_rule")
            == "AUTO/Legacy p50 and p95 < 1 in all four path/start strata",
            "wrong Task8 owner rule",
        )
    shared = task7.validate_records(_adapt_for_task7(records))
    shared.update(
        family=FAMILY,
        stage=stage,
        toolkit=identity["toolkit"],
        windows=windows,
        literals=EXACT.copy(),
        identity=identity,
    )
    return shared


def validate_binding(analysis, result, binding):
    identity = analysis["identity"]
    binaries = binding.get("binaries")
    require(isinstance(binaries, dict), "binding binaries map missing")
    performance = [
        digest
        for path, digest in binaries.items()
        if isinstance(path, str)
        and re.fullmatch(r"gemm_bi_fixed_performance-[0-9a-f]+", Path(path).name)
    ]
    require(len(performance) == 1, "performance binary binding missing or ambiguous")
    performance_sha = performance[0]
    require(
        isinstance(performance_sha, str)
        and re.fullmatch(r"[0-9a-f]{64}", performance_sha),
        "performance binary digest malformed",
    )
    require(
        identity["source_sha"] == result.get("source_sha") == binding.get("measured_source_sha"),
        "source binding differs",
    )
    require(
        identity["binary_sha"] == result.get("binary_sha") == performance_sha,
        "binary binding differs",
    )
    require(
        identity["toolkit"] == result.get("toolkit") == binding.get("toolkit"),
        "toolkit binding differs",
    )
    require(
        result.get("family") == analysis["family"]
        and result.get("stage") == analysis["stage"]
        and result.get("windows") == analysis["windows"],
        "Task8 result identity differs",
    )


def validate_run_dir(run_dir, binding):
    run_dir = Path(run_dir)
    binding = json.loads(Path(binding).read_text()) if not isinstance(binding, dict) else binding
    result = json.loads((run_dir / "result.json").read_text())
    require(
        result.get("jsonl_sha") == sha(run_dir / "records.jsonl")
        and result.get("test_log_sha") == sha(run_dir / "test.log"),
        "copied raw hashes differ",
    )
    analysis = validate_records(read_jsonl(run_dir / "records.jsonl"))
    validate_binding(analysis, result, binding)
    task7.validate_telemetry(run_dir / "pre.json", "PRE")
    task7.validate_telemetry(run_dir / "post.json", "POST")
    outer = json.loads((run_dir / "outer.json").read_text())
    transcript = (run_dir / "ssh.log").read_text()
    require(outer.get("transcript_sha") == sha(run_dir / "ssh.log"), "SSH transcript digest differs")
    task7.validate_attempt_closure(result, outer, transcript)
    require(result.get("literals") == ",".join(EXACT), "result literal control differs")
    analysis["jsonl_sha"] = result["jsonl_sha"]
    return analysis


if __name__ == "__main__":
    require(len(sys.argv) == 3, "usage: analyze.py RUN_DIR BINDING")
    print(json.dumps(validate_run_dir(sys.argv[1], sys.argv[2]), indent=2, sort_keys=True))
