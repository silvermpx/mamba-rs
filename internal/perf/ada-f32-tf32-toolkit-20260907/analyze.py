#!/usr/bin/env python3
"""Independent Task7 JSONL/attempt validator and confirm-screen gate."""
import hashlib
import json
from pathlib import Path
import sys

SCHEMA = "MambaBiFixedAdaToolkitAdmissionV1"
UUID = "GPU-d1edd7be-e88d-aed6-047d-622163306f0e"
EXACT = [f"hot_{cell}:{bias}" for cell in ("a", "b", "d", "e") for bias in (0, 1)]
TF32 = [f"hot_c:{bias}" for bias in (0, 1)]
SHAPES = {
    "hot_a": [4621, 384, 1928],
    "hot_b": [4621, 768, 2304],
    "hot_c": [4621, 1928, 384],
    "hot_d": [2048, 768, 2304],
    "hot_e": [2048, 2304, 768],
}
ARMS = ["actualAUTO", "candidate", "Fast"]
DIRECTIONS = ["candidate/AUTO", "AUTO/Fast", "candidate/Fast"]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def schedule(window, start):
    require(start in (0, 1), "bad start parity")
    reverse = (window + start) % 2 == 1
    comparisons = [2, 1, 0] if reverse else [0, 1, 2]
    result = []
    for traversal, comparison in enumerate(comparisons):
        a, b = [(0, 1), (2, 0), (2, 1)][comparison]
        arms = [b, a, a, b] if reverse else [a, b, b, a]
        result.extend(
            {
                "window": window,
                "traversal": traversal,
                "comparison": comparison,
                "position": position,
                "arm": ARMS[arm],
            }
            for position, arm in enumerate(arms)
        )
    return result


def ratio(bracket):
    require(len(bracket) == 4, "pair does not contain four observations")
    comparison = bracket[0]["comparison"]
    require(comparison in range(3), "bad comparison")
    a, b = [(0, 1), (2, 0), (2, 1)][comparison]
    a, b = ARMS[a], ARMS[b]
    values = [sample["us"] for sample in bracket]
    require(all(isinstance(value, (int, float)) and value > 0 and value < float("inf") for value in values), "bad raw time")
    return sum(sample["us"] for sample in bracket if sample["arm"] == b) / sum(
        sample["us"] for sample in bracket if sample["arm"] == a
    )


def quantiles(values):
    require(values and all(value > 0 and value < float("inf") for value in values), "bad ratio set")
    values = sorted(values)
    return values[round((len(values) - 1) * 0.5)], values[round((len(values) - 1) * 0.95)]


def admitted(strata):
    require(len(strata) == 4, "literal requires four strata")
    require(all(p50 > 0 and p50 < float("inf") and p95 > 0 and p95 < float("inf") for p50, p95 in strata), "invalid literal strata")
    return all(p50 < 1 and p95 < 1 for p50, p95 in strata)


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
        require(record.get("schema") == SCHEMA, f"wrong schema line {index}")
        records.append(record)
    complete = records[-1]
    require(complete.get("kind") == "complete", "completion must be last")
    require(complete.get("preceding_lines") == len(records) - 1, "preceding line count differs")
    require(
        complete.get("preceding_jsonl_sha256") == hashlib.sha256(b"".join(lines[:-1])).hexdigest(),
        "preceding JSONL digest differs",
    )
    return records


def expected_custom(family, arm, shape, bias):
    m, k, n = shape
    pointers = None
    if family == "f32_exact_fast":
        symbol = "gemm_bi_f32_f32_s2" if arm == "actualAUTO" else "gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1"
        contract = {
            "symbol": symbol,
            "grid": [((m + 63) // 64) * ((n + 63) // 64), 1, 1],
            "block": [128, 1, 1],
            "dynamic_shared": 0,
            "parameter_words": [1065353216, 0, m, n, k, k, n, n],
            "driver_abi": (
                [[i * 8, 8] for i in range(4)] + [[32 + i * 4, 4] for i in range(8)]
                if arm == "actualAUTO"
                else [[0, 8], [8, 8], [16, 8], [24, 8], [32, 32]]
            ),
            "terminal_rejected": True,
        }
    else:
        if arm == "actualAUTO":
            contract = {
                "symbol": "gemm_bi_nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3",
                "grid": [111, 1, 1],
                "block": [256, 1, 1],
                "dynamic_shared": 98304,
                "parameter_words": [1065353216, 0, 4621, 1928, 384, 1928, 384, 384],
                "driver_abi": [[0, 8], [8, 8], [16, 8], [24, 8], [32, 32]],
                "terminal_rejected": True,
            }
        else:
            contract = {
                "symbol": "gemm_bi_nn_tf32_v1_m64n64_bk32_s2",
                "grid": [438, 1, 1],
                "block": [128, 1, 1],
                "dynamic_shared": 32768,
                "parameter_words": [4621, 1928, 384, 1928, 384, 384],
                "driver_abi": [[0, 8], [8, 8], [16, 8], [24, 8], [32, 24]],
                "terminal_rejected": True,
            }
    return contract, pointers


def validate_physical(record, family):
    literal = record["literal"]
    bias = literal.endswith(":1")
    require(record.get("bias") is bias, "physical bias differs from literal")
    require(record.get("shape") == SHAPES[literal.split(":")[0]], "physical shape differs from literal")
    expected_auto = "Legacy" if family == "f32_exact_fast" else "Tf32RnaM128N128S3"
    expected_candidate = "F32Sm89N64CopyPlan" if family == "f32_exact_fast" else "Tf32M64S2"
    require(record.get("actual_auto") == expected_auto and record.get("candidate") == expected_candidate, "wrong physical enum")
    for flag in (
        "custom_bits_equal", "fast_repeat_bits", "poison_upload_readback", "noop_rejected",
        "guards", "immutable_inputs", "bias_orientation", "finite_ordering_controls",
    ):
        require(record.get(flag) is True, "failed physical flag " + flag)
    graphs = record["graphs"]
    common_inputs = None
    for arm in ("actualAUTO", "candidate"):
        contract, _ = expected_custom(family, arm, record["shape"], bias)
        one, twenty = graphs[arm]["one"], graphs[arm]["twenty"]
        require(len(one) == 1 and len(twenty) == 20, "wrong custom graph node count")
        for node in one + twenty:
            for key, value in contract.items():
                require(node.get(key) == value, f"wrong {arm} physical {key}")
            require(len(node.get("pointers", [])) == 4 and all(isinstance(value, int) and value > 0 for value in node["pointers"][:3]), "bad custom pointers")
            require((node["pointers"][3] > 0) is bias, "wrong captured bias pointer state")
            require(node["pointers"] == one[0]["pointers"], "custom twenty-node pointers differ from one-node capture")
        inputs = one[0]["pointers"][1:]
        if common_inputs is None:
            common_inputs = inputs
        else:
            require(inputs == common_inputs, "custom arms do not share A/B/bias pointers")
    fast = graphs["Fast"]
    one, twenty = fast["one"], fast["twenty"]
    expected_one = 2 if bias else 1
    require(len(one) == expected_one and len(twenty) == expected_one * 20, "wrong Fast graph node count")
    for nodes, logical in ((one, 1), (twenty, 20)):
        require(all(node["grid"][0] > 0 and node["block"][0] > 0 and node["symbol"] for node in nodes), "bad Fast graph inventory")
        require(sum(node["symbol"] == "bias_broadcast" for node in nodes) == logical * int(bias), "wrong timed Fast bias inventory")


def validate_records(records):
    identities = [record for record in records if record.get("kind") == "identity"]
    require(len(identities) == 1 and records[0] is identities[0], "identity must be unique and first")
    identity = identities[0]
    family = identity.get("family")
    require(family in ("f32_exact_fast", "tf32"), "bad family")
    stage = identity.get("stage")
    windows = {"smoke1": 1, "screen21": 21, "confirm101": 101}.get(stage)
    require(windows == identity.get("windows"), "stage/window mismatch")
    require(identity.get("toolkit") in ("12.8", "13.0"), "bad toolkit")
    require(identity.get("tuning_revision") == 43, "bad revision")
    require(identity.get("numeric_abi_revision") == 5, "bad numeric ABI revision")
    require(identity.get("schedule_revision") == 8, "bad schedule revision")
    require(identity.get("uuid") == UUID and identity.get("cc") == "8.9" and identity.get("sm_count") == 142, "bad device")
    require(identity.get("compiler_target") == "sm_89" and identity.get("nvrtc_library_known") is True, "bad compiler")
    require(identity.get("vendor_compute") == "CUBLAS_COMPUTE_32F_FAST_TF32", "bad Fast compute")
    require(identity.get("vendor_algorithm") == "CUBLAS_GEMM_DEFAULT", "bad vendor algorithm")
    require(identity.get("vendor_math") == "CUBLAS_DEFAULT_MATH", "bad math mode")
    require(identity.get("vendor_pointer_mode") == "CUBLAS_POINTER_MODE_HOST", "bad pointer mode")
    require(identity.get("vendor_atomics") == "CUBLAS_ATOMICS_NOT_ALLOWED", "bad atomics mode")
    require(identity.get("paths") == ["eager", "graph"] and identity.get("start_parities") == [0, 1], "bad strata")
    require(identity.get("warmup_eager") == 128 and identity.get("logical_ops") == 20, "bad timing constants")
    for key in ("source_sha", "binary_sha", "fixed_source_digest", "fixed_invocation_digest", "fixed_artifact_digest", "header_manifest_digest", "nvrtc_library_domain"):
        value = identity.get(key)
        require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "bad identity " + key)
    literals = identity.get("literal_control", "").split(",")
    inventory = EXACT if family == "f32_exact_fast" else TF32
    require(literals and len(literals) == len(set(literals)) and set(literals) <= set(inventory), "bad literal control")
    if stage in ("smoke1", "screen21"):
        require(literals == inventory and identity.get("screen_sha") is None and identity.get("screen_artifact_sha") is None, "full stage literal/screen binding differs")
    else:
        require(identity.get("screen_sha") and identity.get("screen_artifact_sha") == identity["fixed_artifact_digest"], "confirm screen/artifact binding differs")
    allowed_kinds = {
        "identity", "physical", "sample", "pair", "summary", "configuration_complete",
        "literal_decision", "complete",
    }
    require(all(record.get("kind") in allowed_kinds for record in records), "foreign record kind")
    for record in records[1:-1]:
        require(record.get("family") == family and record.get("stage") == stage, "foreign record family/stage")
        require(record.get("literal") in literals, "foreign record literal")
    physical = [record for record in records if record.get("kind") == "physical"]
    require(len(physical) == len(literals) and {record["literal"] for record in physical} == set(literals), "physical literal closure differs")
    for record in physical:
        validate_physical(record, family)
    own = {}
    configurations = 0
    for literal in literals:
        own[literal] = []
        for path in ("eager", "graph"):
            for start in (0, 1):
                base = lambda record: record.get("literal") == literal and record.get("path") == path and record.get("start_parity") == start
                samples = [record for record in records if record.get("kind") == "sample" and base(record)]
                pairs = [record for record in records if record.get("kind") == "pair" and base(record)]
                summaries = [record for record in records if record.get("kind") == "summary" and base(record)]
                completions = [record for record in records if record.get("kind") == "configuration_complete" and base(record)]
                require(len(samples) == 12 * windows and len(pairs) == 3 * windows and len(summaries) == 3 and len(completions) == 1, "configuration count closure differs")
                expected_schedule = [entry for window in range(windows) for entry in schedule(window, start)]
                for chronology, (sample, expected) in enumerate(zip(samples, expected_schedule)):
                    require(sample.get("chronology") == chronology, "raw chronology differs")
                    require(all(sample.get(key) == value for key, value in expected.items()), "raw arm/position/traversal/parity differs")
                    require(sample.get("logical_ops") == 20, "raw logical operation count differs")
                recomputed = [[] for _ in range(3)]
                for index, bracket in enumerate(samples[i:i+4] for i in range(0, len(samples), 4)):
                    value = ratio(bracket)
                    pair = pairs[index]
                    comparison = bracket[0]["comparison"]
                    require(pair.get("window") == bracket[0]["window"] and pair.get("traversal") == bracket[0]["traversal"] and pair.get("comparison") == comparison, "pair key differs")
                    require(pair.get("direction") == DIRECTIONS[comparison], "pair direction differs")
                    require(pair.get("observations") == [index*4+i for i in range(4)], "pair raw indices differ")
                    require(abs(pair.get("ratio") - value) <= 1e-12, "pair ratio differs")
                    recomputed[comparison].append(value)
                summaries.sort(key=lambda record: record["comparison"])
                for comparison, summary in enumerate(summaries):
                    p50, p95 = quantiles(recomputed[comparison])
                    require(summary.get("direction") == DIRECTIONS[comparison] and summary.get("windows") == windows, "summary key differs")
                    require(abs(summary.get("p50") - p50) <= 1e-12 and abs(summary.get("p95") - p95) <= 1e-12, "forged summary")
                completion = completions[0]
                require(completion.get("samples") == 12 * windows and completion.get("pairs") == 3 * windows and completion.get("summaries") == 3 and completion.get("physical_bits_guards_inputs") is True, "configuration completion forged")
                own[literal].append((summaries[0]["p50"], summaries[0]["p95"]))
                configurations += 1
    decisions = [record for record in records if record.get("kind") == "literal_decision"]
    require(len(decisions) == len(literals) and {record["literal"] for record in decisions} == set(literals), "literal decision closure differs")
    eligible = []
    for decision in decisions:
        literal = decision["literal"]
        is_admitted = admitted(own[literal])
        require(decision.get("own_admission") is is_admitted, "forged literal admission")
        if is_admitted:
            eligible.append(literal)
    complete = records[-1]
    require(complete.get("configurations") == configurations == len(literals) * 4, "complete configuration closure differs")
    require(complete.get("expected_configurations") == configurations, "complete expected count differs")
    require(complete.get("samples") == configurations * 12 * windows, "complete raw count differs")
    require(complete.get("pairs") == configurations * 3 * windows and complete.get("summaries") == configurations * 3, "complete derived count differs")
    require(complete.get("literals") == len(literals) and complete.get("rejected") == 0 and complete.get("all_gates_passed") is True and complete.get("passed") is True, "completion flags differ")
    expected_kind_counts = {
        "identity": 1,
        "physical": len(literals),
        "sample": configurations * 12 * windows,
        "pair": configurations * 3 * windows,
        "summary": configurations * 3,
        "configuration_complete": configurations,
        "literal_decision": len(literals),
        "complete": 1,
    }
    actual_kind_counts = {kind: sum(record.get("kind") == kind for record in records) for kind in allowed_kinds}
    require(actual_kind_counts == expected_kind_counts, "global record-kind closure differs")
    return {"valid": True, "family": family, "stage": stage, "toolkit": identity["toolkit"], "windows": windows, "literals": literals, "eligible": eligible, "identity": identity, "own_strata": own}


def validate_telemetry(path, phase):
    record = json.loads(Path(path).read_text())
    require(record.get("phase") == phase and record.get("gpu_exit") == record.get("apps_exit") == 0, phase + " telemetry exit")
    fields = [part.strip() for part in record["gpu"].strip().split(",")]
    require(fields[:3] == [UUID, "NVIDIA RTX 6000 Ada Generation", "8.9"] and not record["apps"].strip(), phase + " telemetry identity/apps")
    if phase == "PRE":
        require(fields[3:] == ["0 %", "0 %"], "PRE not quiet")


def validate_attempt_closure(result, outer, transcript):
    require(result.get("test_exit") == result.get("post_exit") == 0, "test/POST exit closure differs")
    require(outer.get("outer_ssh_exit") == 0 and outer.get("wrapper_complete") is True, "wrapper/outer SSH exit closure differs")
    require(transcript.endswith("WRAPPER_COMPLETE\n"), "wrapper completion absent from transcript tail")


def validate_binding(analysis, result, binding):
    identity = analysis["identity"]
    require(identity["source_sha"] == result.get("source_sha") == binding.get("measured_source_sha"), "source binding differs")
    require(identity["binary_sha"] == result.get("binary_sha") == binding.get("binary_sha"), "binary binding differs")
    require(identity["toolkit"] == result.get("toolkit") == binding.get("toolkit"), "toolkit binding differs")
    require(
        identity["family"] == result.get("family")
        and identity["stage"] == result.get("stage")
        and identity["windows"] == result.get("windows"),
        "run identity differs",
    )


def validate_run_dir(run_dir, binding, screen_path=None):
    run_dir = Path(run_dir)
    binding = json.loads(Path(binding).read_text()) if not isinstance(binding, dict) else binding
    result = json.loads((run_dir / "result.json").read_text())
    require(result.get("jsonl_sha") == sha(run_dir / "records.jsonl") and result.get("test_log_sha") == sha(run_dir / "test.log"), "copied raw hashes differ")
    analysis = validate_records(read_jsonl(run_dir / "records.jsonl"))
    validate_binding(analysis, result, binding)
    validate_telemetry(run_dir / "pre.json", "PRE")
    validate_telemetry(run_dir / "post.json", "POST")
    outer = json.loads((run_dir / "outer.json").read_text())
    transcript = (run_dir / "ssh.log").read_text()
    require(outer.get("transcript_sha") == sha(run_dir / "ssh.log"), "SSH transcript digest differs")
    validate_attempt_closure(result, outer, transcript)
    require(result.get("literals") == ",".join(analysis["literals"]), "result literal control differs")
    if analysis["stage"] == "confirm101":
        require(screen_path is not None, "independent confirm analysis requires its screen")
        screen = validate_screen_for_confirm(
            screen_path, binding, analysis["family"], analysis["toolkit"], result["literals"]
        )
        require(analysis["identity"]["screen_sha"] == screen["screen_sha"], "confirm screen digest differs")
        require(analysis["identity"]["screen_artifact_sha"] == screen["artifact_sha"], "confirm screen artifact differs")
    analysis["jsonl_sha"] = result["jsonl_sha"]
    return analysis


def validate_screen_for_confirm(screen_path, binding, family, toolkit, literals):
    screen_path = Path(screen_path)
    require(screen_path.name == "records.jsonl", "screen path must be canonical records.jsonl")
    analysis = validate_run_dir(screen_path.parent, binding)
    return confirm_gate(analysis, binding, family, toolkit, literals, sha(screen_path))


def confirm_gate(analysis, binding, family, toolkit, literals, screen_sha):
    require(analysis["stage"] == "screen21" and analysis["family"] == family and analysis["toolkit"] == toolkit, "foreign confirm screen")
    require(analysis["identity"]["source_sha"] == binding["measured_source_sha"], "confirm screen source differs")
    require(analysis["identity"]["binary_sha"] == binding["binary_sha"], "confirm screen binary differs")
    requested = literals.split(",")
    require(requested and len(requested) == len(set(requested)), "bad confirm literal subset")
    require(requested == analysis["eligible"], "confirm literals must equal recomputed eligible screen subset")
    require(requested, "screen admitted no confirm literals")
    return {"screen_sha": screen_sha, "artifact_sha": analysis["identity"]["fixed_artifact_digest"]}


if __name__ == "__main__":
    require(len(sys.argv) in (3, 4), "usage: analyze.py RUN_DIR BINDING [SCREEN_JSONL]")
    print(json.dumps(validate_run_dir(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else None), indent=2, sort_keys=True))
