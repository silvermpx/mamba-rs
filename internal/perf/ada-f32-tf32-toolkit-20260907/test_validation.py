#!/usr/bin/env python3
"""Host-only regression and adversarial tests for the Task7 evidence gate."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("task7_analyze", HERE / "analyze.py")
analyze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze)
RUN = HERE / "cuda128-exact-smoke2"
BINDING = json.loads((HERE / "cuda128-binding-final3.json").read_text())


def records():
    result = copy.deepcopy(analyze.read_jsonl(RUN / "records.jsonl"))
    result[0].setdefault("numeric_abi_revision", 5)
    result[0].setdefault("schedule_revision", 8)
    return result


def first(items, kind):
    return next(record for record in items if record.get("kind") == kind)


def recompute_literal_loss(items, literal):
    """Make candidate/AUTO lose honestly, then rebuild every dependent record."""
    for path in ("eager", "graph"):
        for start in (0, 1):
            selected = [
                record
                for record in items
                if record.get("kind") == "sample"
                and record.get("literal") == literal
                and record.get("path") == path
                and record.get("start_parity") == start
            ]
            for sample in selected:
                if sample["comparison"] == 0 and sample["arm"] == "candidate":
                    sample["us"] *= 4
            ratios = [[] for _ in range(3)]
            for bracket in (selected[index : index + 4] for index in range(0, len(selected), 4)):
                value = analyze.ratio(bracket)
                comparison = bracket[0]["comparison"]
                ratios[comparison].append(value)
                pair = next(
                    record
                    for record in items
                    if record.get("kind") == "pair"
                    and record.get("literal") == literal
                    and record.get("path") == path
                    and record.get("start_parity") == start
                    and record.get("window") == bracket[0]["window"]
                    and record.get("traversal") == bracket[0]["traversal"]
                    and record.get("comparison") == comparison
                )
                pair["ratio"] = value
            for comparison in range(3):
                summary = next(
                    record
                    for record in items
                    if record.get("kind") == "summary"
                    and record.get("literal") == literal
                    and record.get("path") == path
                    and record.get("start_parity") == start
                    and record.get("comparison") == comparison
                )
                summary["p50"], summary["p95"] = analyze.quantiles(ratios[comparison])
    decision = next(
        record
        for record in items
        if record.get("kind") == "literal_decision" and record.get("literal") == literal
    )
    decision["own_admission"] = False


class Task7ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.valid = analyze.validate_records(records())

    def assert_rejected(self, mutation):
        candidate = records()
        mutation(candidate)
        with self.assertRaises(ValueError):
            analyze.validate_records(candidate)

    def test_actual_rust_json_parses_and_abi_is_json_arrays(self):
        parsed = analyze.read_jsonl(RUN / "records.jsonl")
        physical = first(parsed, "physical")
        abi = physical["graphs"]["candidate"]["one"][0]["driver_abi"]
        self.assertTrue(abi)
        self.assertTrue(all(isinstance(entry, list) and len(entry) == 2 for entry in abi))
        self.assertTrue(self.valid["valid"])

    def test_tf32_analyzer_expects_rna_auto_not_old_portable_bundle(self):
        contract, _ = analyze.expected_custom("tf32", "actualAUTO", analyze.SHAPES["hot_c"], False)
        self.assertEqual(contract["symbol"], "gemm_bi_nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3")
        self.assertEqual(contract["grid"], [111, 1, 1])
        self.assertEqual(contract["dynamic_shared"], 98_304)
        self.assertEqual(contract["driver_abi"][-1], [32, 32])
        self.assertEqual(
            contract["parameter_words"],
            [1_065_353_216, 0, 4621, 1928, 384, 1928, 384, 384],
        )
        self.assertNotEqual(contract["parameter_words"], [4621, 1928, 384, 1928, 384, 384])

    def test_missing_duplicate_foreign_literal_and_raw_records_reject(self):
        cases = {
            "missing literal": lambda rs: rs[0].__setitem__("literal_control", ",".join(analyze.EXACT[:-1])),
            "duplicate literal": lambda rs: rs[0].__setitem__("literal_control", ",".join(analyze.EXACT + [analyze.EXACT[0]])),
            "foreign literal": lambda rs: rs[0].__setitem__("literal_control", ",".join(analyze.EXACT[:-1] + ["hot_c:0"])),
            "missing raw": lambda rs: rs.pop(next(i for i, r in enumerate(rs) if r.get("kind") == "sample")),
            "duplicate raw": lambda rs: rs.insert(-1, copy.deepcopy(first(rs, "sample"))),
            "foreign raw": lambda rs: rs.insert(-1, {**copy.deepcopy(first(rs, "sample")), "literal": "hot_c:0"}),
            "foreign kind": lambda rs: rs.insert(-1, {"schema": analyze.SCHEMA, "kind": "alien", "family": "f32_exact_fast", "stage": "smoke1", "literal": "hot_a:0"}),
        }
        for name, mutation in cases.items():
            with self.subTest(name=name):
                self.assert_rejected(mutation)

    def test_malformed_identity_vendor_and_physical_contracts_reject(self):
        def identity(key, value):
            return lambda rs: rs[0].__setitem__(key, value)

        def node(key, value):
            return lambda rs: first(rs, "physical")["graphs"]["candidate"]["one"][0].__setitem__(key, value)

        cases = {
            "stage": identity("stage", "confirm101"),
            "revision": identity("tuning_revision", 44),
            "missing numeric ABI": lambda rs: rs[0].pop("numeric_abi_revision"),
            "wrong numeric ABI": identity("numeric_abi_revision", 4),
            "missing schedule revision": lambda rs: rs[0].pop("schedule_revision"),
            "wrong schedule revision": identity("schedule_revision", 7),
            "source": identity("source_sha", "x" * 64),
            "binary": identity("binary_sha", "f" * 63),
            "device": identity("uuid", "GPU-foreign"),
            "toolkit": identity("toolkit", "12.7"),
            "vendor compute": identity("vendor_compute", "CUBLAS_COMPUTE_32F"),
            "vendor math": identity("vendor_math", "CUBLAS_TF32_TENSOR_OP_MATH"),
            "vendor pointer": identity("vendor_pointer_mode", "CUBLAS_POINTER_MODE_DEVICE"),
            "vendor atomics": identity("vendor_atomics", "CUBLAS_ATOMICS_ALLOWED"),
            "abi": node("driver_abi", [[0, 8]]),
            "captured args": node("parameter_words", [0] * 8),
            "captured pointers": node("pointers", [1, 2, 3, 4]),
            "terminal": node("terminal_rejected", False),
            "shape": lambda rs: first(rs, "physical").__setitem__("shape", analyze.SHAPES["hot_b"]),
            "physical flag": lambda rs: first(rs, "physical").__setitem__("immutable_inputs", False),
        }
        for name, mutation in cases.items():
            with self.subTest(name=name):
                self.assert_rejected(mutation)

    def test_raw_schedule_arithmetic_and_completion_forgery_reject(self):
        def sample(key, value):
            return lambda rs: first(rs, "sample").__setitem__(key, value)

        cases = {
            "arm": sample("arm", "Fast"),
            "position": sample("position", 3),
            "traversal": sample("traversal", 2),
            "parity": sample("start_parity", 1),
            "nonpositive": sample("us", 0),
            "nonfinite": sample("us", float("inf")),
            "pair ratio": lambda rs: first(rs, "pair").__setitem__("ratio", 7),
            "pair observations": lambda rs: first(rs, "pair").__setitem__("observations", [1, 2, 3, 4]),
            "summary": lambda rs: first(rs, "summary").__setitem__("p95", 7),
            "config completion": lambda rs: first(rs, "configuration_complete").__setitem__("pairs", 2),
            "global completion": lambda rs: rs[-1].__setitem__("samples", rs[-1]["samples"] - 1),
            "admission": lambda rs: first(rs, "literal_decision").__setitem__("own_admission", False),
        }
        for name, mutation in cases.items():
            with self.subTest(name=name):
                self.assert_rejected(mutation)

    def test_failed_inner_outer_and_binding_closure_reject(self):
        result = json.loads((RUN / "result.json").read_text())
        outer = json.loads((RUN / "outer.json").read_text())
        transcript = (RUN / "ssh.log").read_text()
        analyze.validate_attempt_closure(result, outer, transcript)
        for key in ("test_exit", "post_exit"):
            bad = copy.deepcopy(result)
            bad[key] = 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                analyze.validate_attempt_closure(bad, outer, transcript)
        for key, value in (("outer_ssh_exit", 1), ("wrapper_complete", False)):
            bad = copy.deepcopy(outer)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                analyze.validate_attempt_closure(result, bad, transcript)
        with self.assertRaises(ValueError):
            analyze.validate_attempt_closure(result, outer, transcript + "trailing-data\n")
        analyze.validate_binding(self.valid, result, BINDING)
        for key in ("measured_source_sha", "binary_sha", "toolkit"):
            bad = copy.deepcopy(BINDING)
            bad[key] = "foreign"
            with self.subTest(binding=key), self.assertRaises(ValueError):
                analyze.validate_binding(self.valid, result, bad)

    def test_honestly_recomputed_loss_is_valid_and_siblings_survive(self):
        changed = records()
        recompute_literal_loss(changed, "hot_a:0")
        result = analyze.validate_records(changed)
        self.assertNotIn("hot_a:0", result["eligible"])
        self.assertEqual(result["eligible"], analyze.EXACT[1:])

    def test_mixed_p95_is_valid_loss(self):
        p50, p95 = analyze.quantiles([0.8] * 19 + [1.2, 1.4])
        self.assertEqual((p50, p95), (0.8, 1.2))
        self.assertFalse(analyze.admitted([(p50, p95), (0.8, 0.8), (0.8, 0.8), (0.8, 0.8)]))

    def test_confirm_gate_requires_exact_recomputed_subset_and_identity(self):
        screen = copy.deepcopy(self.valid)
        screen["stage"] = "screen21"
        screen["identity"]["stage"] = "screen21"
        requested = ",".join(screen["eligible"])
        expected = analyze.confirm_gate(screen, BINDING, "f32_exact_fast", "12.8", requested, "a" * 64)
        self.assertEqual(expected["artifact_sha"], screen["identity"]["fixed_artifact_digest"])
        rejects = [
            ("tf32", "12.8", requested),
            ("f32_exact_fast", "13.0", requested),
            ("f32_exact_fast", "12.8", ",".join(screen["eligible"][:-1])),
            ("f32_exact_fast", "12.8", requested + "," + screen["eligible"][0]),
            ("f32_exact_fast", "12.8", requested + ",hot_c:0"),
            ("f32_exact_fast", "12.8", ""),
        ]
        for family, toolkit, literals in rejects:
            with self.subTest(family=family, toolkit=toolkit, literals=literals), self.assertRaises(ValueError):
                analyze.confirm_gate(screen, BINDING, family, toolkit, literals, "a" * 64)
        for key in ("source_sha", "binary_sha"):
            foreign = copy.deepcopy(screen)
            foreign["identity"][key] = "f" * 64
            with self.subTest(identity=key), self.assertRaises(ValueError):
                analyze.confirm_gate(foreign, BINDING, "f32_exact_fast", "12.8", requested, "a" * 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
