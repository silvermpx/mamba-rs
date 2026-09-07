#!/usr/bin/env python3
"""Host-only adversarial tests for the Task8 post-AUTO evidence adapter."""

import copy
import importlib.util
import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
TASK7 = HERE.parent / "ada-f32-tf32-toolkit-20260907"
SPEC = importlib.util.spec_from_file_location("task8_analyze", HERE / "analyze.py")
analyze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze)
RUN_SPEC = importlib.util.spec_from_file_location("task8_run", HERE / "run.py")
run = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(run)


def task7_records():
    path = TASK7 / "cuda128-final5b-exact-smoke" / "records.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def post_records():
    records = copy.deepcopy(task7_records())
    for record in records:
        record["schema"] = analyze.SCHEMA
        if "family" in record:
            record["family"] = analyze.FAMILY
        if record.get("kind") == "identity":
            record["tuning_revision"] = 44
            record["promotion_basis"] = copy.deepcopy(analyze.PROMOTION_BASIS)
        if record.get("kind") == "physical":
            record["former_incumbent"] = "Legacy"
            record["actual_auto"] = "F32Sm89N64CopyPlan"
            record["public_auto_enum_verified"] = True
            record.pop("candidate")
            graphs = record["graphs"]
            record["graphs"] = {
                "Legacy": graphs["actualAUTO"],
                "AUTO": graphs["candidate"],
                "Fast": graphs["Fast"],
            }
        if record.get("kind") == "sample":
            record["arm"] = {"actualAUTO": "Legacy", "candidate": "AUTO", "Fast": "Fast"}[
                record["arm"]
            ]
        if record.get("kind") in ("pair", "summary"):
            record["direction"] = {
                "candidate/AUTO": "AUTO/Legacy",
                "AUTO/Fast": "Legacy/Fast",
                "candidate/Fast": "AUTO/Fast",
            }[record["direction"]]
        if record.get("kind") == "literal_decision":
            record["own_rule"] = (
                "AUTO/Legacy p50 and p95 < 1 in all four path/start strata"
            )
    return records


def first(records, kind):
    return next(record for record in records if record.get("kind") == kind)


class Task8ValidationTests(unittest.TestCase):
    def assert_rejected(self, mutation):
        candidate = post_records()
        mutation(candidate)
        with self.assertRaises(ValueError):
            analyze.validate_records(candidate)

    def test_valid_post_schema_adapts_shared_task7_arithmetic(self):
        result = analyze.validate_records(post_records())
        self.assertTrue(result["valid"])
        self.assertEqual(result["family"], analyze.FAMILY)
        self.assertEqual(result["stage"], "smoke1")
        self.assertEqual(result["literals"], analyze.EXACT)

    def test_wrong_epoch_stage_family_toolkit_or_role_rejects(self):
        cases = {
            "epoch": lambda rs: rs[0].__setitem__("tuning_revision", 43),
            "stage": lambda rs: rs[0].__setitem__("stage", "screen21"),
            "family": lambda rs: rs[0].__setitem__("family", "f32_exact_fast"),
            "toolkit": lambda rs: rs[0].__setitem__("toolkit", "13.2"),
            "promotion basis": lambda rs: rs[0]["promotion_basis"].__setitem__(
                "cuda128_confirm_sha", "f" * 64
            ),
            "arm": lambda rs: first(rs, "sample").__setitem__("arm", "candidate"),
            "direction": lambda rs: first(rs, "summary").__setitem__(
                "direction", "candidate/AUTO"
            ),
        }
        for name, mutation in cases.items():
            with self.subTest(name=name):
                self.assert_rejected(mutation)

    def test_wrong_old_or_new_auto_physical_contract_rejects(self):
        cases = {
            "former enum": lambda rs: first(rs, "physical").__setitem__(
                "former_incumbent", "F32N128S2"
            ),
            "AUTO enum": lambda rs: first(rs, "physical").__setitem__(
                "actual_auto", "Legacy"
            ),
            "AUTO enum gate": lambda rs: first(rs, "physical").__setitem__(
                "public_auto_enum_verified", False
            ),
            "AUTO symbol": lambda rs: first(rs, "physical")["graphs"]["AUTO"]["one"][
                0
            ].__setitem__("symbol", "gemm_bi_f32_f32_s2"),
            "Legacy symbol": lambda rs: first(rs, "physical")["graphs"]["Legacy"]["one"][
                0
            ].__setitem__("symbol", "gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1"),
            "ABI": lambda rs: first(rs, "physical")["graphs"]["AUTO"]["one"][0].__setitem__(
                "driver_abi", [[0, 8]]
            ),
            "poison": lambda rs: first(rs, "physical").__setitem__(
                "poison_upload_readback", False
            ),
        }
        for name, mutation in cases.items():
            with self.subTest(name=name):
                self.assert_rejected(mutation)

    def test_valid_loss_and_mixed_p95_remain_evidence_not_admission(self):
        self.assertFalse(analyze.admitted([(1.1, 1.2)] * 4))
        p50, p95 = analyze.quantiles([0.8] * 19 + [1.2, 1.4])
        self.assertEqual((p50, p95), (0.8, 1.2))
        self.assertFalse(analyze.admitted([(p50, p95), (0.8, 0.8), (0.8, 0.8), (0.8, 0.8)]))

    def test_binding_and_exit_closure_reject_foreign_or_failed_attempt(self):
        analysis = analyze.validate_records(post_records())
        identity = analysis["identity"]
        result = {
            "test_exit": 0,
            "post_exit": 0,
            "source_sha": identity["source_sha"],
            "binary_sha": identity["binary_sha"],
            "toolkit": identity["toolkit"],
            "family": identity["family"],
            "stage": identity["stage"],
            "windows": identity["windows"],
        }
        binding = {
            "measured_source_sha": identity["source_sha"],
            "toolkit": identity["toolkit"],
            "binaries": {
                "/root/target/release/deps/gemm_bi_fixed_performance-deadbeef": identity[
                    "binary_sha"
                ],
                "/root/target/release/deps/gemm_bi_fixed_correctness-11111111": "1" * 64,
                "/root/target/release/deps/gemm_bi_fixed_sm89_exact_n64-22222222": "2" * 64,
            },
        }
        analyze.validate_binding(analysis, result, binding)
        for key in ("measured_source_sha", "toolkit"):
            bad = copy.deepcopy(binding)
            bad[key] = "foreign"
            with self.subTest(key=key), self.assertRaises(ValueError):
                analyze.validate_binding(analysis, result, bad)
        bad = copy.deepcopy(binding)
        bad["binaries"][
            "/root/target/release/deps/gemm_bi_fixed_performance-deadbeef"
        ] = "foreign"
        with self.assertRaises(ValueError):
            analyze.validate_binding(analysis, result, bad)
        for mutation in ("missing", "ambiguous"):
            bad = copy.deepcopy(binding)
            if mutation == "missing":
                del bad["binaries"][
                    "/root/target/release/deps/gemm_bi_fixed_performance-deadbeef"
                ]
            else:
                bad["binaries"][
                    "/root/other/release/deps/gemm_bi_fixed_performance-cafebabe"
                ] = identity["binary_sha"]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                analyze.validate_binding(analysis, result, bad)

    def test_actual_builder_binding_schema_selects_only_performance_executable(self):
        binding = json.loads((HERE / "cuda128-binding-final1.json").read_text())
        performance_path, performance_sha = next(
            (path, digest)
            for path, digest in binding["binaries"].items()
            if Path(path).name.startswith("gemm_bi_fixed_performance-")
        )
        records = post_records()
        identity = first(records, "identity")
        identity["source_sha"] = binding["measured_source_sha"]
        identity["binary_sha"] = performance_sha
        analysis = analyze.validate_records(records)
        result = {
            "source_sha": binding["measured_source_sha"],
            "binary_sha": performance_sha,
            "toolkit": binding["toolkit"],
            "family": analysis["family"],
            "stage": analysis["stage"],
            "windows": analysis["windows"],
        }
        analyze.validate_binding(analysis, result, binding)
        wrong_role = copy.deepcopy(binding)
        other_path, other_sha = next(
            (path, digest)
            for path, digest in wrong_role["binaries"].items()
            if path != performance_path
        )
        wrong_role["binaries"][performance_path] = other_sha
        wrong_role["binaries"][other_path] = performance_sha
        with self.assertRaises(ValueError):
            analyze.validate_binding(analysis, result, wrong_role)

    def test_functional_inventory_uses_fixed_rna_all_three_and_triad_only_132(self):
        fixed = {
            (
                "gemm_bi_fixed_correctness",
                "fixed_sm89_rna_wide_actual_auto_hot_a_route_and_graph",
            ),
            (
                "gemm_bi_fixed_correctness",
                "fixed_sm89_rna_wide_actual_auto_all_cells_prefix_views_and_graph_bits",
            ),
        }
        triad = {
            ("gemm_bi_tf32_cohort_binding", "tf32_cohort_binds_on_this_board"),
            (
                "gemm_bi_tf32_cohort_binding",
                "sm89_tf32_bias_cohort_serves_the_qualified_wide_epilogues",
            ),
        }
        for toolkit in ("12.8", "13.0", "13.2"):
            checks = run.functional_checks(toolkit)
            inventory = {(stem, test_name) for _label, stem, test_name in checks}
            self.assertTrue(fixed <= inventory, toolkit)
            self.assertEqual(triad <= inventory, toolkit == "13.2", toolkit)
            self.assertTrue(
                {stem for _label, stem, _test_name in checks}
                <= set(run.BOUND_BINARY_STEMS),
                toolkit,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
