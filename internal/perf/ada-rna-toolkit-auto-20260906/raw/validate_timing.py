#!/usr/bin/env python3
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = {
    "12.8": ["timing101-cuda128-abde-m64.log", "timing101-cuda128-c-m128.log"],
    "13.0": ["timing101-cuda130-abde-m64.log", "timing101-cuda130-c-m128.log"],
    "13.2": ["timing101-cuda132-all-m64.log"],
}
IDENTITY = {
    "12.8": {
        "nvrtc": [12, 8],
        "fixed_invocation_digest": "ae8e2e3db0255db26419c8292c70ea945f46076770ddb3522d34443addeaf374",
        "fixed_artifact_digest": "c71288517eb76b839ce23b2f915b010ca73b817b08d5aa0eb2e88a7b2c9f2a3e",
        "header_manifest_digest": "9924f331b7c7e70041f74e8a9b39d072930c493c921ddf19beb34b6263682fc8",
        "nvrtc_library_domain": "26b0a3a02044ffcbc1693fd83e9261beffa692a4fbcfe3ac5e9d8c87980bb155",
    },
    "13.0": {
        "nvrtc": [13, 0],
        "fixed_invocation_digest": "4d6815a9cdc06297113b72dc2d9ae9fac46b50a0583c561f358727d4a6f2cf49",
        "fixed_artifact_digest": "c90cd431d3c2df95e849b99f6fcef8e6a7e28f64d97e851ec39f2445a7dc5822",
        "header_manifest_digest": "7801fef3bdeb57597ff2028997aa9d6190fe63dec216b8bf0d1d5a07235f8685",
        "nvrtc_library_domain": "709b91c36bfb0ed966ee69adc8d6f87ff110eecf3dfb5060367f183ce614eb0d",
    },
    "13.2": {
        "nvrtc": [13, 2],
        "fixed_invocation_digest": "4e99acb556ad1f6bb2aafa382f1618db9c9ecf37286fa7da37a62d86399c14ce",
        "fixed_artifact_digest": "c6dc1ee707ceceb512b58f226f8b5288097162c5257c41ef89a2e235fda1945f",
        "header_manifest_digest": "e893dcebd4b2eb9d2e8cd84721c99684550413437d318113a8c45a6fa9ac1f73",
        "nvrtc_library_domain": "d031a53eb97235b70f62f652932db1bdf728ea229c8ca809d53c5ffd91642687",
    },
}
SOURCE = "7ccad9938c4ee7cb0080d62b7053e082537fcb2ad184288ffc59a513cf526301"
RNA_SYMBOL = "gemm_bi_nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3"
CELLS = {"hot_a", "hot_b", "hot_c", "hot_d", "hot_e"}
ORDERS = {"auto_forced_vendor", "vendor_forced_auto"}


def quantile(values, index):
    return sorted(values)[index]


summary = {}
for version, filenames in FILES.items():
    objects = []
    for filename in filenames:
        for line in (ROOT / filename).read_text().splitlines():
            if line.startswith("{"):
                objects.append(json.loads(line))
    records = [o for o in objects if o["schema"] == "MambaBiFixedExplicitForcedRungV2"]
    completions = [
        o for o in objects if o["schema"] == "MambaBiFixedExplicitForcedRungCompleteV2"
    ]
    rejections = [o for o in objects if "Rejected" in o["schema"]]
    assert len(records) == 40, (version, len(records))
    assert not rejections, (version, rejections)
    assert sum(o["records"] for o in completions) == 40
    assert sum(o["rejected"] for o in completions) == 0
    assert completions and all(o["passed"] for o in completions)
    keys = {(o["cell"], o["bias"], o["path"], o["order"]) for o in records}
    expected_keys = {
        (cell, bias, path, order)
        for cell in CELLS
        for bias in (False, True)
        for path in ("eager", "graph")
        for order in ORDERS
    }
    assert keys == expected_keys, (version, expected_keys - keys, keys - expected_keys)

    ratios = {}
    for record in records:
        expected_identity = IDENTITY[version]
        assert record["cc"] == "8.9" and record["sm_count"] == 142
        assert record["nvrtc"] == expected_identity["nvrtc"]
        assert record["nvrtc_library_known"] is True
        assert record["fixed_source_digest"] == SOURCE
        for field in (
            "fixed_invocation_digest",
            "fixed_artifact_digest",
            "header_manifest_digest",
            "nvrtc_library_domain",
        ):
            assert record[field] == expected_identity[field], (version, field)
        assert record["tuning_table_revision"] == 41
        assert record["auto_tile"] == "Tf32RnaM128N128S3"
        expected_old = (
            "Tf32M128S2"
            if version in {"12.8", "13.0"} and record["cell"] == "hot_c"
            else "Tf32M64S2"
        )
        assert record["forced_tile"] == expected_old
        auto_graph = record["graphs"]["auto"]
        assert auto_graph["node_count"] == 1 and auto_graph["non_kernel_nodes"] == 0
        assert len(auto_graph["kernels"]) == 1
        auto_kernel = auto_graph["kernels"][0]
        assert auto_kernel["symbol"] == RNA_SYMBOL
        assert auto_kernel["block"] == [256, 1, 1]
        assert auto_kernel["shared_bytes"] == 98_304
        assert all(record[field] is True for field in (
            "raw_storage_bits_equal",
            "vendor_repeat_bits_equal",
            "auto_bits_equal",
            "repeat_bits_equal",
        ))
        assert record["graph_replay_bits_equal"] is (record["path"] == "graph")
        assert record["windows"] == 101
        for field in ("auto_samples_us", "forced_samples_us", "vendor_samples_us"):
            samples = record[field]
            assert len(samples) == 101
            assert all(math.isfinite(value) and value > 0 for value in samples)
        assert record["vendor_comparator"] == "CUBLAS_COMPUTE_32F_FAST_TF32"
        assert record["vendor_compute"] == "CUBLAS_COMPUTE_32F_FAST_TF32"
        assert record["reference_compute"] == "CUBLAS_COMPUTE_32F_PEDANTIC"
        assert record["vendor_bias_broadcast_timed"] is record["bias"]
        assert record["vendor_gemm_beta"] == (1 if record["bias"] else 0)
        vendor_symbols = {k["symbol"] for k in record["graphs"]["vendor"]["kernels"]}
        assert ("bias_broadcast" in vendor_symbols) is record["bias"]

        auto_old = [
            auto / old
            for auto, old in zip(record["auto_samples_us"], record["forced_samples_us"])
        ]
        auto_fast = [
            auto / fast
            for auto, fast in zip(record["auto_samples_us"], record["vendor_samples_us"])
        ]
        cohort = {
            "auto_old_p50": quantile(auto_old, 50),
            "auto_old_p95": quantile(auto_old, 95),
            "auto_fast_p50": quantile(auto_fast, 50),
            "auto_fast_p95": quantile(auto_fast, 95),
        }
        assert cohort["auto_old_p50"] < 1 and cohort["auto_old_p95"] < 1, (
            version,
            record["cell"],
            record["bias"],
            record["path"],
            record["order"],
            cohort,
        )
        ratios.setdefault((record["cell"], record["bias"]), []).append(cohort)

    cells = {}
    for (cell, bias), cohorts in sorted(ratios.items()):
        assert len(cohorts) == 4
        worst = {field: max(c[field] for c in cohorts) for field in cohorts[0]}
        worst["fast_win_all_cohorts"] = (
            worst["auto_fast_p50"] < 1 and worst["auto_fast_p95"] < 1
        )
        cells[f"{cell}:{int(bias)}"] = worst
    summary[version] = {
        "records": len(records),
        "unique_records": len(keys),
        "rejections": 0,
        "all_ten_internal_p50_p95_wins": True,
        "fast_wins": sum(v["fast_win_all_cohorts"] for v in cells.values()),
        "cells": cells,
    }

print(json.dumps(summary, indent=2, sort_keys=True))
