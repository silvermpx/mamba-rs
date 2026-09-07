#!/usr/bin/env python3
"""Validate one focused finalist forced-rung 21-window transcript."""

import argparse
import json
from pathlib import Path


def records(path: Path) -> list[dict]:
    found = []
    for line in path.read_text().splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            value = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and str(value.get("schema", "")).startswith("MambaBiFixedExplicitForcedRung"):
            found.append(value)
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--row", required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--tile", required=True)
    args = parser.parse_args()

    found = records(args.log)
    timed = [r for r in found if r["schema"] == "MambaBiFixedExplicitForcedRungV2"]
    rejected = [r for r in found if r["schema"] == "MambaBiFixedExplicitForcedRungRejectedV2"]
    complete = [r for r in found if r["schema"] == "MambaBiFixedExplicitForcedRungCompleteV2"]
    expected = {
        (path, order)
        for path in ("eager", "graph")
        for order in ("auto_forced_vendor", "vendor_forced_auto")
    }
    actual = {(r["path"], r["order"]) for r in timed}
    assert len(timed) == 4 and actual == expected, (len(timed), actual)
    assert not rejected, rejected
    assert len(complete) == 1 and complete[0]["records"] == 4 and complete[0]["rejected"] == 0
    for record in timed:
        assert (record["row"], record["cell"], record["bias"], record["forced_tile"]) == (
            args.row,
            args.cell,
            False,
            args.tile,
        )
        assert record["windows"] == 21
        assert len(record["auto_samples_us"]) == 21
        assert len(record["forced_samples_us"]) == 21
        assert len(record["vendor_samples_us"]) == 21
        assert record["raw_storage_bits_equal"] is True
        assert record["repeat_bits_equal"] is True
    identity_fields = [
        "fixed_source_digest",
        "fixed_invocation_digest",
        "fixed_artifact_digest",
        "header_manifest_digest",
        "nvrtc_library_domain",
    ]
    identities = {tuple(r[field] for field in identity_fields) for r in timed}
    assert len(identities) == 1, identities
    worst_p50 = max(r["forced_over_auto_p50"] for r in timed)
    worst_p95 = max(r["forced_over_auto_p95"] for r in timed)
    summary = {
        "schema": "MambaBiFixedFinalistForced21AnalysisV1",
        "row": args.row,
        "cell": args.cell,
        "tile": args.tile,
        "records": len(timed),
        "rejected": len(rejected),
        "worst_forced_over_auto_p50": worst_p50,
        "worst_forced_over_auto_p95": worst_p95,
        "admitted": worst_p50 <= 0.985 and worst_p95 <= 1.0,
        "strata": [
            {
                "path": r["path"],
                "order": r["order"],
                "forced_over_auto_p50": r["forced_over_auto_p50"],
                "forced_over_auto_p95": r["forced_over_auto_p95"],
                "forced_over_fast_p50": r["forced_over_vendor_p50"],
                "forced_over_fast_p95": r["forced_over_vendor_p95"],
            }
            for r in sorted(timed, key=lambda r: (r["path"], r["order"]))
        ],
        "identity": dict(zip(identity_fields, next(iter(identities)))),
    }
    args.output.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
