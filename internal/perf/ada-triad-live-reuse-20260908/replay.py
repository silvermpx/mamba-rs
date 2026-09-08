#!/usr/bin/env python3
"""Independent raw once7 replay for live Fixed-holder Triad discovery (no GPU)."""
import hashlib
import json
import math
from pathlib import Path

PERF = Path(__file__).resolve().parent.parent
STRATA = ("eager/ABBA", "eager/BAAB", "graph/ABBA", "graph/BAAB")
RUNS = (
    ("ada-scalar-nn-live-fixed-screen-20260908/evidence/repair1/once7-cuda132/test.log", "nn", 6, True),
    ("ada-scalar-nn-live-fixed-screen-20260908/evidence/cuda128/once7/test.log", "nn", 6, True),
    ("ada-scalar-nn-live-fixed-screen-20260908/evidence/cuda130/once7/test.log", "nn", 6, True),
    ("ada-scalar-nt-fixed-copyplan-screen-20260908/evidence/once7-cuda132/test.log", "nt", 1, True),
    ("ada-scalar-nt-fixed-copyplan-screen-20260908/evidence/cuda128/repair2/once7/test.log", "nt", 1, True),
    ("ada-scalar-nt-fixed-copyplan-screen-20260908/evidence/cuda130/repair1/once7/test.log", "nt", 1, True),
    ("ada-half-nn-fixed-s3-screen-20260908/evidence/once7-cuda132/test.log", "half", 2, False),
)


def close(left, right):
    assert math.isclose(left, right, rel_tol=0, abs_tol=2e-8), (left, right)


def key(row):
    return row["cell"], row.get("dtype", "F32"), row["comparator"]


def quantiles(samples, candidate_first):
    indices = (0, 3) if candidate_first else (1, 2)
    ratios = []
    for sample in samples:
        assert len(sample) == 4 and all(math.isfinite(x) and x > 0 for x in sample)
        candidate = sum(sample[i] for i in indices) / 2
        reference = sum(sample[i] for i in range(4) if i not in indices) / 2
        ratios.append(candidate / reference)
    assert len(ratios) == 7
    ratios.sort()
    return ratios[3], ratios[6]


def replay(relative, kind, count, passed):
    path = PERF / relative
    raw = path.read_bytes()
    text = raw.decode()
    rows = []
    for line in text.splitlines():
        if line.startswith("{"):
            rows.append(json.loads(line))
    prefixes = {"nn": "MambaBiLiveCopyPlan", "nt": "MambaBiScalarNtAdaDiscovery", "half": "MambaBiHalfNnTileAdaDiscovery"}
    prefix = prefixes[kind]
    screens = [r for r in rows if r["schema"].startswith(prefix + "Screen")]
    decisions = [r for r in rows if r["schema"].startswith(prefix + "Decision")]
    assert len(screens) == count * 4 and len(decisions) == count
    assert ("test result: ok. 1 passed; 0 failed;" in text) == passed
    assert passed or "test result: FAILED. 0 passed; 1 failed;" in text
    grouped = {}
    for row in screens:
        assert row["windows"] == 7 and row["order"] in ("ABBA", "BAAB")
        if kind == "nn":
            candidate_first = row["candidate_first"]
            assert candidate_first is (row["order"] == "ABBA")
            assert row["logical_gemms_per_observation"] == 1
            assert row["reseed_scope"] == "C+A+B" and row["warmup_brackets"] == 8
            assert row["candidate_module"] == "Fixed"
            assert row["comparator"] in ("actual_auto", "cublas_fast_tf32")
            assert row["ratio_direction"] == "candidate_over_reference"
        elif kind == "nt":
            candidate_first = row["order"] == "BAAB"
            assert row["candidate_nodes_per_observation"] == 2
            assert row["auto_nodes_per_observation"] == 1
            assert row["candidate_modules"] == ["TriadScalar", "Fixed"]
            assert row["comparator"] == "actual_auto"
            assert row["ratio_direction"] == "candidate_over_auto"
        else:
            candidate_first = row["order"] == "ABBA"
            assert row["logical_gemms_per_observation"] == 1
            assert row["candidate"] == "fixed_sm89_tc128_s3"
            assert row["comparator"] == "forced_tc128"
            assert row["ratio_direction"] == "candidate_over_forced_tc128"
        pair = quantiles(row["raw_observations_us"], candidate_first)
        close(pair[0], row["ratio_p50"])
        close(pair[1], row["ratio_p95"])
        strata = grouped.setdefault(key(row), {})
        lane = row["path"] + "/" + row["order"]
        assert lane not in strata
        strata[lane] = pair
    seen = set()
    for decision in decisions:
        cell = key(decision)
        assert cell not in seen
        seen.add(cell)
        strata = grouped[cell]
        assert set(strata) == set(STRATA)
        retain = all(0 < x < .99 for pair in strata.values() for x in pair)
        assert decision["retain"] is retain and decision["promotion"] is False
        for lane, stored in zip(STRATA, decision["strata"], strict=True):
            for actual, claimed in zip(strata[lane], stored, strict=True):
                close(actual, claimed)
        p50s = [p[0] for p in strata.values()]
        p95 = max(p[1] for p in strata.values())
        print(f"{cell}: {'ADVANCE' if retain else 'STOP'} p50 {min(p50s):.6f}–{max(p50s):.6f}; worst p95 {p95:.6f}")
    print(hashlib.sha256(raw).hexdigest(), relative)
    return len(screens) * 7


if __name__ == "__main__":
    # Opposite order encodings really must reverse the arms, not the ratio label.
    assert quantiles([[2., 4., 4., 2.]] * 7, True) == (.5, .5)
    assert quantiles([[2., 4., 4., 2.]] * 7, False) == (2., 2.)
    brackets = sum(replay(*run) for run in RUNS)
    print(f"PASS: {brackets} brackets / {4 * brackets} timed observations independently replayed")
