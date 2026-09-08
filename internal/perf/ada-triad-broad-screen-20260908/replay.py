#!/usr/bin/env python3
"""Recompute recorded scalar discovery quantiles; never runs GPU work."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS = [
    ("ada-scalar-nn-m64-screen-20260908/evidence/once7-cuda132/test.log", 2, False),
    ("ada-scalar-nn-m64-screen-20260908/evidence/prism-repair/once7-cuda132/test.log", 1, True),
    ("ada-scalar-nt-d768-out-screen-20260908/evidence/graph-warmup-repair/once7-cuda132/test.log", 1, True),
    ("ada-scalar-nn-fixed-copyplan-overwrite-screen-20260908/evidence/once7-cuda132/test.log", 3, True),
]


def close(left, right):
    assert abs(left - right) < 2e-8, (left, right)


def replay(relative, expected_cells, passed):
    path = ROOT / "internal/perf" / relative
    raw = path.read_bytes()
    text = raw.decode()
    rows = [json.loads(line) for line in text.splitlines() if line.startswith('{"schema":')]
    screens = [row for row in rows if row["schema"].endswith("DiscoveryScreenV1")]
    decisions = [row for row in rows if row["schema"].endswith("DiscoveryDecisionV1")]
    assert len(decisions) == expected_cells and len(screens) == expected_cells * 4
    assert ("test result: ok. 1 passed; 0 failed;" in text) == passed
    assert passed or "red zone changed at guard element 32" in text
    grouped = {}
    for row in screens:
        nn = row["op"] == "NN"
        assert row["comparator"] == ("generic_exact" if nn else "actual_auto")
        assert row["ratio_direction"] == ("candidate_over_generic_exact" if nn else "candidate_over_auto")
        assert row["windows"] == 7
        samples = row["observations_us"] if nn else row["raw_observations_us"]
        assert len(samples) == 7
        if nn:
            assert row["reseed_position"] == "before_start_event"
            assert row["post_download_before_next_reset"] is True
            assert row["logical_gemms_per_observation"] == 1
            arms = row["observation_arms"]
            assert arms == (["production_m64n64", "production_generic", "production_generic", "production_m64n64"] if row["order"] == "ABBA" else ["production_generic", "production_m64n64", "production_m64n64", "production_generic"])
            candidate_indices = [i for i, arm in enumerate(arms) if arm == "production_m64n64"]
        else:
            assert row["candidate_nodes_per_observation"] == 2
            assert row["auto_nodes_per_observation"] == 1
            candidate_indices = [1, 2] if row["order"] == "ABBA" else [0, 3]
        ratios = []
        for index, sample in enumerate(samples):
            assert len(sample) == 4 and all(math.isfinite(x) and x > 0 for x in sample)
            candidate = sum(sample[i] for i in candidate_indices) / 2
            reference = sum(sample[i] for i in range(4) if i not in candidate_indices) / 2
            ratios.append(candidate / reference)
            if nn:
                close(candidate, row["candidate_samples_us"][index])
                close(reference, row["generic_samples_us"][index])
                close(ratios[-1], row["ratios"][index])
        ratios.sort()
        pair = (ratios[3], ratios[6])
        close(pair[0], row["ratio_p50"])
        close(pair[1], row["ratio_p95"])
        strata = grouped.setdefault(row["cell"], {})
        key = row["path"] + "/" + row["order"]
        assert key not in strata
        strata[key] = pair
    for decision in decisions:
        strata = grouped[decision["cell"]]
        assert set(strata) == {"eager/ABBA", "eager/BAAB", "graph/ABBA", "graph/BAAB"}
        retained = all(0 < value < .99 for pair in strata.values() for value in pair)
        assert decision["retain"] is retained and decision["promotion"] is False
        assert decision["decision"] == ("advance_to_full_qualification" if retained else "stop_no_retry")
        for key, stored in zip(decision["strata_order"], decision["strata"], strict=True):
            for expected, actual in zip(strata[key], stored, strict=True):
                close(expected, actual)
        print(decision["op"], decision["cell"], "ADVANCE" if retained else "STOP", strata)
    print(hashlib.sha256(raw).hexdigest(), relative)
    return len(screens) * 7


if __name__ == "__main__":
    brackets = sum(replay(*run) for run in RUNS)
    print(f"PASS: {brackets} brackets / {brackets * 4} timed observations independently replayed")
