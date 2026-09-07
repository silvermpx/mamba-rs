#!/usr/bin/env python3
"""Build and run the frozen CUDA 13.2 Task3 actual-AUTO closure."""

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
import time

ROOT = Path("/root/mamba-ada-exact-toolkit-auto-20260907")
ENV_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
ONCE21_RUNNER = Path(
    "/root/evidence-ada-triad-finalist-integration-20260907/task3-once21-cuda132/run.py"
)
EVIDENCE = Path(
    "/root/evidence-ada-triad-finalist-integration-20260907/task3-auto-cuda132-final"
)
GENERATION = "triad-nt-padded36-two-arm1"
EXPECTED = {
    "src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs":
        "7f32111aeda43cfd069c32ccbf1c77b31bb598db805f422d02bc3b295fb2e550",
    "tests/gemm_bi_tf32_cohort_binding.rs":
        "9a54bcb0d569db72176895577a4dc54f27017ab20be5a0f9085dd02d1c98323d",
    "tests/gemm_bi_performance_matrix.rs":
        "3a2dc3aaba9b60ea4f2295260ae41782356eaece48acc8424d44f06a53c43b2d",
    "src/mamba_ssm/gpu/gemm_bi_triad/qualification.rs":
        "0e3f222ea55e9888e2e712a01d02e4060f03de457d4df1948c58dcce47ae5f74",
}
EXPECTED_CACHE = {
    "mamba-kernels-v1-3809af034718aa08dfdfc1b6baeb407715436dc7db87ef39825b033c2fd72541.bin": "5851ebd49595a0b2ef665073289e52f903eec3266deadfc5ad5d9f2096ce6d16",
    "mamba-kernels-v1-822ebf0dda7dcf9c4f5c5a2213884dde42ab45f5cf238479a81856042c302b7c.bin": "5035921fc8e078e5454c9c5392f9e9ee6ca6a3de54c1e08cb54bba432fa37546",
    "mamba-kernels-v1-e4f76515e441adc28862775000fce218e2d2cca61b47ce484e15eb1144c40fb3.bin": "f0be052fc160b9e5f4b6a897878781327968c0015d551128415a94b85e6fc0aa",
    "mamba-kernels-v1-e61681f002f8ecdd61677fc89a8f0faeac5ce494db503fcc97fd344633ec2d65.bin": "1da42f2fbc44dfd9093959831296d86de2414a6d9bb042bc1640730c68d36ac5",
}
TESTS = [
    ("lib", "mamba_ssm::gpu::gemm_bi_triad::dispatch::tests::sm89_finalist_measured_cohorts_select_and_decline_to_prior_routes", False),
    ("lib", "mamba_ssm::gpu::gemm_bi_triad::dispatch::tests::sm89_finalist_is_forced_only_and_uses_only_its_own_pointer_binding", False),
    ("cohort", "sm89_finalist_admitted_cell_filter_is_strict", False),
    ("cohort", "sm89_nt_compact_finalist_actual_auto_symbols_graphs_and_bits", True),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def hashes(path: Path) -> dict[str, str]:
    return {item.name: sha(item) for item in sorted(path.iterdir()) if item.is_file()}


def source_manifest() -> dict[str, object]:
    paths = [ROOT / "Cargo.toml", ROOT / "Cargo.lock"]
    for directory in ("src", "kernels", "tests"):
        paths.extend(
            item for item in (ROOT / directory).rglob("*")
            if item.is_file() and not item.name.startswith("._")
            and "__pycache__" not in item.parts and item.suffix != ".pyc"
        )
    sources = {str(item.relative_to(ROOT)): sha(item) for item in sorted(set(paths))}
    mismatch = {
        relative: {"expected": expected, "actual": sources.get(relative)}
        for relative, expected in EXPECTED.items() if sources.get(relative) != expected
    }
    if mismatch:
        raise RuntimeError(f"frozen source mismatch: {mismatch}")
    return {"schema": "MambaTriadAdaFinalistTask3AutoSourcesV1", "count": len(sources), "expected_frozen": EXPECTED, "sources": sources}


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("env_runner", ENV_RUNNER)
    assert spec and spec.loader
    env_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(env_module)
    spec2 = importlib.util.spec_from_file_location("once21", ONCE21_RUNNER)
    assert spec2 and spec2.loader
    once21 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(once21)
    context = env_module.env_for("13.2", GENERATION)
    environment = context["env"]
    cache = Path(context["cache"])
    if stat.S_IMODE(cache.stat().st_mode) != 0o700 or hashes(cache) != EXPECTED_CACHE:
        raise RuntimeError("private cache binding changed")
    write_json(EVIDENCE / "source-manifest.json", source_manifest())
    pre = once21.telemetry("BUILD_PRE", environment)
    write_json(EVIDENCE / "build-pre.json", pre)
    if not once21.quiet(pre):
        raise RuntimeError("build PRE is not quiet")
    args = ["cargo", "test", "--release", "--features", str(context["feature"]), "--no-run", "--lib", "--test", "gemm_bi_tf32_cohort_binding", "--message-format=json-render-diagnostics"]
    with (EVIDENCE / "build.log").open("x") as output:
        build = subprocess.run(args, cwd=ROOT, env=environment, stdout=output, stderr=subprocess.STDOUT, check=False)
    artifacts: dict[str, Path] = {}
    for line in (EVIDENCE / "build.log").read_text().splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = message.get("target", {}).get("name")
        if message.get("reason") == "compiler-artifact" and name in ("mamba_rs", "gemm_bi_tf32_cohort_binding") and message.get("executable"):
            artifacts["lib" if name == "mamba_rs" else "cohort"] = Path(message["executable"])
    if build.returncode != 0 or set(artifacts) != {"lib", "cohort"}:
        raise RuntimeError(f"build closure failed: {build.returncode}/{artifacts}")
    lists = {}
    for label, binary in artifacts.items():
        listed = subprocess.run([str(binary), "--list"], cwd=ROOT, env=environment, text=True, capture_output=True, check=False)
        (EVIDENCE / f"list-{label}.log").write_text(listed.stdout + listed.stderr)
        names = {line.removesuffix(": test") for line in listed.stdout.splitlines() if line.endswith(": test")}
        expected = {test for target, test, _ in TESTS if target == label}
        if listed.returncode != 0 or not expected.issubset(names):
            raise RuntimeError(f"{label} exact test list changed")
        lists[label] = {"binary": str(binary), "binary_sha256": sha(binary), "count": len(names), "expected": sorted(expected)}
    results = []
    for index, (target, test, ignored) in enumerate(TESTS, 1):
        run_pre = once21.telemetry("PRE", environment)
        write_json(EVIDENCE / f"test-{index}-pre.json", run_pre)
        if not once21.quiet(run_pre):
            raise RuntimeError(f"test {index} PRE is not quiet")
        run_args = [str(artifacts[target]), test, "--exact"] + (["--ignored", "--nocapture"] if ignored else ["--nocapture"])
        with (EVIDENCE / f"test-{index}.log").open("x") as output:
            run = subprocess.run(run_args, cwd=ROOT, env=environment, stdout=output, stderr=subprocess.STDOUT, check=False)
        text = (EVIDENCE / f"test-{index}.log").read_text()
        executed_one = "1 passed" in text and "0 passed" not in text
        release = once21.telemetry("RELEASE", environment)
        write_json(EVIDENCE / f"test-{index}-release.json", release)
        if run.returncode != 0 or not executed_one:
            raise RuntimeError(f"test {index} failed: exit={run.returncode}, executed_one={executed_one}")
        results.append({"target": target, "test": test, "ignored": ignored, "args": run_args, "exit": run.returncode, "executed_one": executed_one})
    drain_samples = []
    consecutive = 0
    for _ in range(120):
        sample = once21.telemetry("DRAIN", environment)
        drain_samples.append(sample)
        consecutive = consecutive + 1 if once21.quiet(sample) else 0
        if consecutive >= 5:
            break
        time.sleep(1)
    write_json(EVIDENCE / "drain.json", {"schema": "MambaTriadAdaFinalistDrainV1", "required_consecutive_quiet": 5, "samples": drain_samples, "complete": consecutive >= 5})
    if consecutive < 5 or hashes(cache) != EXPECTED_CACHE:
        raise RuntimeError("drain/cache closure failed")
    receipt = {"schema": "MambaTriadAdaFinalistTask3AutoReceiptV1", "toolkit": "13.2", "feature": context["feature"], "generation": GENERATION, "build_args": args, "build_exit": build.returncode, "source_manifest_sha256": sha(EVIDENCE / "source-manifest.json"), "artifacts": lists, "runs": results, "cache_artifacts": hashes(cache), "drain_complete": True, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(EVIDENCE / "command.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    print("WRAPPER_COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
