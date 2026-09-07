#!/usr/bin/env python3
"""Build/list the frozen Task3 resource and paired-timing targets on CUDA 13.2."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time


ENV_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
ROOT = Path("/root/mamba-ada-exact-toolkit-auto-20260907")
EVIDENCE = Path(
    "/root/evidence-ada-triad-finalist-integration-20260907/task3-build-cuda132-final"
)
GENERATION = "triad-nt-padded36-two-arm1"
HEAD = "7a79bbcd"
EXPECTED = {
    "src/mamba_ssm/gpu/gemm_bi_triad/qualification.rs": "0e3f222ea55e9888e2e712a01d02e4060f03de457d4df1948c58dcce47ae5f74",
    "tests/gemm_bi_performance_matrix.rs": "bbbf459341073ac3e074889ecce1e89eafa7247b84ec7fa20dd2de19604938ce",
    "tests/gemm_bi_tf32_cohort_binding.rs": "9d265cb1640895c60eed3674a9be89e93c198058f4754336c0a28a371cd07fad",
    "tests/support/fixed_full_mantissa.rs": "234d4d87dd479780a3cacee34e5f7f1e868e0465536caba3866a46a16a39cb22",
}
EXPECTED_CACHE = {
    "mamba-kernels-v1-3809af034718aa08dfdfc1b6baeb407715436dc7db87ef39825b033c2fd72541.bin": "5851ebd49595a0b2ef665073289e52f903eec3266deadfc5ad5d9f2096ce6d16",
    "mamba-kernels-v1-822ebf0dda7dcf9c4f5c5a2213884dde42ab45f5cf238479a81856042c302b7c.bin": "5035921fc8e078e5454c9c5392f9e9ee6ca6a3de54c1e08cb54bba432fa37546",
    "mamba-kernels-v1-e4f76515e441adc28862775000fce218e2d2cca61b47ce484e15eb1144c40fb3.bin": "f0be052fc160b9e5f4b6a897878781327968c0015d551128415a94b85e6fc0aa",
    "mamba-kernels-v1-e61681f002f8ecdd61677fc89a8f0faeac5ce494db503fcc97fd344633ec2d65.bin": "1da42f2fbc44dfd9093959831296d86de2414a6d9bb042bc1640730c68d36ac5",
}
EXPECTED_TESTS = {
    "mamba_rs": "mamba_ssm::gpu::gemm_bi_triad::qualification::tests::sm89_nt_compact_finalist_resources_k0_and_live_revisions",
    "gemm_bi_performance_matrix": "sm89_nt_finalist_once21::gemm_bi_sm89_nt_finalist_current_fast_once21",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def cache_hashes(cache: Path) -> dict[str, str]:
    return {path.name: sha(path) for path in sorted(cache.iterdir()) if path.is_file()}


def source_manifest() -> dict[str, object]:
    paths = [ROOT / "Cargo.toml", ROOT / "Cargo.lock"]
    for directory in ("src", "kernels", "tests"):
        paths.extend(
            path
            for path in (ROOT / directory).rglob("*")
            if path.is_file()
            and not path.name.startswith("._")
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        )
    sources = {
        str(path.relative_to(ROOT)): sha(path)
        for path in sorted(set(paths))
    }
    mismatches = {
        relative: {"expected": expected, "actual": sources.get(relative)}
        for relative, expected in EXPECTED.items()
        if sources.get(relative) != expected
    }
    if mismatches:
        raise RuntimeError(f"frozen source mismatch: {mismatches}")
    return {
        "schema": "MambaTriadAdaFinalistTask3BuildSourcesV1",
        "head": HEAD,
        "count": len(sources),
        "excluded_local_wip_not_synced": [
            "tests/gemm_bi_sm120_tf32_selector_qualification.rs",
            "tests/support/triad_discovery_samples.rs",
        ],
        "expected_frozen": EXPECTED,
        "sources": sources,
    }


def telemetry(phase: str, environment: dict[str, str]) -> dict[str, object]:
    value: dict[str, object] = {
        "phase": phase,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment_constructor": str(ENV_RUNNER) + "::env_for",
    }
    for kind, query in (
        ("gpu", "--query-gpu=uuid,name,compute_cap,utilization.gpu,utilization.memory"),
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
        and fields[3:] == ["0 %", "0 %"]
        and not str(value["apps"]).strip()
    )


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("env_runner", ENV_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load environment constructor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = module.env_for("13.2", GENERATION)
    environment = context["env"]
    assert isinstance(environment, dict)
    cache = Path(context["cache"])
    initial_cache = cache_hashes(cache)
    if stat.S_IMODE(cache.stat().st_mode) != 0o700 or initial_cache != EXPECTED_CACHE:
        raise RuntimeError("Task3 cache is not the bound private four-artifact cache")
    manifest = source_manifest()
    write_json(EVIDENCE / "source-manifest.json", manifest)
    pre = telemetry("PRE", environment)
    write_json(EVIDENCE / "pre.json", pre)
    if not quiet(pre):
        raise RuntimeError(f"build PRE is not quiet/no-apps: {pre}")
    args = [
        "cargo",
        "test",
        "--release",
        "--features",
        str(context["feature"]),
        "--no-run",
        "--lib",
        "--test",
        "gemm_bi_performance_matrix",
        "--message-format=json-render-diagnostics",
    ]
    with (EVIDENCE / "build.log").open("x") as output:
        build = subprocess.run(
            args,
            cwd=ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    artifacts: dict[str, Path] = {}
    for line in (EVIDENCE / "build.log").read_text().splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = message.get("target", {}).get("name")
        if (
            message.get("reason") == "compiler-artifact"
            and name in EXPECTED_TESTS
            and message.get("executable")
        ):
            artifacts[name] = Path(message["executable"])
    lists = {}
    if build.returncode == 0:
        for name, binary in sorted(artifacts.items()):
            result = subprocess.run(
                [str(binary), "--list"],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            text = result.stdout + result.stderr
            (EVIDENCE / f"list-{name}.log").write_text(text)
            names = {
                line.removesuffix(": test")
                for line in text.splitlines()
                if line.endswith(": test")
            }
            lists[name] = {
                "exit": result.returncode,
                "count": len(names),
                "expected": EXPECTED_TESTS[name],
                "expected_listed": EXPECTED_TESTS[name] in names,
                "binary": str(binary),
                "binary_sha256": sha(binary),
            }
    release = telemetry("RELEASE", environment)
    write_json(EVIDENCE / "release.json", release)
    final_cache = cache_hashes(cache)
    receipt = {
        "schema": "MambaTriadAdaFinalistTask3BuildReceiptV1",
        "head": HEAD,
        "toolkit": context["toolkit"],
        "feature": context["feature"],
        "generation": GENERATION,
        "target": str(context["target"]),
        "cache": str(cache),
        "cache_mode": oct(stat.S_IMODE(cache.stat().st_mode)),
        "cache_reused_not_cold": True,
        "cache_before": initial_cache,
        "cache_after": final_cache,
        "cache_unchanged": initial_cache == final_cache == EXPECTED_CACHE,
        "args": args,
        "exit": build.returncode,
        "source_manifest_sha256": sha(EVIDENCE / "source-manifest.json"),
        "pre_utc": pre["utc"],
        "release_utc": release["utc"],
        "artifacts": lists,
    }
    write_json(EVIDENCE / "command.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    print("WRAPPER_COMPLETE")
    ok = (
        build.returncode == 0
        and set(lists) == set(EXPECTED_TESTS)
        and all(value["exit"] == 0 and value["expected_listed"] for value in lists.values())
        and receipt["cache_unchanged"]
    )
    return 0 if ok else 97


if __name__ == "__main__":
    sys.exit(main())
