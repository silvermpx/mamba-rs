#!/usr/bin/env python3
"""Frozen Task2 CUDA 13.2 compile/list and host-only checkpoint."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time


TASK8_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
ROOT = Path("/root/mamba-ada-exact-toolkit-auto-20260907")
EVIDENCE = Path(
    "/root/evidence-ada-triad-finalist-integration-20260907/"
    "task2-build-cuda132-checkpoint1"
)
GENERATION = "triad-nt-padded36-two-arm1"
HEAD = "04c71ebe44e1788b2ec1b9ff6a3c032e21c66cb4"
EXPECTED = {
    "src/mamba_ssm/gpu/context.rs": "95dd4e04b8ac1706e1e320d3dd1d8963a0691c0a6ba99632ee0b169a514e2e8b",
    "src/mamba_ssm/gpu/kernel_identity.rs": "dac81ac78b6909924d34813b88bd9d0604f0f5404304d057d63fa109f45809a9",
    "src/mamba_ssm/gpu/kernels.rs": "0fce369993afab12839838faf153549bc2aba469b689fa6335ff021ea00d6a92",
    "src/mamba_ssm/gpu/gemm_bi_triad/mod.rs": "189a68c27fa0571e250d0502c7e7188740f71ac0847bc2ebd99298070ad90839",
    "src/mamba_ssm/gpu/gemm_bi_triad/contract.rs": "b5bdd1b8b9199ea712256746cea9037f48468b32c3bf371b9d184ba41585d57a",
    "src/mamba_ssm/gpu/gemm_bi_triad/modules.rs": "03a2aadf0c3728a832cc3f7966c96bb7ce85dc915ef38d9c4279f40f046e25da",
    "src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs": "4b3c4a375fdd60cfaf987f3fa5cb0a270fde29609afaec1159b5b0105b8e7f5a",
    "src/mamba_ssm/gpu/gemm_bi_triad/launch.rs": "3f1938709071f3ed015ff48fd6d92e26f582c733c4532114a174f25ad7587746",
    "src/mamba_ssm/gpu/gemm_bi_triad/qualification.rs": "b5f663f7ab8a96aaa16acf91050462892f6763139eaa69cd9ac8c6c4fd4b08d6",
    "tests/gemm_bi_tf32_selector.rs": "05cfb2b06f33998ac12f60f50df8032b7bee1a17ded2b08990586a611dd03254",
    "tests/gemm_bi_performance_matrix.rs": "9689abcc399d82984628db9e834ebaa0552faee7fadbce8cf3ed4903701d745c",
    "src/mamba_ssm/gpu/gemm_bi_triad/sm89_finalist_source.rs": "2a51afde2fd1e4203e63acfc2e10c9072e814d075039cbb8baecc19beba5b029",
    "kernels/gemm_bi_triad/sm89_nt_compact.cuh": "dd2a8494666547e797defcfcab3e52a774729abbfe62e630fd0df896391c21e4",
}
EXPECTED_CACHE = {
    "mamba-kernels-v1-3809af034718aa08dfdfc1b6baeb407715436dc7db87ef39825b033c2fd72541.bin": "5851ebd49595a0b2ef665073289e52f903eec3266deadfc5ad5d9f2096ce6d16",
    "mamba-kernels-v1-822ebf0dda7dcf9c4f5c5a2213884dde42ab45f5cf238479a81856042c302b7c.bin": "5035921fc8e078e5454c9c5392f9e9ee6ca6a3de54c1e08cb54bba432fa37546",
    "mamba-kernels-v1-e4f76515e441adc28862775000fce218e2d2cca61b47ce484e15eb1144c40fb3.bin": "f0be052fc160b9e5f4b6a897878781327968c0015d551128415a94b85e6fc0aa",
}
TARGETS = (
    "gemm_bi_tf32_cohort_binding",
    "gemm_bi_performance_matrix",
    "gemm_bi_tf32_selector",
)
HOST_TEST_SUFFIXES = (
    "sm89_finalist_is_forced_only_and_uses_only_its_own_pointer_binding",
    "ada_finalist_is_an_optional_fourth_artifact_without_changing_three_module_identity",
    "sm89_finalist_inventory_replaces_one_legacy_entry_and_rejects_foreign_families",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def cache_hashes(cache: Path) -> dict[str, str]:
    return {path.name: sha(path) for path in sorted(cache.iterdir()) if path.is_file()}


def telemetry(phase: str, context: dict[str, object]) -> dict[str, object]:
    environment = context["env"]
    assert isinstance(environment, dict)
    value: dict[str, object] = {
        "phase": phase,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment_constructor": str(TASK8_RUNNER) + "::env_for",
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
    write_json(EVIDENCE / (phase.lower() + ".json"), value)
    fields = [field.strip() for field in str(value["gpu"]).strip().split(",")]
    if (
        value["gpu_exit"] != 0
        or value["apps_exit"] != 0
        or fields[:3]
        != [
            "GPU-d1edd7be-e88d-aed6-047d-622163306f0e",
            "NVIDIA RTX 6000 Ada Generation",
            "8.9",
        ]
        or fields[3:] != ["0 %", "0 %"]
        or str(value["apps"]).strip()
    ):
        raise RuntimeError(f"{phase} is not strict quiet/no-apps: {value}")
    return value


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
        "schema": "MambaTriadAdaFinalistTask2BuildSourcesV1",
        "head": HEAD,
        "count": len(sources),
        "excluded_local_wip_not_synced": [
            "tests/gemm_bi_sm120_tf32_selector_qualification.rs",
            "tests/support/triad_discovery_samples.rs",
        ],
        "expected_frozen": EXPECTED,
        "sources": sources,
    }


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("task8_run", TASK8_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load environment constructor")
    task8_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task8_run)
    context = task8_run.env_for("13.2", GENERATION)
    environment = context["env"]
    assert isinstance(environment, dict)
    cache = Path(context["cache"])
    if stat.S_IMODE(cache.stat().st_mode) != 0o700:
        raise RuntimeError("cache is not private 0700")
    cache_before = cache_hashes(cache)
    if cache_before != EXPECTED_CACHE:
        raise RuntimeError(f"production cache changed before build: {cache_before}")
    manifest = source_manifest()
    write_json(EVIDENCE / "source-manifest.json", manifest)
    pre = telemetry("PRE", context)
    args = [
        "cargo",
        "test",
        "--release",
        "--features",
        str(context["feature"]),
        "--no-run",
        "--lib",
    ]
    for target in TARGETS:
        args.extend(("--test", target))
    args.append("--message-format=json-render-diagnostics")
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with (EVIDENCE / "build.log").open("x") as output:
        build = subprocess.run(
            args,
            cwd=ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    artifacts: dict[str, str] = {}
    for line in (EVIDENCE / "build.log").read_text().splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("reason") != "compiler-artifact" or not message.get("executable"):
            continue
        name = message.get("target", {}).get("name")
        if name == "mamba_rs" or name in TARGETS:
            artifacts[str(name)] = str(message["executable"])
    lists: dict[str, dict[str, object]] = {}
    names_by_target: dict[str, set[str]] = {}
    if build.returncode == 0:
        for target, executable in sorted(artifacts.items()):
            listed = subprocess.run(
                [executable, "--list"],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            text = listed.stdout + listed.stderr
            (EVIDENCE / f"list-{target}.log").write_text(text)
            names = {
                line.removesuffix(": test")
                for line in text.splitlines()
                if line.endswith(": test")
            }
            names_by_target[target] = names
            lists[target] = {
                "exit": listed.returncode,
                "count": len(names),
                "binary": executable,
                "binary_sha256": sha(Path(executable)),
            }
    lib_names = names_by_target.get("mamba_rs", set())
    selected: dict[str, str] = {}
    for suffix in HOST_TEST_SUFFIXES:
        matches = [name for name in lib_names if name.endswith("::" + suffix) or name == suffix]
        if len(matches) != 1:
            raise RuntimeError(f"host-only test suffix {suffix} matched {matches}")
        selected[suffix] = matches[0]
    host_results: dict[str, dict[str, object]] = {}
    if build.returncode == 0:
        lib_binary = artifacts["mamba_rs"]
        for index, (suffix, full_name) in enumerate(selected.items(), start=1):
            result = subprocess.run(
                [lib_binary, full_name, "--exact", "--nocapture"],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            text = result.stdout + result.stderr
            (EVIDENCE / f"host-{index}-{suffix}.log").write_text(text)
            executed_one = "1 passed" in text and "0 passed" not in text
            host_results[suffix] = {
                "full_name": full_name,
                "exit": result.returncode,
                "executed_one": executed_one,
            }
    cache_after = cache_hashes(cache)
    release = telemetry("RELEASE", context)
    receipt = {
        "schema": "MambaTriadAdaFinalistTask2BuildHostReceiptV1",
        "head": HEAD,
        "toolkit": context["toolkit"],
        "feature": context["feature"],
        "generation": GENERATION,
        "target": str(context["target"]),
        "cache": str(cache),
        "cache_mode": oct(stat.S_IMODE(cache.stat().st_mode)),
        "cache_stage": "rust_build_list_host_only_no_nvrtc_load",
        "cache_before": cache_before,
        "cache_after": cache_after,
        "cache_unchanged": cache_before == cache_after == EXPECTED_CACHE,
        "args": args,
        "started_utc": started,
        "exit": build.returncode,
        "pre_utc": pre["utc"],
        "release_utc": release["utc"],
        "source_manifest_sha256": sha(EVIDENCE / "source-manifest.json"),
        "artifacts": lists,
        "host_tests": host_results,
    }
    write_json(EVIDENCE / "command.json", receipt)
    ok = (
        build.returncode == 0
        and set(artifacts) == {"mamba_rs", *TARGETS}
        and all(value["exit"] == 0 for value in lists.values())
        and all(
            value["exit"] == 0 and value["executed_one"]
            for value in host_results.values()
        )
        and len(host_results) == len(HOST_TEST_SUFFIXES)
        and receipt["cache_unchanged"]
    )
    print(json.dumps(receipt, sort_keys=True))
    print("WRAPPER_COMPLETE")
    return 0 if ok else 97


if __name__ == "__main__":
    sys.exit(main())
