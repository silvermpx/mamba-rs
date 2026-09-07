#!/usr/bin/env python3
"""Build/list and run the one authorized early SM89 finalist bits smoke."""

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
    "task3-early-bits-cuda132"
)
GENERATION = "triad-nt-padded36-two-arm1"
HEAD = "c1faa72476830ca725aca681ff1bad4bab4513f6"
TEST_SUFFIX = "sm89_nt_compact_finalist_forced_matches_portable_rna_bits"
EXPECTED = {
    "tests/gemm_bi_tf32_cohort_binding.rs": "9d265cb1640895c60eed3674a9be89e93c198058f4754336c0a28a371cd07fad",
    "tests/support/fixed_full_mantissa.rs": "234d4d87dd479780a3cacee34e5f7f1e868e0465536caba3866a46a16a39cb22",
    "src/mamba_ssm/gpu/context.rs": "24c33c48a07d243c09bed0d615461c32d852f7db418f187eb25d043f200505ea",
    "src/mamba_ssm/gpu/kernel_identity.rs": "c7018df211b815475a1b8024d80aa6dfdbc30a74020677325c10afade60d50ec",
    "src/mamba_ssm/gpu/kernels.rs": "0fce369993afab12839838faf153549bc2aba469b689fa6335ff021ea00d6a92",
    "src/mamba_ssm/gpu/gemm_bi_triad/mod.rs": "189a68c27fa0571e250d0502c7e7188740f71ac0847bc2ebd99298070ad90839",
    "src/mamba_ssm/gpu/gemm_bi_triad/contract.rs": "7ee9ea133f01a0a01ee8404f29a178c008034b51e5281a73744e04c4cddc2d5f",
    "src/mamba_ssm/gpu/gemm_bi_triad/modules.rs": "c7e6de79d664afd69bf448cfd7b27f7e02977df0b95187cd77c8460cce15f8d6",
    "src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs": "fbcae2dd621348748d1e35af6fd30c9fbeaf13309e9b3fa0d87fde8ac39f2951",
    "src/mamba_ssm/gpu/gemm_bi_triad/launch.rs": "3f1938709071f3ed015ff48fd6d92e26f582c733c4532114a174f25ad7587746",
    "src/mamba_ssm/gpu/gemm_bi_triad/qualification.rs": "b5f663f7ab8a96aaa16acf91050462892f6763139eaa69cd9ac8c6c4fd4b08d6",
    "src/mamba_ssm/gpu/gemm_bi_triad/sm89_finalist_source.rs": "2a51afde2fd1e4203e63acfc2e10c9072e814d075039cbb8baecc19beba5b029",
    "kernels/gemm_bi_triad/sm89_nt_compact.cuh": "dd2a8494666547e797defcfcab3e52a774729abbfe62e630fd0df896391c21e4",
}
OLD_CACHE = {
    "mamba-kernels-v1-3809af034718aa08dfdfc1b6baeb407715436dc7db87ef39825b033c2fd72541.bin": "5851ebd49595a0b2ef665073289e52f903eec3266deadfc5ad5d9f2096ce6d16",
    "mamba-kernels-v1-822ebf0dda7dcf9c4f5c5a2213884dde42ab45f5cf238479a81856042c302b7c.bin": "5035921fc8e078e5454c9c5392f9e9ee6ca6a3de54c1e08cb54bba432fa37546",
    "mamba-kernels-v1-e4f76515e441adc28862775000fce218e2d2cca61b47ce484e15eb1144c40fb3.bin": "f0be052fc160b9e5f4b6a897878781327968c0015d551128415a94b85e6fc0aa",
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
        "schema": "MambaTriadAdaFinalistEarlyBitsSourcesV1",
        "head": HEAD,
        "count": len(sources),
        "excluded_local_wip_not_synced": [
            "src/mamba_ssm/gpu/gemm_bi_triad/qualification.rs (Task3 WIP; remote is committed Task2)",
            "tests/gemm_bi_performance_matrix.rs (Task3 WIP; remote is committed Task2)",
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
    cache_initial = cache_hashes(cache)
    if cache_initial != OLD_CACHE:
        raise RuntimeError(f"initial cache is not exact old three: {cache_initial}")
    manifest = source_manifest()
    write_json(EVIDENCE / "source-manifest.json", manifest)

    build_args = [
        "cargo",
        "test",
        "--release",
        "--features",
        str(context["feature"]),
        "--test",
        "gemm_bi_tf32_cohort_binding",
        "--no-run",
        "--message-format=json-render-diagnostics",
    ]
    with (EVIDENCE / "build.log").open("x") as output:
        build = subprocess.run(
            build_args,
            cwd=ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    executables = []
    for line in (EVIDENCE / "build.log").read_text().splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            message.get("reason") == "compiler-artifact"
            and message.get("target", {}).get("name") == "gemm_bi_tf32_cohort_binding"
            and message.get("executable")
        ):
            executables.append(Path(message["executable"]))
    cache_after_build = cache_hashes(cache)
    if build.returncode != 0 or len(executables) != 1 or cache_after_build != OLD_CACHE:
        raise RuntimeError(
            f"build/list precondition failed: exit={build.returncode} executables={executables} "
            f"cache={cache_after_build}"
        )
    binary = executables[0]
    listing = subprocess.run(
        [str(binary), "--list"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    list_text = listing.stdout + listing.stderr
    (EVIDENCE / "test-list.log").write_text(list_text)
    names = {
        line.removesuffix(": test")
        for line in list_text.splitlines()
        if line.endswith(": test")
    }
    matches = [name for name in names if name == TEST_SUFFIX or name.endswith("::" + TEST_SUFFIX)]
    if listing.returncode != 0 or len(matches) != 1:
        raise RuntimeError(f"exact test preflight failed: exit={listing.returncode}, matches={matches}")
    full_test = matches[0]
    pre = telemetry("PRE", environment)
    write_json(EVIDENCE / "pre.json", pre)
    if not quiet(pre):
        raise RuntimeError(f"PRE is not strict quiet/no-apps: {pre}")
    run_args = [str(binary), full_test, "--exact", "--ignored", "--nocapture"]
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with (EVIDENCE / "test.log").open("x") as output:
        run = subprocess.run(
            run_args,
            cwd=ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    release = telemetry("RELEASE", environment)
    write_json(EVIDENCE / "release.json", release)
    drain = release
    if not quiet(drain):
        for _ in range(60):
            time.sleep(1)
            drain = telemetry("DRAIN", environment)
            if quiet(drain):
                break
    write_json(EVIDENCE / "drain.json", drain)
    output = (EVIDENCE / "test.log").read_text()
    executed_one = "1 passed" in output and "0 passed" not in output
    records = [
        json.loads(line)
        for line in output.splitlines()
        if line.startswith("{") and line.endswith("}")
    ]
    bits = [record for record in records if record.get("kind") == "sm89_nt_finalist_bits"]
    cache_final = cache_hashes(cache)
    new_cache = {
        name: digest for name, digest in cache_final.items() if name not in OLD_CACHE
    }
    old_unchanged = all(cache_final.get(name) == digest for name, digest in OLD_CACHE.items())
    receipt = {
        "schema": "MambaTriadAdaFinalistEarlyBitsReceiptV1",
        "head": HEAD,
        "toolkit": context["toolkit"],
        "feature": context["feature"],
        "generation": GENERATION,
        "target": str(context["target"]),
        "cache": str(cache),
        "cache_mode": oct(stat.S_IMODE(cache.stat().st_mode)),
        "cache_reused_not_cold": True,
        "cache_initial": cache_initial,
        "cache_after_build": cache_after_build,
        "cache_final": cache_final,
        "old_three_unchanged": old_unchanged,
        "new_cache_artifacts": new_cache,
        "build_args": build_args,
        "build_exit": build.returncode,
        "binary": str(binary),
        "binary_sha256": sha(binary),
        "list_exit": listing.returncode,
        "listed_test": full_test,
        "run_args": run_args,
        "run_started_utc": started,
        "run_exit": run.returncode,
        "executed_one": executed_one,
        "bits_record_count": len(bits),
        "bits_cases": [record.get("case") for record in bits],
        "source_manifest_sha256": sha(EVIDENCE / "source-manifest.json"),
        "pre_utc": pre["utc"],
        "release_utc": release["utc"],
        "release_was_quiet": quiet(release),
        "drain_utc": drain["utc"],
        "drain_quiet": quiet(drain),
    }
    write_json(EVIDENCE / "command.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    print("WRAPPER_COMPLETE")
    ok = (
        run.returncode == 0
        and executed_one
        and len(bits) == 7
        and old_unchanged
        and len(new_cache) == 1
        and quiet(drain)
    )
    return 0 if ok else 97


if __name__ == "__main__":
    sys.exit(main())
