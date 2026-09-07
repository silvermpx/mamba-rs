#!/usr/bin/env python3
"""Task8 cross-toolkit builder/runner; every command uses env_for()."""

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time


UUID = "GPU-d1edd7be-e88d-aed6-047d-622163306f0e"
ROOT = Path("/root/mamba-ada-exact-toolkit-auto-20260907")
EVIDENCE = Path("/root/evidence-ada-exact-toolkit-auto-20260907")
TOOLKITS = {
    "12.8": ("128", "12080"),
    "13.0": ("130", "13000"),
    "13.2": ("132", "13020"),
}
LITERALS = ",".join(
    f"hot_{cell}:{bias}" for cell in ("a", "b", "d", "e") for bias in (0, 1)
)
BOUND_BINARY_STEMS = (
    "gemm_bi_fixed_performance",
    "gemm_bi_fixed_sm89_exact_n64",
    "gemm_bi_fixed_sm89_pipeline",
    "gemm_bi_tf32_cohort_binding",
    "gemm_bi_fixed_correctness",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def env_for(toolkit, generation):
    require(toolkit in TOOLKITS, "unsupported toolkit")
    require(re.fullmatch(r"[a-z0-9-]+", generation), "invalid generation")
    tag, feature = TOOLKITS[toolkit]
    cuda = Path("/usr/local/cuda-" + toolkit)
    target = Path(f"/root/target-ada-exact-toolkit-auto-{generation}-cuda{tag}-20260907")
    cache = Path(f"/root/mamba-kcache-ada-exact-toolkit-auto-{generation}-cuda{tag}-20260907")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("MAMBA_FIXED_") and key != "NVIDIA_TF32_OVERRIDE"
    }
    environment.update(
        CUDA_HOME=str(cuda),
        CUDA_PATH=str(cuda),
        PATH=str(cuda / "bin") + ":/root/.cargo/bin:" + os.environ["PATH"],
        LD_LIBRARY_PATH=str(cuda / "lib64"),
        CARGO_TARGET_DIR=str(target),
        MAMBA_RS_KERNEL_CACHE=str(cache),
    )
    return {
        "toolkit": toolkit,
        "tag": tag,
        "feature": "cuda,cudarc/cuda-" + feature,
        "cuda": cuda,
        "target": target,
        "cache": cache,
        "env": environment,
    }


def source_inputs():
    paths = [ROOT / "Cargo.toml", ROOT / "Cargo.lock"]
    for directory in ("src", "kernels", "tests"):
        paths.extend(
            path
            for path in (ROOT / directory).rglob("*")
            if path.is_file() and not path.name.startswith("._")
        )
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(paths)}


def measured_source_sha():
    digest = hashlib.sha256()
    for relative in (
        "src/mamba_ssm/gpu/gemm_bi_fixed.rs",
        "src/mamba_ssm/gpu/kernel_identity.rs",
        "tests/gemm_bi_fixed_performance.rs",
        "tests/support/fixed_sm89_toolkit_admission.rs",
    ):
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def command(args, log, context):
    print("COMMAND " + json.dumps([str(arg) for arg in args]), flush=True)
    with log.open("x") as output:
        result = subprocess.run(
            args,
            cwd=ROOT,
            env=context["env"],
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    print(f"COMMAND_EXIT {log.name} {result.returncode}", flush=True)
    return result.returncode


def telemetry(phase, directory, context):
    data = {"phase": phase, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for kind, query in [
        ("gpu", "--query-gpu=uuid,name,compute_cap,utilization.gpu,utilization.memory"),
        ("apps", "--query-compute-apps=pid,gpu_uuid,process_name"),
    ]:
        result = subprocess.run(
            ["/usr/bin/nvidia-smi", query, "--format=csv,noheader"],
            env=context["env"],
            text=True,
            capture_output=True,
        )
        data[kind] = result.stdout
        data[kind + "_stderr"] = result.stderr
        data[kind + "_exit"] = result.returncode
    (directory / (phase.lower() + ".json")).write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data), flush=True)
    require(data["gpu_exit"] == data["apps_exit"] == 0, phase + " telemetry failed")
    fields = [part.strip() for part in data["gpu"].strip().split(",")]
    require(fields[:3] == [UUID, "NVIDIA RTX 6000 Ada Generation", "8.9"], phase + " identity")
    require(not data["apps"].strip(), phase + " active compute app")
    if phase in ("PRE", "RELEASE"):
        require(fields[3:] == ["0 %", "0 %"], phase + " is not quiet")


def cargo(context):
    return ["cargo", "test", "--release", "--features", context["feature"]]


def find_binary(context, stem):
    binaries = [
        path
        for path in (context["target"] / "release/deps").glob(stem + "-*")
        if path.is_file() and os.access(path, os.X_OK) and "." not in path.name
    ]
    require(len(binaries) == 1, f"ambiguous {stem} binary inventory {binaries}")
    return binaries[0]


def tool_hashes():
    paths = [EVIDENCE / name for name in ("run.py", "analyze.py", "remote.py")]
    paths.append(Path("/root/evidence-ada-f32-tf32-toolkit-20260907/analyze.py"))
    return {str(path): sha(path) for path in paths}


def verify_binding(binding, context):
    require(binding["toolkit"] == context["toolkit"], "wrong toolkit binding")
    require(binding["feature"] == context["feature"], "wrong feature binding")
    require(binding["inputs"] == source_inputs(), "source/build inputs changed")
    require(binding["measured_source_sha"] == measured_source_sha(), "measured source changed")
    require(stat.S_IMODE(context["cache"].stat().st_mode) == 0o700, "cache mode changed")
    for path, expected in (
        binding["binaries"]
        | binding["tools"]
        | binding["libraries"]
        | binding["support_tools"]
    ).items():
        require(sha(path) == expected, "bound artifact changed " + path)


def functional_checks(toolkit):
    require(toolkit in TOOLKITS, "unsupported functional toolkit")
    checks = [
        (
            "exact-auto",
            "gemm_bi_fixed_sm89_exact_n64",
            "fixed_sm89_exact_n64_auto_prefix_view_graph_bits",
        ),
        (
            "half-holders",
            "gemm_bi_fixed_sm89_pipeline",
            "fixed_sm89_half_swizzle_and_pipeline_holders_are_independently_live",
        ),
        (
            "half-auto",
            "gemm_bi_fixed_sm89_pipeline",
            "fixed_sm89_half_pipeline_auto_hot_cell_prefix_view_graph_bits",
        ),
        (
            "fixed-rna-hot-a",
            "gemm_bi_fixed_correctness",
            "fixed_sm89_rna_wide_actual_auto_hot_a_route_and_graph",
        ),
        (
            "fixed-rna-all-cells",
            "gemm_bi_fixed_correctness",
            "fixed_sm89_rna_wide_actual_auto_all_cells_prefix_views_and_graph_bits",
        ),
    ]
    if toolkit == "13.2":
        checks.extend(
            [
                (
                    "triad-tf32-cohort",
                    "gemm_bi_tf32_cohort_binding",
                    "tf32_cohort_binds_on_this_board",
                ),
                (
                    "triad-tf32-bias",
                    "gemm_bi_tf32_cohort_binding",
                    "sm89_tf32_bias_cohort_serves_the_qualified_wide_epilogues",
                ),
            ]
        )
    return checks


def main():
    require(len(sys.argv) >= 5, "usage: run.py OP TOOLKIT ATTEMPT GENERATION [OPTIONS]")
    operation, toolkit, attempt, generation, *options = sys.argv[1:]
    require(re.fullmatch(r"[a-z0-9-]+", attempt), "invalid attempt")
    context = env_for(toolkit, generation)
    directory = EVIDENCE / f"cuda{context['tag']}-{attempt}"
    directory.mkdir(parents=True)
    context["cache"].mkdir(mode=0o700, exist_ok=True)
    require(stat.S_IMODE(context["cache"].stat().st_mode) == 0o700, "cache not private0700")
    cargo_test = cargo(context)
    binding_path = EVIDENCE / f"cuda{context['tag']}-binding-{generation}.json"

    if operation == "focused":
        require(not options, "focused takes no options")
        code = command(
            cargo_test
            + [
                "--test",
                "gemm_bi_fixed_performance",
                "toolkit_admission::tests::",
                "--",
                "--nocapture",
                "--test-threads=1",
            ],
            directory / "focused.log",
            context,
        )
        raise SystemExit(code)

    if operation == "library":
        require(not options, "library takes no options")
        code = command(
            cargo_test + ["--lib", "--", "--nocapture", "--test-threads=1"],
            directory / "library.log",
            context,
        )
        raise SystemExit(code)

    if operation == "build":
        require(not options, "build takes no options")
        for name, args in [
            ("nvcc", [str(context["cuda"] / "bin/nvcc"), "--version"]),
            ("rustc", ["rustc", "-Vv"]),
            ("cargo", ["cargo", "-V"]),
        ]:
            require(command(args, directory / (name + ".log"), context) == 0, name + " failed")
        suites = [
            ("library", ["--lib", "--", "--nocapture", "--test-threads=1"]),
            (
                "focused",
                [
                    "--test",
                    "gemm_bi_fixed_performance",
                    "toolkit_admission::tests::",
                    "--",
                    "--nocapture",
                    "--test-threads=1",
                ],
            ),
            (
                "performance-nonignored",
                ["--test", "gemm_bi_fixed_performance", "--", "--test-threads=1"],
            ),
            (
                "exact-nonignored",
                ["--test", "gemm_bi_fixed_sm89_exact_n64", "--", "--test-threads=1"],
            ),
            (
                "half-nonignored",
                ["--test", "gemm_bi_fixed_sm89_pipeline", "--", "--test-threads=1"],
            ),
            (
                "tf32-cohort-nonignored",
                ["--test", "gemm_bi_tf32_cohort_binding", "--", "--test-threads=1"],
            ),
            (
                "fixed-correctness-nonignored",
                ["--test", "gemm_bi_fixed_correctness", "--", "--test-threads=1"],
            ),
        ]
        for name, args in suites:
            require(
                command(cargo_test + args, directory / (name + ".log"), context) == 0,
                name + " failed",
            )
        binaries = {}
        for stem in BOUND_BINARY_STEMS:
            binary = find_binary(context, stem)
            binaries[str(binary)] = sha(binary)
        libraries = {}
        for pattern in ("libnvrtc.so.*", "libcublas.so.*"):
            for path in (context["cuda"] / "targets/x86_64-linux/lib").glob(pattern):
                libraries[str(path)] = sha(path)
        binding = {
            "toolkit": toolkit,
            "feature": context["feature"],
            "cuda": str(context["cuda"]),
            "target": str(context["target"]),
            "cache": str(context["cache"]),
            "cache_mode": oct(stat.S_IMODE(context["cache"].stat().st_mode)),
            "inputs": source_inputs(),
            "measured_source_sha": measured_source_sha(),
            "binaries": binaries,
            "tools": {
                str(path): sha(path)
                for path in (context["cuda"] / "bin/nvcc", context["cuda"] / "bin/ptxas")
            },
            "libraries": libraries,
            "support_tools": tool_hashes(),
        }
        require(not binding_path.exists(), "binding already exists")
        binding_path.write_text(json.dumps(binding, indent=2) + "\n")
        print(
            "BUILD_COMPLETE "
            + json.dumps({key: value for key, value in binding.items() if key != "inputs"}),
            flush=True,
        )
        print("WRAPPER_COMPLETE", flush=True)
        return

    binding = json.loads(binding_path.read_text())
    verify_binding(binding, context)
    binary_by_stem = {
        Path(path).name.split("-")[0]: Path(path) for path in binding["binaries"]
    }

    if operation == "functional":
        require(not options, "functional takes no options")
        telemetry("PRE", directory, context)
        exits = {}
        for name, stem, test_name in functional_checks(toolkit):
            exits[name] = command(
                [
                    str(binary_by_stem[stem]),
                    "--ignored",
                    "--exact",
                    test_name,
                    "--nocapture",
                    "--test-threads=1",
                ],
                directory / (name + ".log"),
                context,
            )
        telemetry("POST", directory, context)
        verify_binding(binding, context)
        result = {"toolkit": toolkit, "operation": operation, "exits": exits}
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print("FUNCTIONAL_RESULT " + json.dumps(result), flush=True)
        require(all(code == 0 for code in exits.values()), "functional exit closure failed")
        print("WRAPPER_COMPLETE", flush=True)
        return

    if operation == "run":
        require(len(options) == 1, "run requires smoke1 or post101")
        stage = options[0]
        windows = {"smoke1": 1, "post101": 101}.get(stage)
        require(windows is not None, "invalid Task8 stage")
        require(toolkit in ("12.8", "13.0"), "Task8 timing excludes 13.2")
        jsonl = directory / "records.jsonl"
        performance = binary_by_stem["gemm_bi_fixed_performance"]
        context["env"].update(
            {
                "MAMBA_FIXED_ADA_EXACT_POST_AUTO": "1",
                "MAMBA_FIXED_ADA_VENDOR": "1",
                "MAMBA_FIXED_ADA_EXACT_POST_STAGE": stage,
                "MAMBA_FIXED_ADA_EXACT_POST_WINDOWS": str(windows),
                "MAMBA_FIXED_ADA_EXACT_POST_TOOLKIT": toolkit,
                "MAMBA_FIXED_ADA_EXACT_POST_LITERALS": LITERALS,
                "MAMBA_FIXED_ADA_EXACT_POST_TUNING_REVISION": "44",
                "MAMBA_FIXED_ADA_EXACT_POST_SOURCE_SHA": binding["measured_source_sha"],
                "MAMBA_FIXED_ADA_EXACT_POST_BINARY_SHA": binding["binaries"][str(performance)],
                "MAMBA_FIXED_ADA_EXACT_POST_JSONL": str(jsonl),
                "MAMBA_FIXED_VENDOR_TILES": "Legacy,F32Sm89N64CopyPlan",
                "MAMBA_FIXED_VENDOR_PATHS": "eager,graph",
                "MAMBA_FIXED_VENDOR_EXACT_CC": "8.9",
            }
        )
        telemetry("PRE", directory, context)
        test_exit = command(
            [
                str(performance),
                "--ignored",
                "--exact",
                "fixed_ada_exact_post_auto_paired_precision_cublas",
                "--nocapture",
                "--test-threads=1",
            ],
            directory / "test.log",
            context,
        )
        post_exit = 0
        try:
            telemetry("POST", directory, context)
            verify_binding(binding, context)
        except Exception as error:
            print("POSTCHECK_FAILURE " + repr(error), flush=True)
            post_exit = 1
        result = {
            "test_exit": test_exit,
            "post_exit": post_exit,
            "toolkit": toolkit,
            "family": "f32_exact_post_auto",
            "stage": stage,
            "windows": windows,
            "literals": LITERALS,
            "source_sha": binding["measured_source_sha"],
            "binary_sha": binding["binaries"][str(performance)],
            "jsonl_sha": sha(jsonl) if jsonl.exists() else None,
            "test_log_sha": sha(directory / "test.log"),
            "cache_files": {
                str(path): sha(path) for path in sorted(context["cache"].glob("*.bin"))
            },
        }
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print("RUN_RESULT " + json.dumps(result), flush=True)
        require(test_exit == post_exit == 0, "run/POST exit closure failed")
        print("WRAPPER_COMPLETE", flush=True)
        return

    if operation == "release":
        require(not options, "release takes no options")
        telemetry("RELEASE", directory, context)
        print("WRAPPER_COMPLETE", flush=True)
        return

    raise ValueError("unsupported operation")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("WRAPPER_FAILURE " + repr(error), flush=True)
        raise SystemExit(1)
