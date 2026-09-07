#!/usr/bin/env python3
"""Task7 cross-toolkit builder/runner; every command uses env_for()."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

from analyze import validate_screen_for_confirm

UUID = "GPU-d1edd7be-e88d-aed6-047d-622163306f0e"
ROOT = Path("/root/mamba-ada-f32-tf32-toolkit-20260907")
EVIDENCE = Path("/root/evidence-ada-f32-tf32-toolkit-20260907")
TOOLKITS = {"12.8": ("128", "12080"), "13.0": ("130", "13000")}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def env_for(toolkit):
    require(toolkit in TOOLKITS, "unsupported toolkit")
    tag, feature = TOOLKITS[toolkit]
    cuda = Path("/usr/local/cuda-" + toolkit)
    target = Path(f"/root/target-ada-f32-tf32-toolkit-final5b-cuda{tag}-20260907")
    cache = Path(f"/root/mamba-kcache-ada-f32-tf32-toolkit-final5b-cuda{tag}-20260907")
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
            args, cwd=ROOT, env=context["env"], stdout=output, stderr=subprocess.STDOUT
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
            env=context["env"], text=True, capture_output=True,
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


def main():
    operation, toolkit, attempt, *options = sys.argv[1:]
    require(re.fullmatch(r"[a-z0-9-]+", attempt), "invalid attempt")
    context = env_for(toolkit)
    directory = EVIDENCE / f"cuda{context['tag']}-{attempt}"
    directory.mkdir(parents=True)
    context["cache"].mkdir(mode=0o700, exist_ok=True)
    require(stat.S_IMODE(context["cache"].stat().st_mode) == 0o700, "cache not private0700")
    test = [
        "cargo", "test", "--release", "--features", context["feature"],
        "--test", "gemm_bi_fixed_performance",
    ]
    if operation == "host":
        require(not options, "host takes no options")
        exit_code = command(
            test + ["toolkit_admission::tests::", "--", "--nocapture", "--test-threads=1"],
            directory / "host.log", context,
        )
        print("HOST_RESULT " + json.dumps({"exit": exit_code, "toolkit": toolkit}), flush=True)
        sys.exit(exit_code)
    binding_path = EVIDENCE / f"cuda{context['tag']}-binding-final5b.json"
    if operation == "build":
        require(not options, "build takes no options")
        for name, args in [
            ("nvcc", [str(context["cuda"] / "bin/nvcc"), "--version"]),
            ("rustc", ["rustc", "-Vv"]),
            ("cargo", ["cargo", "-V"]),
        ]:
            require(command(args, directory / (name + ".log"), context) == 0, name + " failed")
        require(
            command(
                test + ["toolkit_admission::tests::", "--", "--nocapture", "--test-threads=1"],
                directory / "focused.log", context,
            ) == 0,
            "focused host tests failed",
        )
        require(
            command(test + ["--", "--test-threads=1"], directory / "nonignored.log", context) == 0,
            "nonignored performance tests failed",
        )
        binaries = [
            path
            for path in (context["target"] / "release/deps").glob("gemm_bi_fixed_performance-*")
            if path.is_file() and os.access(path, os.X_OK)
        ]
        require(len(binaries) == 1, f"ambiguous test binary inventory {binaries}")
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
            "binary": str(binaries[0]),
            "binary_sha": sha(binaries[0]),
            "tools": {
                str(path): sha(path)
                for path in (context["cuda"] / "bin/nvcc", context["cuda"] / "bin/ptxas")
            },
            "libraries": libraries,
        }
        require(not binding_path.exists(), "binding already exists")
        binding_path.write_text(json.dumps(binding, indent=2) + "\n")
        print("BUILD_COMPLETE " + json.dumps({k: v for k, v in binding.items() if k != "inputs"}), flush=True)
        return
    require(operation == "run", "unsupported operation")
    require(len(options) in (3, 4), "run requires family stage literals [screen_jsonl]")
    family, stage, literals, *screen = options
    require(family in ("f32_exact_fast", "tf32"), "bad family")
    windows = {"smoke1": 1, "screen21": 21, "confirm101": 101}.get(stage)
    require(windows is not None, "bad stage")
    require((stage == "confirm101") == bool(screen), "confirm screen binding mismatch")
    binding = json.loads(binding_path.read_text())
    require(binding["toolkit"] == toolkit and binding["feature"] == context["feature"], "wrong binding")
    require(binding["inputs"] == source_inputs(), "source/build inputs changed")
    require(binding["measured_source_sha"] == measured_source_sha(), "measured Rust source changed")
    binary = Path(binding["binary"])
    require(sha(binary) == binding["binary_sha"], "binary changed")
    require(stat.S_IMODE(context["cache"].stat().st_mode) == 0o700, "cache mode changed")
    for path, expected in (binding["tools"] | binding["libraries"]).items():
        require(sha(path) == expected, "tool/library changed " + path)
    jsonl = directory / "records.jsonl"
    controls = {
        "MAMBA_FIXED_ADA_TOOLKIT_ADMISSION": "1",
        "MAMBA_FIXED_ADA_VENDOR": "1",
        "MAMBA_FIXED_ADA_ROWS": family,
        "MAMBA_FIXED_ADA_LITERALS": literals,
        "MAMBA_FIXED_ADA_WINDOWS": str(windows),
        "MAMBA_FIXED_ADA_TOOLKIT": toolkit,
        "MAMBA_FIXED_ADA_TUNING_REVISION": "43",
        "MAMBA_FIXED_ADA_STAGE": stage,
        "MAMBA_FIXED_ADA_SOURCE_SHA": binding["measured_source_sha"],
        "MAMBA_FIXED_ADA_BINARY_SHA": binding["binary_sha"],
        "MAMBA_FIXED_ADA_JSONL": str(jsonl),
        "MAMBA_FIXED_VENDOR_TILES": (
            "F32Sm89N64CopyPlan" if family == "f32_exact_fast" else "Tf32M64S2"
        ),
        "MAMBA_FIXED_VENDOR_PATHS": "eager,graph",
        "MAMBA_FIXED_VENDOR_EXACT_CC": "8.9",
    }
    if screen:
        screen_path = Path(screen[0])
        screen_gate = validate_screen_for_confirm(
            screen_path, binding, family, toolkit, literals
        )
        controls["MAMBA_FIXED_ADA_SCREEN_SHA"] = screen_gate["screen_sha"]
        controls["MAMBA_FIXED_ADA_SCREEN_ARTIFACT_SHA"] = screen_gate["artifact_sha"]
    context["env"].update(controls)
    telemetry("PRE", directory, context)
    test_exit = command(
        [
            str(binary), "--ignored", "--exact",
            "fixed_ada_forced_rungs_paired_precision_cublas",
            "--nocapture", "--test-threads=1",
        ],
        directory / "test.log", context,
    )
    post_exit = 0
    try:
        telemetry("POST", directory, context)
        require(binding["inputs"] == source_inputs(), "post source inputs changed")
        require(sha(binary) == binding["binary_sha"], "post binary changed")
    except Exception as error:
        print("POSTCHECK_FAILURE " + repr(error), flush=True)
        post_exit = 1
    result = {
        "test_exit": test_exit,
        "post_exit": post_exit,
        "toolkit": toolkit,
        "family": family,
        "stage": stage,
        "windows": windows,
        "literals": literals,
        "source_sha": binding["measured_source_sha"],
        "binary_sha": binding["binary_sha"],
        "jsonl_sha": sha(jsonl) if jsonl.exists() else None,
        "test_log_sha": sha(directory / "test.log"),
        "cache_files": {str(path): sha(path) for path in sorted(context["cache"].glob("*.bin"))},
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RUN_RESULT " + json.dumps(result), flush=True)
    require(test_exit == post_exit == 0, "run/POST exit closure failed")
    print("WRAPPER_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("WRAPPER_FAILURE " + repr(error), flush=True)
        sys.exit(1)
