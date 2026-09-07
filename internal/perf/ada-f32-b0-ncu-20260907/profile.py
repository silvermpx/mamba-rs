#!/usr/bin/env python3
"""Read-only Nsight Compute acquisition for production exact-F32 B0."""

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time


TASK8 = Path("/root/evidence-ada-exact-toolkit-auto-20260907")
OUT = Path("/root/evidence-ada-f32-b0-ncu-20260907/run3")
PRIOR = Path("/root/evidence-ada-f32-b0-ncu-20260907/run2")
EXPECTED_BINARY_SHA = "4ecefea9345226073e9671017bb14450ebed562be8c5c551a2e6d9de3859868d"
CUSTOM_SYMBOL = "gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1"
# Match the actual CUTLASS tensor-op kernel family, then record the full name
# from the report instead of treating the historical CUDA12.8/13.0 name as truth.
FAST_FILTER = (
    "_ZN7cutlass7Kernel2I52cutlass_80_tensorop_s1688gemm_"
    ".*_nn_align4EEvNT_6ParamsE"
)
SECTIONS = (
    "LaunchStats",
    "Occupancy",
    "SpeedOfLight",
    "ComputeWorkloadAnalysis",
    "SchedulerStats",
    "WarpStateStats",
    "MemoryWorkloadAnalysis",
    "InstructionStats",
    "SourceCounters",
)
EXPECTED_ADDITIVE_INPUTS = [
    "tests/gemm_bi_fixed_half_batch_discovery.cu",
    "tests/gemm_bi_fixed_half_batch_discovery.rs",
    "tests/gemm_bi_fixed_tf32_n96_discovery.cu",
    "tests/gemm_bi_fixed_tf32_n96_discovery.rs",
    "tests/gemm_bi_fixed_tf32_w4_discovery.cu",
    "tests/gemm_bi_fixed_tf32_w4_discovery.rs",
    "tests/support/fixed_half_batch_layout.rs",
]
METRICS = ",".join(
    (
        "dram__bytes.sum",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "lts__t_sectors.sum",
        "lts__t_sectors_lookup_hit.sum",
        "lts__t_sectors_lookup_miss.sum",
        "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum",
    )
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_task8():
    spec = importlib.util.spec_from_file_location("task8_runner", TASK8 / "run.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load accepted Task8 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_frozen_basis(task8, binding, context):
    """Verify the frozen Task8 basis while permitting later additive tests."""
    if binding["toolkit"] != context["toolkit"]:
        raise RuntimeError("wrong toolkit binding")
    if binding["feature"] != context["feature"]:
        raise RuntimeError("wrong feature binding")
    current = task8.source_inputs()
    missing = sorted(set(binding["inputs"]) - set(current))
    changed = sorted(
        path
        for path, expected in binding["inputs"].items()
        if path in current and current[path] != expected
    )
    if missing or changed:
        raise RuntimeError(f"frozen Task8 inputs changed: missing={missing} changed={changed}")
    if binding["measured_source_sha"] != task8.measured_source_sha():
        raise RuntimeError("measured source changed")
    if stat.S_IMODE(context["cache"].stat().st_mode) != 0o700:
        raise RuntimeError("cache mode changed")
    for path, expected in (
        binding["binaries"]
        | binding["tools"]
        | binding["libraries"]
        | binding["support_tools"]
    ).items():
        if sha(path) != expected:
            raise RuntimeError("bound artifact changed " + path)
    additive = sorted(set(current) - set(binding["inputs"]))
    if additive != EXPECTED_ADDITIVE_INPUTS:
        raise RuntimeError(f"unexpected additive source inventory: {additive}")
    return additive


def run(args, output, *, cwd, env, commands):
    args = [str(arg) for arg in args]
    print("COMMAND " + json.dumps(args), flush=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with output.open("x") as stream:
        result = subprocess.run(
            args,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    commands.append(
        {
            "args": args,
            "output": output.name,
            "started_utc": started,
            "exit": result.returncode,
        }
    )
    print(f"COMMAND_EXIT {output.name} {result.returncode}", flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {output.name} exit {result.returncode}")


def captured_from_raw(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 2:
        # Nsight raw exports have one units row followed by one captured row.
        raise RuntimeError(f"unexpected raw row count {len(rows)} in {path}")
    captured = rows[-1]
    return {
        key: captured.get(key)
        for key in (
            "Kernel Name",
            "Grid Size",
            "Block Size",
            "launch__grid_dim_x",
            "launch__grid_dim_y",
            "launch__grid_dim_z",
            "launch__block_dim_x",
            "launch__block_dim_y",
            "launch__block_dim_z",
            "launch__registers_per_thread",
            "launch__shared_mem_per_block_static",
            "launch__shared_mem_per_block_dynamic",
            "gpu__time_duration.sum",
        )
    }


def capture(label, kernel_filter, binary, ncu, context, commands):
    args = [
        ncu,
        "--config-file",
        "off",
        "--target-processes",
        "application-only",
        "--kernel-name-base",
        "mangled",
        "--kernel-name",
        "regex:^" + kernel_filter + "$",
        "--launch-skip",
        "130",
        "--launch-count",
        "1",
        "--replay-mode",
        "kernel",
        "--clock-control",
        "none",
        "--cache-control",
        "none",
        "--pipeline-boost-state",
        "dynamic",
    ]
    for section in SECTIONS:
        args.extend(("--section", section))
    args.extend(("--metrics", METRICS, "--export", OUT / label))
    args.extend(
        (
            binary,
            "fixed_ada_forced_rungs_paired_precision_cublas",
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        )
    )
    run(
        args,
        OUT / f"ncu-{label}.log",
        cwd=context["root"],
        env=context["env"],
        commands=commands,
    )
    report = OUT / f"{label}.ncu-rep"
    if not report.is_file() or report.stat().st_size == 0:
        raise RuntimeError(f"missing report {report}")
    if "test result: ok. 1 passed; 0 failed;" not in (
        OUT / f"ncu-{label}.log"
    ).read_text():
        raise RuntimeError(f"test did not pass during {label}")
    for suffix, options in (
        ("details", ("--page", "details", "--csv")),
        ("raw", ("--page", "raw", "--csv")),
        ("sass", ("--page", "source", "--print-source", "sass", "--csv")),
        ("session", ("--page", "session")),
    ):
        run(
            [ncu, "--config-file", "off", "--import", report, *options],
            OUT / (f"{label}-{suffix}.csv" if suffix != "session" else f"{label}-{suffix}.txt"),
            cwd=context["root"],
            env=context["env"],
            commands=commands,
        )
    return captured_from_raw(OUT / f"{label}-raw.csv")


def main():
    if len(sys.argv) != 1:
        raise SystemExit("profile.py takes no arguments")
    OUT.mkdir(mode=0o700, parents=True)
    commands = []
    status = {"stage": "preflight", "complete": False}
    try:
        task8 = load_task8()
        context = task8.env_for("13.2", "final2")
        context["root"] = task8.ROOT
        context["env"]["CUDA_VISIBLE_DEVICES"] = task8.UUID
        context["env"].update(
            MAMBA_FIXED_ADA_VENDOR="1",
            MAMBA_FIXED_ADA_ROWS="f32_exact_fast",
            MAMBA_FIXED_ADA_CELLS="hot_b",
            MAMBA_FIXED_ADA_BIAS="0",
            MAMBA_FIXED_ADA_WINDOWS="1",
            MAMBA_FIXED_VENDOR_TILES="F32Sm89N64CopyPlan",
            MAMBA_FIXED_VENDOR_PATHS="eager",
            MAMBA_FIXED_VENDOR_EXACT_CC="8.9",
        )
        if "MAMBA_FIXED_ADA_TOOLKIT_ADMISSION" in context["env"]:
            raise RuntimeError("toolkit admission mode is forbidden")
        if context["env"].get("NVIDIA_TF32_OVERRIDE") == "0":
            raise RuntimeError("NVIDIA_TF32_OVERRIDE=0 is forbidden")

        binding_path = TASK8 / "cuda132-binding-final2.json"
        binding = json.loads(binding_path.read_text())
        additive_inputs = verify_frozen_basis(task8, binding, context)
        binary = task8.find_binary(context, "gemm_bi_fixed_performance")
        if sha(binary) != EXPECTED_BINARY_SHA:
            raise RuntimeError("accepted performance binary changed")
        ncu = context["cuda"] / "bin/ncu"
        if not ncu.is_file():
            raise RuntimeError("CUDA 13.2 ncu missing")

        identity = {
            "purpose": "counter-only-not-admission",
            "toolkit": "13.2",
            "generation": "final2",
            "binding": str(binding_path),
            "binding_sha": sha(binding_path),
            "measured_source_sha": binding["measured_source_sha"],
            "binary": str(binary),
            "binary_sha": sha(binary),
            "cache": str(context["cache"]),
            "cache_mode": oct(context["cache"].stat().st_mode & 0o777),
            "row": "f32_exact_fast",
            "cell": "hot_b",
            "shape_mkn": [4621, 768, 2304],
            "bias": False,
            "windows": 1,
            "auto_and_forced_tile": "F32Sm89N64CopyPlan",
            "vendor_compute": "CUBLAS_COMPUTE_32F_FAST_TF32",
            "custom_symbol": CUSTOM_SYMBOL,
            "fast_filter": FAST_FILTER,
            "launch_skip_matching": 130,
            "launch_count": 1,
            "replay_mode": "kernel",
            "clock_control": "none",
            "cache_control": "none",
            "pipeline_boost_state": "dynamic",
            "sections": list(SECTIONS),
            "metrics": METRICS.split(","),
            "bound_input_count": len(binding["inputs"]),
            "additive_unbound_inputs": additive_inputs,
            "continued_from": {
                "run": str(PRIOR),
                "unprofiled_baseline_sha": sha(PRIOR / "unprofiled-baseline.log"),
                "copyplan_report_sha": sha(PRIOR / "copyplan.ncu-rep"),
                "copyplan_raw_sha": sha(PRIOR / "copyplan-raw.csv"),
            },
        }
        (OUT / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")

        status["stage"] = "tool-and-pre"
        run([ncu, "--version"], OUT / "ncu-version.log", cwd=context["root"], env=context["env"], commands=commands)
        run([ncu, "--help"], OUT / "ncu-help.log", cwd=context["root"], env=context["env"], commands=commands)
        run(
            [
                "/usr/bin/nvidia-smi",
                "--query-gpu=uuid,name,compute_cap,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,utilization.gpu,utilization.memory,memory.used",
                "--format=csv,noheader",
            ],
            OUT / "environment-pre.log",
            cwd=context["root"], env=context["env"], commands=commands,
        )
        task8.telemetry("PRE", OUT, context)

        status["stage"] = "validate-prior-baseline"
        records = []
        for line in (PRIOR / "unprofiled-baseline.log").read_text().splitlines():
            if line.startswith('{"schema":"MambaBiFixedExplicitForcedRung'):
                records.append(json.loads(line))
        samples = [record for record in records if "order" in record]
        if len(samples) != 2 or not all(
            record["row"] == "f32_exact_fast"
            and record["cell"] == "hot_b"
            and record["bias"] is False
            and record["auto_tile"] == "F32Sm89N64CopyPlan"
            and record["forced_tile"] == "F32Sm89N64CopyPlan"
            and record["path"] == "eager"
            and record["vendor_compute"] == "CUBLAS_COMPUTE_32F_FAST_TF32"
            and record["raw_storage_bits_equal"] is True
            and record["repeat_bits_equal"] is True
            for record in samples
        ):
            raise RuntimeError("unprofiled comparator identity/result mismatch")
        (OUT / "validated-prior-baseline.json").write_text(json.dumps(records, indent=2) + "\n")

        status["stage"] = "validate-prior-copyplan"
        captures = {"copyplan": captured_from_raw(PRIOR / "copyplan-raw.csv")}
        status["stage"] = "fast-profile"
        captures["fast"] = capture("fast", FAST_FILTER, binary, ncu, context, commands)
        if captures["copyplan"]["Kernel Name"] != CUSTOM_SYMBOL:
            raise RuntimeError("wrong CopyPlan kernel captured")
        if not captures["fast"]["Kernel Name"] or "cutlass_80_tensorop_s1688gemm_" not in captures["fast"]["Kernel Name"]:
            raise RuntimeError("wrong Fast kernel captured")
        (OUT / "captured.json").write_text(json.dumps(captures, indent=2) + "\n")

        status["stage"] = "release"
        if verify_frozen_basis(task8, binding, context) != additive_inputs:
            raise RuntimeError("additive source inventory changed during acquisition")
        task8.telemetry("RELEASE", OUT, context)
        run(
            [
                "/usr/bin/nvidia-smi",
                "--query-gpu=uuid,name,compute_cap,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,utilization.gpu,utilization.memory,memory.used",
                "--format=csv,noheader",
            ],
            OUT / "environment-release.log",
            cwd=context["root"], env=context["env"], commands=commands,
        )
        (OUT / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        manifest = {
            path.name: sha(path)
            for path in sorted(OUT.iterdir())
            if path.is_file() and path.name not in ("manifest.json", "status.json")
        }
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        status = {"stage": "complete", "complete": True, "commands": len(commands)}
        print("F32_B0_NCU_COUNTERS_COMPLETE_NOT_ADMISSION", flush=True)
    finally:
        status["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (OUT / "status.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
