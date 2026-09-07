#!/usr/bin/env python3
"""Read-only Nsight Compute acquisition for production Fixed TF32 E0."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


TASK8 = Path("/root/evidence-ada-exact-toolkit-auto-20260907")
OUT = Path("/root/evidence-ada-tf32-e0-ncu-20260907/run")
EXPECTED_BINARY_SHA = "4ecefea9345226073e9671017bb14450ebed562be8c5c551a2e6d9de3859868d"
CUSTOM_SYMBOL = "gemm_bi_nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3"
FAST_SYMBOL = (
    "_ZN7cutlass7Kernel2I52cutlass_80_tensorop_s1688gemm_"
    "128x128_16x5_nn_align4EEvNT_6ParamsE"
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
)
TRAFFIC_METRICS = ",".join(
    (
        "dram__bytes.sum",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "lts__t_sectors.sum",
        "lts__t_sectors_lookup_hit.sum",
        "lts__t_sectors_lookup_miss.sum",
        "lts__t_sectors_op_read.sum",
        "lts__t_sectors_op_write.sum",
        "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum",
        "smsp__inst_executed_pipe_tensor_op_hmma.sum",
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


def ncu_capture(label, symbol, binary, ncu, context, commands):
    common = [
        ncu,
        "--config-file",
        "off",
        "--target-processes",
        "application-only",
        "--kernel-name-base",
        "mangled",
        "--kernel-name",
        "regex:^" + symbol + "$",
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
    test = [
        binary,
        "fixed_ada_production_auto_paired_precision_cublas",
        "--exact",
        "--ignored",
        "--nocapture",
        "--test-threads=1",
    ]
    section_args = list(common)
    for section in SECTIONS:
        section_args.extend(("--section", section))
    section_args.extend(("--export", OUT / label, *test))
    run(
        section_args,
        OUT / f"ncu-{label}.log",
        cwd=context["root"],
        env=context["env"],
        commands=commands,
    )
    report = OUT / f"{label}.ncu-rep"
    if not report.is_file() or report.stat().st_size == 0:
        raise RuntimeError(f"missing report {report}")
    profile_log = (OUT / f"ncu-{label}.log").read_text()
    if "test result: ok. 1 passed; 0 failed;" not in profile_log:
        raise RuntimeError(f"test did not pass during {label}")

    traffic_args = [
        *common,
        "--metrics",
        TRAFFIC_METRICS,
        "--export",
        OUT / f"{label}-traffic",
        *test,
    ]
    run(
        traffic_args,
        OUT / f"ncu-{label}-traffic.log",
        cwd=context["root"],
        env=context["env"],
        commands=commands,
    )
    traffic_report = OUT / f"{label}-traffic.ncu-rep"
    if not traffic_report.is_file() or traffic_report.stat().st_size == 0:
        raise RuntimeError(f"missing report {traffic_report}")

    exports = (
        (report, "details", ["--page", "details", "--csv"]),
        (report, "raw", ["--page", "raw", "--csv"]),
        (
            report,
            "sass",
            ["--page", "source", "--print-source", "sass", "--csv"],
        ),
        (report, "session", ["--page", "session"]),
        (traffic_report, "traffic", ["--page", "raw", "--csv"]),
    )
    for source, suffix, options in exports:
        run(
            [ncu, "--config-file", "off", "--import", source, *options],
            OUT / f"{label}-{suffix}.csv"
            if suffix != "session"
            else OUT / f"{label}-{suffix}.txt",
            cwd=context["root"],
            env=context["env"],
            commands=commands,
        )


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
            MAMBA_FIXED_VENDOR_EXACT_CC="8.9",
            MAMBA_FIXED_ADA_WINDOWS="1",
            MAMBA_FIXED_ADA_ROWS="tf32",
            MAMBA_FIXED_ADA_CELLS="hot_e",
            MAMBA_FIXED_ADA_BIAS="0",
        )
        if context["env"].get("NVIDIA_TF32_OVERRIDE") == "0":
            raise RuntimeError("NVIDIA_TF32_OVERRIDE=0 is forbidden")

        binding_path = TASK8 / "cuda132-binding-final2.json"
        binding = json.loads(binding_path.read_text())
        task8.verify_binding(binding, context)
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
            "row": "tf32",
            "cell": "hot_e",
            "shape_mkn": [2048, 2304, 768],
            "bias": False,
            "windows": 1,
            "vendor_compute": "CUBLAS_COMPUTE_32F_FAST_TF32",
            "custom_symbol": CUSTOM_SYMBOL,
            "fast_symbol": FAST_SYMBOL,
            "launch_skip_matching": 130,
            "launch_count": 1,
            "replay_mode": "kernel",
            "clock_control": "none",
            "cache_control": "none",
            "pipeline_boost_state": "dynamic",
            "sections": list(SECTIONS),
            "traffic_metrics": TRAFFIC_METRICS.split(","),
        }
        (OUT / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")

        status["stage"] = "tool-and-pre"
        run(
            [ncu, "--version"],
            OUT / "ncu-version.log",
            cwd=context["root"],
            env=context["env"],
            commands=commands,
        )
        run(
            [ncu, "--help"],
            OUT / "ncu-help.log",
            cwd=context["root"],
            env=context["env"],
            commands=commands,
        )
        run(
            [
                "/usr/bin/nvidia-smi",
                "--query-gpu=uuid,name,compute_cap,driver_version,pstate,"
                "clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,"
                "utilization.gpu,utilization.memory,memory.used",
                "--format=csv,noheader",
            ],
            OUT / "environment-pre.log",
            cwd=context["root"],
            env=context["env"],
            commands=commands,
        )
        task8.telemetry("PRE", OUT, context)

        status["stage"] = "unprofiled-baseline"
        test = [
            binary,
            "fixed_ada_production_auto_paired_precision_cublas",
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ]
        run(
            test,
            OUT / "unprofiled-baseline.log",
            cwd=context["root"],
            env=context["env"],
            commands=commands,
        )
        records = []
        for line in (OUT / "unprofiled-baseline.log").read_text().splitlines():
            if line.startswith('{"schema":"MambaBiFixedExplicitAutoVendor'):
                records.append(json.loads(line))
        samples = [record for record in records if "order" in record]
        if len(samples) != 2 or not all(
            record["row"] == "tf32"
            and record["cell"] == "hot_e"
            and record["bias"] is False
            and record["auto_tile"] == "Tf32RnaM128N128S3"
            and record["vendor_compute"] == "CUBLAS_COMPUTE_32F_FAST_TF32"
            and record["repeat_bits_equal"] is True
            for record in samples
        ):
            raise RuntimeError("unprofiled comparator identity/result mismatch")
        (OUT / "unprofiled-baseline.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )

        status["stage"] = "custom-profile"
        ncu_capture("custom-rna", CUSTOM_SYMBOL, binary, ncu, context, commands)
        status["stage"] = "fast-profile"
        ncu_capture("fast-tf32", FAST_SYMBOL, binary, ncu, context, commands)

        status["stage"] = "release"
        task8.verify_binding(binding, context)
        task8.telemetry("RELEASE", OUT, context)
        run(
            [
                "/usr/bin/nvidia-smi",
                "--query-gpu=uuid,name,compute_cap,driver_version,pstate,"
                "clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,"
                "utilization.gpu,utilization.memory,memory.used",
                "--format=csv,noheader",
            ],
            OUT / "environment-release.log",
            cwd=context["root"],
            env=context["env"],
            commands=commands,
        )
        (OUT / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        manifest = {
            path.name: sha(path)
            for path in sorted(OUT.iterdir())
            if path.is_file() and path.name not in ("manifest.json", "status.json")
        }
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        status = {"stage": "complete", "complete": True, "commands": len(commands)}
        print("TF32_E0_NCU_COUNTERS_COMPLETE_NOT_ADMISSION", flush=True)
    finally:
        status["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (OUT / "status.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
