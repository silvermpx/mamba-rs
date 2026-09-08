#!/usr/bin/env python3
"""Capture one NCU launch of the winning Ada F32 TN direct-fold kernel."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path("/root/mamba-ada-exact-toolkit-auto-20260907")
OUT = Path("/root/evidence-ada-triad-f32-tn-d128-direct-20260908/ncu-m16n16")
ENV_RUNNER = Path("/root/evidence-ada-exact-toolkit-auto-20260907/run.py")
TELEMETRY_RUNNER = Path(
    "/root/evidence-ada-triad-nt-padded36-two-arm-20260907/run-arm.py"
)
BINARY = Path(
    "/root/target-ada-exact-toolkit-auto-triad-nt-padded36-two-arm1-cuda132-20260907/"
    "release/deps/gemm_bi_tn_d128_fused_contract_tournament-8219629cb310a691"
)
BINARY_SHA = "0060f788d46e8a5bcf251cba047a6a6ed86e129198df062206d1b5ccc0e607fd"
TEST = "cuda_tournament::ada_d128_direct_fold_two_arm_once7"
SYMBOL = "gemm_bi_tn_ada_d128_direct_m16n16_f64fold_v1"
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


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, output: Path, *, env: dict[str, str]) -> dict[str, object]:
    args = [str(x) for x in args]
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with output.open("x") as stream:
        result = subprocess.run(
            args,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    command = {"args": args, "output": output.name, "started_utc": started, "exit": result.returncode}
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {output} exit {result.returncode}")
    return command


def main() -> int:
    OUT.mkdir(mode=0o700, parents=True, exist_ok=False)
    env_runner = load("f32_tn_ncu_env", ENV_RUNNER)
    telemetry_runner = load("f32_tn_ncu_telemetry", TELEMETRY_RUNNER)
    context = env_runner.env_for("13.2", "triad-nt-padded36-two-arm1")
    env = dict(context["env"])
    if sha(BINARY) != BINARY_SHA:
        raise RuntimeError("test binary changed")
    ncu = Path(context["cuda"]) / "bin/ncu"
    if not ncu.is_file():
        raise RuntimeError("CUDA 13.2 NCU missing")
    commands = []
    telemetry_runner.telemetry("PRE", context, OUT, True)
    args = [
        ncu,
        "--config-file", "off",
        "--target-processes", "application-only",
        "--kernel-name-base", "mangled",
        "--kernel-name", f"regex:^{SYMBOL}$",
        "--launch-skip", "0",
        "--launch-count", "1",
        "--replay-mode", "kernel",
        "--clock-control", "none",
        "--cache-control", "none",
        "--pipeline-boost-state", "dynamic",
    ]
    for section in SECTIONS:
        args.extend(("--section", section))
    report_stem = OUT / "direct-m16n16"
    args.extend(("--metrics", METRICS, "--export", report_stem))
    args.extend((BINARY, TEST, "--ignored", "--exact", "--nocapture"))
    commands.append(run(args, OUT / "ncu.log", env=env))
    telemetry_runner.telemetry("RELEASE", context, OUT, False)
    time.sleep(5)
    telemetry_runner.telemetry("DRAIN", context, OUT, True)
    report = OUT / "direct-m16n16.ncu-rep"
    if not report.is_file() or not report.stat().st_size:
        raise RuntimeError("NCU report missing")
    for label, options in (
        ("details", ("--page", "details", "--csv")),
        ("raw", ("--page", "raw", "--csv")),
        ("sass", ("--page", "source", "--print-source", "sass", "--csv")),
        ("session", ("--page", "session")),
    ):
        suffix = "txt" if label == "session" else "csv"
        commands.append(
            run(
                [ncu, "--config-file", "off", "--import", report, *options],
                OUT / f"direct-m16n16-{label}.{suffix}",
                env=env,
            )
        )
    receipt = {
        "schema": "AdaTnDirectM16N16NcuReceiptV1",
        "binary_sha256": BINARY_SHA,
        "symbol": SYMBOL,
        "launch_skip": 0,
        "launch_count": 1,
        "clock_control": "none",
        "cache_control": "none",
        "commands": commands,
        "report_sha256": sha(report),
        "telemetry_runner_sha256": sha(TELEMETRY_RUNNER),
    }
    (OUT / "command.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
