#!/usr/bin/env python3
"""Run the bounded F32 B0 profile remotely and mirror raw evidence."""

import hashlib
import json
from pathlib import Path
import subprocess


LOCAL = Path(__file__).resolve().parent
REMOTE = "/root/evidence-ada-f32-b0-ncu-20260907"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    transcript = LOCAL / "outer-ssh3.log"
    setup = subprocess.run(
        ["ssh", "ada", "mkdir", "-p", "-m", "700", REMOTE],
        text=True,
        capture_output=True,
    )
    if setup.returncode != 0:
        raise SystemExit(setup.stderr or setup.returncode)
    subprocess.run(["scp", str(LOCAL / "profile.py"), f"ada:{REMOTE}/profile.py"], check=True)
    with transcript.open("x") as output:
        process = subprocess.Popen(
            ["ssh", "ada", "python3", f"{REMOTE}/profile.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            output.write(line)
            output.flush()
        exit_code = process.wait()
    closure = {
        "outer_ssh_exit": exit_code,
        "complete_marker": "F32_B0_NCU_COUNTERS_COMPLETE_NOT_ADMISSION\n" in transcript.read_text(),
        "transcript_sha": sha(transcript),
        "profile_tool_sha": sha(LOCAL / "profile.py"),
    }
    (LOCAL / "outer3.json").write_text(json.dumps(closure, indent=2) + "\n")
    mirror = LOCAL / "run3"
    if mirror.exists():
        raise RuntimeError("local run3 mirror already exists")
    subprocess.run(["scp", "-r", f"ada:{REMOTE}/run3", str(LOCAL)], check=True)
    print("OUTER_CLOSURE " + json.dumps(closure), flush=True)
    if exit_code != 0 or not closure["complete_marker"]:
        raise SystemExit(exit_code or 1)


if __name__ == "__main__":
    main()
