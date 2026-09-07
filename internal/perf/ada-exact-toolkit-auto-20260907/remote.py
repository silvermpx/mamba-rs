#!/usr/bin/env python3
"""Run one remote Task8 operation and bind wrapper/outer-SSH closure."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


LOCAL = Path(__file__).resolve().parent
REMOTE_EVIDENCE = "/root/evidence-ada-exact-toolkit-auto-20260907"


def main():
    operation, toolkit, attempt, generation, *options = sys.argv[1:]
    command = [
        "python3",
        f"{REMOTE_EVIDENCE}/run.py",
        operation,
        toolkit,
        attempt,
        generation,
        *options,
    ]
    tag = toolkit.replace(".", "")
    transcript = LOCAL / f"cuda{tag}-{attempt}-ssh.log"
    with transcript.open("x") as output:
        process = subprocess.Popen(
            ["ssh", "ada", *command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            output.write(line)
            output.flush()
        outer_exit = process.wait()
    text = transcript.read_text()
    closure = {
        "outer_ssh_exit": outer_exit,
        "wrapper_complete": "WRAPPER_COMPLETE\n" in text,
        "transcript_sha": hashlib.sha256(transcript.read_bytes()).hexdigest(),
    }
    closure_path = LOCAL / f"cuda{tag}-{attempt}-outer.json"
    closure_path.write_text(json.dumps(closure, indent=2) + "\n")
    remote = f"ada:{REMOTE_EVIDENCE}/cuda{tag}-{attempt}/"
    subprocess.run(["scp", str(transcript), remote + "ssh.log"], check=True)
    subprocess.run(["scp", str(closure_path), remote + "outer.json"], check=True)
    print("OUTER_CLOSURE " + json.dumps(closure), flush=True)
    if outer_exit != 0 or not closure["wrapper_complete"]:
        raise SystemExit(outer_exit or 1)


if __name__ == "__main__":
    main()
