# SPDX-License-Identifier: MIT

"""Status contract for Snakemake's cluster-generic executor (stdout only)."""

import argparse
import re
import subprocess
import sys

FAILED_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}


def classify_accounting(output, job_id):
    for line in output.splitlines():
        fields = [value.strip() for value in line.split("|")]
        if len(fields) < 3 or fields[0] != job_id:
            continue
        state = fields[1].split()[0].rstrip("+") if fields[1] else ""
        if state == "COMPLETED":
            return "success" if fields[2] == "0:0" else "failed"
        return "failed" if state in FAILED_STATES else "running"
    return None


def job_status(job_id, runner=None):
    if not re.fullmatch(r"\d+(?:_\d+)?(?:\+\d+)?", job_id):
        raise ValueError(
            "Expected a numeric Slurm job ID, optionally an array/heterogeneous index."
        )
    runner = runner or subprocess.run
    try:
        account = runner(
            [
                "sacct",
                "-X",
                "--noheader",
                "--parsable2",
                "--jobs",
                job_id,
                "--format=JobIDRaw,State%40,ExitCode",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        result = (
            classify_accounting(account.stdout, job_id)
            if account.returncode == 0
            else None
        )
        if result:
            return result
        queue = runner(
            ["squeue", "--noheader", "--jobs", job_id, "--format=%T"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if queue.returncode == 0 and queue.stdout.strip():
            return "running"
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"Slurm status lookup unavailable: {error}", file=sys.stderr)
    # Accounting can lag; absence is not evidence of successful completion.
    print(
        f"No terminal Slurm accounting record yet for {job_id}; inspect sacct/squeue if this persists.",
        file=sys.stderr,
    )
    return "running"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    args = parser.parse_args()
    print(job_status(args.job_id))
