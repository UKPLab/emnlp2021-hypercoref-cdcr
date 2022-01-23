#!/usr/bin/env python3
# When called inside a Slurm job array task, this script returns the i-th file from a directory with a specific file
# extension, based on the job array task ID.

import sys
from pathlib import Path

import slurmee


def get_ith_file_by_array_task_id(path, suffix):
    slurm_job_id = slurmee.get_job_id()
    job_array_info = slurmee.get_job_array_info()

    assert slurm_job_id is not None and job_array_info is not None, "This method is intended only for usage with Slurm job arrays"

    p = Path(path)
    assert p.exists() and p.is_dir(), f"{path} is not a directory or does not exist"

    task_id = job_array_info["task_id"]
    files = sorted([f for f in p.iterdir() if f.is_file() and f.suffix == suffix])
    file_at_ith = files[task_id]
    sys.stdout.write(str(file_at_ith))


if __name__ == "__main__":
    get_ith_file_by_array_task_id(*sys.argv[1:])
