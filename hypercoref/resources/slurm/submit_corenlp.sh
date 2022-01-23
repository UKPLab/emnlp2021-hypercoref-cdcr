#!/usr/bin/env bash
#
# Job which runs CoreNLP for one day. Reschedules itself as long as there are jobs in the queue whose job name contains
# `needs_corenlp`.
#
#SBATCH --job-name=corenlp
#SBATCH --chdir=/.../hypercoref/
#SBATCH --output=/.../corenlp_stdout.txt
#SBATCH --error=/.../corenlp_stderr.txt
#SBATCH --open-mode=append
# ☝️ append to same logfile
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=...
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=...
# ☝️ always use the same node so that we can configure other jobs to always point to this node
#SBATCH --mem=16384
#SBATCH --time=1-0
#SBATCH --dependency=singleton
# ☝️ only one job with this name permitted

#    We could let Slurm cancel our job after 1 day, but then we would get a heart attack every time there is a "FAILED"
#    email. Better let the job complete an hour before being cancelled.
#    |
#    |               script arguments are: port, threads
#    |               |
srun --time=23:00:00 resources/scripts/run_corenlp.sh 9000 16

# Get names of pending or running jobs from own user. If at least one job name contains the substring "needs_corenlp",
# reschedule this job.
jobnames=$(squeue --me -O "Name:250" --noheader)
if [[ $jobnames == *"needs_corenlp"* ]]; then
    sbatch --quiet resources/slurm/submit_corenlp.sh
fi