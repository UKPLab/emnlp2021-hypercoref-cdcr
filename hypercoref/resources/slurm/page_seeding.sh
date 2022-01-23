#!/bin/bash
#
#SBATCH --job-name=hypercoref_page_seeding
#SBATCH --chdir=/.../hypercoref/
#SBATCH --exclude=...
# ☝️ keep the CoreNLP node free
#SBATCH --mem=262144
#SBATCH --time=2-23:00:00

# disable debugging if not specified
if [ -z "$debug" ]; then
    debug=false
fi

srun /.../hypercoref/venv/bin/python3 run_pipeline.py \
  --debug $debug \
  --ip xxx.xxx.xxx.xxx \
  --port 12345 \
  run \
  resources/yaml/device_slurm.yaml \
  resources/yaml/page_seeding/page_seeding.yaml