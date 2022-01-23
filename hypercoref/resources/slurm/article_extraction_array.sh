#!/bin/bash
#
# Finds .yaml files under 'resources/yaml/article_extraction/english/' and uses these to start article extraction jobs.
#
#SBATCH --job-name=hypercoref_array__needs_corenlp
#SBATCH --chdir=/.../hypercoref/
#SBATCH --exclude=...
# ☝️ keep the CoreNLP node free
#SBATCH --mem=64512
#SBATCH --time=6-23:00:00
#SBATCH --array=0-60%2
# ☝️ note that the upper bound is closed, i.e. 0-3 produces four jobs

# disable debugging if not specified
if [ -z "$debug" ]; then
    debug=false
fi

JOB_YAML="$(/.../hypercoref/venv/bin/python3 python/util/slurm_job_array_ith_file.py 'resources/yaml/article_extraction/english' '.yaml')"

srun /.../hypercoref/venv/bin/python3 run_pipeline.py \
  --debug $debug \
  --ip xxx.xxx.xxx.xxx \
  --port 12345 \
  run \
  resources/yaml/device_slurm.yaml \
  resources/yaml/article_extraction/base.yaml \
  $JOB_YAML