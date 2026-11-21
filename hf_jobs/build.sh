#!/bin/bash
# usage: hf_jobs/build.sh <out.py>   Bundles the job: header + the same helpers and RMU script the Kaggle runs used.
cd "$(dirname "$0")/.." && { cat hf_jobs/header.py kaggle/common.py; grep -v '^sh("pip install' kaggle/06_rmu_circuitkit_layers_grid.py; } | sed 's#/kaggle/#/tmp/kg/#g' > "$1"
