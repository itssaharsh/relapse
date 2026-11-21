#!/bin/bash
# usage: hf_jobs/build_real.sh <out.py>   Bundles the real-file job: header + the helpers and grid script the Kaggle runs used.
cd "$(dirname "$0")/.." && { cat hf_jobs/header_real.py kaggle/common.py; grep -v '^sh("pip install' kaggle/08_final_grid.py; } | sed 's#/kaggle/#/tmp/kg/#g' > "$1"
