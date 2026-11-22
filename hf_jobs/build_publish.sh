#!/bin/bash
# usage: hf_jobs/build_publish.sh <out.py>   Bundles the job that publishes one released model.
cd "$(dirname "$0")/.." && cat hf_jobs/header_real.py kaggle/common.py kaggle/18_publish.py | sed 's#/kaggle/#/tmp/kg/#g' > "$1"
