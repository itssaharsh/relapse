#!/bin/bash
# usage: hf_jobs/build_shift.sh <out.py>   Bundles the pipeline-shift job: header + the shared helpers + kaggle/13_pipeline_shift.py.
cd "$(dirname "$0")/.." && cat hf_jobs/header.py kaggle/common.py kaggle/13_pipeline_shift.py | sed 's#/kaggle/#/tmp/kg/#g' > "$1"
