#!/bin/bash
# usage: hf_jobs/build_consolidate.sh <out.py>   Bundles the file-consolidation job: header + helpers + kaggle/16_consolidate.py.
cd "$(dirname "$0")/.." && cat hf_jobs/header.py kaggle/common.py kaggle/16_consolidate.py | sed 's#/kaggle/#/tmp/kg/#g' > "$1"
