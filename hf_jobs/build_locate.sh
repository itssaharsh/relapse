#!/bin/bash
# usage: hf_jobs/build_locate.sh <out.py>   Bundles the job that rounds one part of the model at a time: header + helpers + kaggle/12_locate_relapse.py.
cd "$(dirname "$0")/.." && { cat hf_jobs/header_real.py kaggle/common.py; grep -v '^sh("pip install' kaggle/12_locate_relapse.py; } | sed 's#/kaggle/#/tmp/kg/#g' > "$1"
