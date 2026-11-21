#!/bin/bash
# usage: hf_jobs/build_trigger.sh <out.py>   Bundles the job that records how strongly each saved edit fires, per file format.
cd "$(dirname "$0")/.." && { cat hf_jobs/header_real.py kaggle/common.py; grep -v '^sh("pip install' kaggle/14_trigger.py; } | sed 's#/kaggle/#/tmp/kg/#g' > "$1"
