#!/bin/bash
# usage: hf_jobs/build_certify.sh <out.py>   Bundles the full check of finished models: header + helpers + kaggle/15_certify.py.
cd "$(dirname "$0")/.." && { cat hf_jobs/header_real.py kaggle/common.py; grep -v '^sh("pip install' kaggle/15_certify.py; } | sed 's#/kaggle/#/tmp/kg/#g' > "$1"
