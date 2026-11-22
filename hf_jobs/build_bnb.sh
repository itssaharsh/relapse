#!/bin/bash
# usage: hf_jobs/build_bnb.sh <out.py>   Bundles the job that loads saved models through bitsandbytes NF4 and scores them.
cd "$(dirname "$0")/.." && cat hf_jobs/header_real.py kaggle/common.py kaggle/17_bnb_real.py | sed 's#/kaggle/#/tmp/kg/#g' > "$1"
