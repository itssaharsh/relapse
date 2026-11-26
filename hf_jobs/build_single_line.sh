#!/bin/bash
# usage: hf_jobs/build_single_line.sh <out.py>   Bundles the single-line rescoring job of pre-registration 6: header + the shared helpers + kaggle/20_single_line.py.
cd "$(dirname "$0")/.." && cat hf_jobs/header_single.py kaggle/common.py kaggle/20_single_line.py | sed 's#/kaggle/#/tmp/kg/#g' > "$1"
