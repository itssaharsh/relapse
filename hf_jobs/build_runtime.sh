#!/bin/bash
# usage: hf_jobs/build_runtime.sh <out.py>   Bundles the job that scores GGUF files through llama.cpp's own runtime.
cd "$(dirname "$0")/.." && cat hf_jobs/header_real.py kaggle/common.py kaggle/19_llamacpp_runtime.py | sed 's#/kaggle/#/tmp/kg/#g' > "$1"
