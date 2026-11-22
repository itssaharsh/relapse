#!/bin/bash
# usage: hf_jobs/build_npo.sh <out.py>   Bundles the NPO (SafeTune, LoRA) job: header + helpers + kaggle/07_npo_lora_sweep.py.
cd "$(dirname "$0")/.." && { cat hf_jobs/header.py kaggle/common.py; grep -v '^sh("pip install' kaggle/07_npo_lora_sweep.py; } | sed 's#/kaggle/#/tmp/kg/#g' > "$1"
