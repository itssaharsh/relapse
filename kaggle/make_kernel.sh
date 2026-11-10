#!/bin/bash
# usage: ./make_kernel.sh <script.py> > run.py   (prepends the shared helpers)
cd "$(dirname "$0")" && cat common.py "$1"
