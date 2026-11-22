"""Exploratory: the Base seed-12 relapse scored inside llama.cpp's own runtime (kaggle/19_llamacpp_runtime.py).

    python scripts/analyze_runtime.py [--fetch]
Each GGUF file (f16 and Q4_0, of the original Tiny Aya Base and of the seed-12 run at learning rate 1e-3) was scored on the
first 600 hazard questions through llama-cpp-python on CPU. Written before the results existed; not pre-registered.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analyze_map import ROOT, download, paired_p

OUT = os.path.join(ROOT, "results", "runtime")
MODELS = {"orig": "base", "s12": "base_W10-12_lr0.001_s12"}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fetch", action="store_true"); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.fetch:
        for m in MODELS:
            for prec in ("f16", "Q4_0"):
                for off in (0, 300):
                    for f in ("runtime.json", "runtime_items.json"):
                        download(f"real/runtime_{m}_{prec}_{off}/{f}", os.path.join(OUT, f"{m}_{prec}_{off}_{f}"))
    rng = np.random.default_rng(0)
    res = {}
    for m, key in MODELS.items():
        right, same, n = {}, 0, 0
        for prec in ("f16", "Q4_0"):
            right[prec] = []
            for off in (0, 300):
                right[prec] += json.load(open(os.path.join(OUT, f"{m}_{prec}_{off}_runtime_items.json")))[f"{key}|{prec}"]
                row = json.load(open(os.path.join(OUT, f"{m}_{prec}_{off}_runtime.json")))[0]
                same += row["prompts_tokenized_like_official"]; n += row["n"]
        delta, lo, hi, p = paired_p(right["f16"], right["Q4_0"], rng)
        res[m] = {"questions": len(right["f16"]), "f16": float(np.mean(right["f16"])), "Q4_0": float(np.mean(right["Q4_0"])), "change": delta, "ci95": [lo, hi], "p": p,
                  "prompts_tokenized_like_official": same, "prompts": n}
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    for m, r in res.items():
        print(m, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
