"""How big is each run's edit next to the 4-bit rounding error in the same matrices?

RMU edits three MLP matrices. If a 4-bit file gave knowledge back because rounding wiped out a small edit, the edit
would be small next to the rounding error. This measures it for every run of the sweep, per edited matrix:

    ||edited - original|| / ||Q4_0(edited) - edited||      (Frobenius norms)

Q4_0 is the bit-exact round trip from kaggle/common.py.

    python scripts/edit_vs_rounding.py <tiny-aya-global dir> <dir with the sweep's .safetensors files>
writes results/raw/edit_vs_rounding.json
"""
from __future__ import annotations

import json
import os
import sys

import torch
from safetensors import safe_open

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kaggle"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def main(base_dir: str, sweep_dir: str) -> None:
    from common import fq_q4_0
    from export_sweep import RUNS, fname

    index = json.load(open(os.path.join(base_dir, "model.safetensors.index.json")))["weight_map"]
    base: dict = {}

    def original(name: str) -> torch.Tensor:
        if name not in base:
            with safe_open(os.path.join(base_dir, index[name]), "pt") as f:
                base[name] = f.get_tensor(name).float()
        return base[name]

    rows = []
    for i in RUNS:
        path = os.path.join(sweep_dir, fname(i))
        if not os.path.exists(path):
            continue
        ratios, rel = [], []
        with safe_open(path, "pt") as f:
            for name in sorted(f.keys(), key=lambda n: int(n.split(".layers.")[1].split(".")[0])):
                w = f.get_tensor(name).float()
                edit = w - original(name)
                ratios.append(round(float(edit.norm() / (fq_q4_0(w) - w).norm()), 4))
                rel.append(round(float(edit.norm() / original(name).norm()), 4))
        rows.append({"id": i, "layers": RUNS[i][2], "edit_over_rounding": ratios, "mean": round(sum(ratios) / len(ratios), 4),
                     "edit_over_weight": rel})
        print(rows[-1], flush=True)
    json.dump(rows, open(os.path.join(ROOT, "results", "raw", "edit_vs_rounding.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
