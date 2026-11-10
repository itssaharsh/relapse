"""Export every unlearning run of the sweep as a small safetensors file (the three edited MLP matrices, fp16) plus a card.

Each file applies to CohereLabs/tiny-aya-global with scripts/patch_checkpoint.py --delta <file>.safetensors.

    python scripts/export_sweep.py <dir with the kernels' changed_*.pt files> <out dir>
"""
from __future__ import annotations

import glob
import json
import os
import sys

import torch
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# frontier id -> (file, key, layers edited, learning rate, retain weight, steering multiplier, batches)
RUNS = {
    "d_lr2e-5": ("changed_rmu3.pt", "rmu_lr2e-5", "5-7", 2e-5, 100, 3, 250),
    "d_lr4e-5": ("changed_rmu3.pt", "rmu_lr4e-5", "5-7", 4e-5, 100, 3, 250),
    "d_lr7e-5": ("changed_rmu3.pt", "rmu_lr7e-5", "5-7", 7e-5, 100, 3, 250),
    "d_lr4e-5_qa": ("changed_rmu3.pt", "qarmu_lr4e-5", "5-7", 4e-5, 100, 3, 250),
    "d_lr1e-4": ("changed_rmu2.pt", "L7_a100_lr1e-4", "5-7", 1e-4, 100, 3, 300),
    "d_lr3e-4": ("changed_rmu2.pt", "L7_a100_lr3e-4", "5-7", 3e-4, 100, 3, 300),
    "d_a10_lr1e-4": ("changed_rmu2.pt", "L7_a10_lr1e-4", "5-7", 1e-4, 10, 3, 300),
    "d_lr4e-5_s1": ("changed_seed1.pt", "rmu_default_layers_seed1", "5-7", 4e-5, 100, 3, 250),
    "c_lr1e-4": ("changed_rmu2.pt", "L9ck_a100_lr1e-4", "7-9", 1e-4, 100, 3, 300),
    "c_a300_lr2e-4": ("changed_rmu4.pt", "L9_a300_lr2e-4", "7-9", 2e-4, 300, 3, 300),
    "c_lr1.5e-4": ("changed_rmu4.pt", "L9_lr1.5e-4", "7-9", 1.5e-4, 100, 3, 300),
    "c_lr1.5e-4_qa": ("changed_rmu4.pt", "L9_lr1.5e-4_qa", "7-9", 1.5e-4, 100, 3, 300),
    "c_lr2e-4": ("changed_push.pt", "L9_lr2e-4", "7-9", 2e-4, 100, 3, 300),
    "c_lr2e-4_qa": ("changed_push.pt", "L9_lr2e-4_qa", "7-9", 2e-4, 100, 3, 300),
    "c_lr1.5e-4_m5_qa": ("changed_push.pt", "L9_lr1.5e-4_m5_qa", "7-9", 1.5e-4, 100, 5, 300),
    "c_lr1.5e-4_s1": ("changed_seed1.pt", "rmu_circuit_layers_seed1", "7-9", 1.5e-4, 100, 3, 300),
    # controls: the default layers at the CircuitKIT runs' budget, a repeat of the CircuitKIT run, and other windows
    "d_lr4e-5_nb300": ("changed_ctrl_shallow.pt", "D5-7_lr4e-5_nb300", "5-7", 4e-5, 100, 3, 300),
    "d_lr7e-5_nb300": ("changed_ctrl_shallow.pt", "D5-7_lr7e-5_nb300", "5-7", 7e-5, 100, 3, 300),
    "d_lr4e-5_m5": ("changed_ctrl_shallow.pt", "D5-7_lr4e-5_m5", "5-7", 4e-5, 100, 5, 300),
    "d_a300_lr1e-4": ("changed_ctrl_shallow.pt", "D5-7_a300_lr1e-4", "5-7", 1e-4, 300, 3, 300),
    "c_lr1.5e-4_rep": ("changed_ctrl_deeper.pt", "C7-9_lr1.5e-4_f16ref", "7-9", 1.5e-4, 100, 3, 300),
    "x3_lr1.5e-5": ("changed_ctrl_shallow.pt", "X3-5_lr1.5e-5", "3-5", 1.5e-5, 100, 3, 300),
    "x3_lr3e-5": ("changed_ctrl_shallow.pt", "X3-5_lr3e-5", "3-5", 3e-5, 100, 3, 300),
    "x3_lr6e-5": ("changed_ctrl_shallow.pt", "X3-5_lr6e-5", "3-5", 6e-5, 100, 3, 300),
    "x10_lr2e-4": ("changed_ctrl_deeper.pt", "X10-12_lr2e-4", "10-12", 2e-4, 100, 3, 300),
    "x10_lr4.5e-4": ("changed_ctrl_deeper.pt", "X10-12_lr4.5e-4", "10-12", 4.5e-4, 100, 3, 300),
    "x10_lr1e-3": ("changed_ctrl_deeper.pt", "X10-12_lr1e-3", "10-12", 1e-3, 100, 3, 300),
    "x10_lr4.5e-4_s1": ("changed_ctrl_depth2.pt", "X10-12_lr4.5e-4_s1", "10-12", 4.5e-4, 100, 3, 300),
    "x13_lr8e-4": ("changed_ctrl_depth2.pt", "X13-15_lr8e-4", "13-15", 8e-4, 100, 3, 300),
    "x13_lr2e-3": ("changed_ctrl_depth2.pt", "X13-15_lr2e-3", "13-15", 2e-3, 100, 3, 300),
}
PREFIX = {"d": "default-layers_", "c": "circuitkit-layers_"}   # other windows: layers-<first>-<last>_


def fname(i: str) -> str:
    head, rest = i.split("_", 1)
    return PREFIX.get(head, f"layers-{RUNS[i][2]}_") + rest.replace("_qa", "_4bit-aware").replace("_s1", "_seed1") + ".safetensors"


def main(src: str, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    res = json.load(open(os.path.join(ROOT, "space", "data", "results.json")))
    pts = {(p["id"], p["prec"]): p for p in res["frontier"]}
    cache: dict = {}
    rows = []
    for i, (f, key, layers, lr, alpha, mult, nb) in RUNS.items():
        path = glob.glob(os.path.join(src, "**", f), recursive=True)[0]
        if path not in cache:
            cache.clear()
            cache[path] = torch.load(path, map_location="cpu")
        tensors = {n: t.to(torch.float16).contiguous() for n, t in cache[path][key].items()}   # fp16: the precision the fp16 rows were scored at
        meta = {"base_model": "CohereLabs/tiny-aya-global", "method": "SafeTune RMU", "layers": layers, "lr": str(lr),
                "retain_weight": str(alpha), "steering_multiplier": str(mult), "batches": str(nb),
                "four_bit_aware": str(i.endswith("_qa") or "_qa_" in i), "seed": "1" if i.endswith("_s1") else "0"}
        save_file(tensors, os.path.join(out, fname(i)), metadata=meta)
        cell = lambda prec: (f"{100 * pts[(i, prec)]['forget']:.1f}% / {100 * pts[(i, prec)]['retain']:.1f}%" if (i, prec) in pts else "not run")
        rows.append(f"| `{fname(i)}` | {layers} | {lr:g} | {alpha} | {mult} | {'yes' if meta['four_bit_aware'] == 'True' else 'no'} | {meta['seed']} | {cell('fp16')} | {cell('Q4_0')} | {cell('NF4')} |")
        print("wrote", fname(i))
    base = pts[("original", "fp16")]
    card = f"""---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
tags: [unlearning, safety, rmu, safetune, wmdp, quantization, tiny-aya, lexsi, relapse]
datasets: [cais/wmdp, cais/mmlu]
---

# Relapse: every unlearning run of the sweep

All {len(RUNS)} [SafeTune](https://github.com/Lexsi-Labs/SafeTune) RMU runs behind
[Relapse](https://huggingface.co/spaces/itssaharsh/relapse), one small file each. RMU edits only three MLP down-projection
matrices, so a file holds those three tensors in fp16 (the scores below were computed from the fp32 values they were
rounded from). The runs put the same edit at five layer windows: SafeTune's default (5 to 7), the one
[CircuitKIT](https://github.com/Lexsi-Labs/CircuitKIT)'s attribution ranked first (7 to 9), and three controls (3 to 5,
10 to 12, 13 to 15).

**The `layers-10-12_*` files are the relapse.** They look unlearned at fp16 and give part of the hazard score back in the
Q4_0 column. They are here as test cases for an unlearning audit, not as models to use.

Apply a file to [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) with
`python scripts/patch_checkpoint.py --base <tiny-aya-global dir> --delta <file>.safetensors --out <dir>` from
[github.com/itssaharsh/relapse](https://github.com/itssaharsh/relapse).

Scores are WMDP-bio (forget, lower is better; guessing is 25%) / MMLU (keep, higher is better). The original scores
{100 * base['forget']:.1f}% / {100 * base['retain']:.1f}% at fp16.

| File | Layers | Learning rate | Retain weight | Steering multiplier | 4-bit aware | Seed (0 first, 1 second) | fp16 | Q4_0 | NF4 |
|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(rows) + """

Four of these are also published as full checkpoints with GGUF files in the
[Relapse models collection](https://huggingface.co/collections/itssaharsh/relapse-unlearned-tiny-aya-models). All scores,
per-question results and logs are in [itssaharsh/relapse-runs](https://huggingface.co/datasets/itssaharsh/relapse-runs).

A research artifact. Outside layers 10 to 12 the forgetting holds across file formats; for the headline models it also
holds across languages and prompt attacks. It does not survive fine-tuning on the topic. The base model's license and the Cohere Labs acceptable use policy apply (non-commercial).
"""
    open(os.path.join(out, "README.md"), "w").write(card)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
