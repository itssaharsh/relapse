"""Build a full Hugging Face checkpoint of an unlearned Tiny Aya by patching the base model's safetensors in place.

RMU only changes three MLP down-projection matrices, so the unlearned checkpoint is the base checkpoint with
those tensors' bytes overwritten (same name, shape and dtype). No full model is loaded into memory.

    python scripts/patch_checkpoint.py --base <tiny-aya-global dir> --delta changed.pt --key <run> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct

import torch


def header(path: str) -> tuple[int, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def patch(base: str, out: str, tensors: dict[str, torch.Tensor]) -> list[str]:
    os.makedirs(out, exist_ok=True)
    index = json.load(open(os.path.join(base, "model.safetensors.index.json")))["weight_map"]
    for f in os.listdir(base):
        src = os.path.join(base, f)
        if os.path.isfile(src) and not f.startswith("."):
            shutil.copyfile(src, os.path.join(out, f))
    done = []
    for name, t in tensors.items():
        shard = os.path.join(out, index[name])
        start, hdr = header(shard)
        meta = hdr[name]
        assert meta["dtype"] == "BF16", meta["dtype"]
        assert list(t.shape) == meta["shape"], (name, t.shape, meta["shape"])
        raw = t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()
        b, e = meta["data_offsets"]
        assert e - b == len(raw)
        with open(shard, "r+b") as f:
            f.seek(start + b)
            f.write(raw)
        done.append(name)
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--delta", required=True, help="torch file {run: {param: tensor}}")
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tensors = torch.load(a.delta, map_location="cpu")[a.key]
    print("patched", patch(a.base, a.out, tensors))


if __name__ == "__main__":
    main()
