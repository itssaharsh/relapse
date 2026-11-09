"""Real llama.cpp round trips: HF checkpoint -> GGUF -> llama-quantize (Q4_K_M, Q6_K, ...) -> weights back
into the PyTorch model for scoring on GPU."""
from __future__ import annotations

import glob
import os
import subprocess
import sys

import torch

LLAMA = os.environ.get("RELAPSE_LLAMA_CPP", "/tmp/llama.cpp")


def _sh(cmd: str) -> None:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{cmd[:120]}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def setup(build: bool = True) -> None:
    """Clone llama.cpp (for convert_hf_to_gguf.py and gguf-py) and get a llama-quantize binary."""
    if not os.path.exists(LLAMA):
        _sh(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {LLAMA}")
        # tiny-aya-global's tokenizer hashes differently from tiny-aya-base under transformers 5.x;
        # it is the same tokenizer family, so an unknown hash maps to the tiny_aya pre-tokenizer.
        bp = f"{LLAMA}/conversion/base.py"
        src = open(bp).read().replace(
            'raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")',
            'res = "tiny_aya"')
        open(bp, "w").write(src)
    sys.path.insert(0, f"{LLAMA}/gguf-py")
    binp = f"{LLAMA}/build/bin/llama-quantize"
    if build and not os.path.exists(binp):
        pre = glob.glob("/kaggle/input/**/llama-quantize", recursive=True)
        os.makedirs(os.path.dirname(binp), exist_ok=True)
        if pre:
            _sh(f"cp {pre[0]} {binp} && chmod +x {binp}")
        else:
            _sh(f"cd {LLAMA} && cmake -B build -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF "
                f"&& cmake --build build --target llama-quantize -j 4")


def hf_to_gguf(hf_dir: str, out: str, outtype: str = "f16") -> str:
    _sh(f"cd {LLAMA} && python convert_hf_to_gguf.py {hf_dir} --outtype {outtype} --outfile {out}")
    return out


def quantize(src: str, dst: str, qtype: str) -> str:
    _sh(f"{LLAMA}/build/bin/llama-quantize {src} {dst} {qtype}")
    return dst


@torch.no_grad()
def load_into(model, path: str) -> dict[str, str]:
    """Overwrite the model's parameters with the dequantized tensors of a GGUF file. Returns name -> type."""
    import gguf
    from gguf import GGUFReader
    from gguf.quants import dequantize

    tmap = gguf.get_tensor_name_map(gguf.MODEL_ARCH.COHERE2, model.config.num_hidden_layers)
    names = {}
    for n, _ in model.named_parameters():
        g = tmap.get_name(n, try_suffixes=(".weight", ".bias"))
        if g:
            names[g] = n
    params = dict(model.named_parameters())
    types = {}
    for t in GGUFReader(path).tensors:
        hf = names.get(t.name)
        if hf is None:
            continue
        arr = dequantize(t.data, t.tensor_type).reshape(list(reversed([int(x) for x in t.shape])))
        params[hf].copy_(torch.from_numpy(arr.astype("float32")).to(params[hf].dtype))
        types[hf] = t.tensor_type.name
    missing = [n for n in params if n not in types]
    if missing:
        raise RuntimeError(f"{len(missing)} parameters not found in {path}: {missing[:3]}")
    return types
