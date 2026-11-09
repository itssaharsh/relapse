"""Storage-precision round trips: what the weights become when a model is saved at bf16, GGUF or NF4.

bf16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1 and NF4 are exact re-implementations (Q8_0..Q5_1 use llama.cpp's own
gguf-py code; NF4 uses bitsandbytes' kernels). k-quants (Q4_K_M, Q6_K, ...) go through real llama.cpp
in relapse.gguf.
"""
from __future__ import annotations

import random

import torch

GGUF_SIMPLE = {"Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1"}


def _gguf_roundtrip(t: torch.Tensor, qname: str) -> torch.Tensor:
    from gguf.constants import GGMLQuantizationType as Q
    from gguf.quants import dequantize, quantize

    qt = getattr(Q, qname)
    a = t.detach().float().cpu().numpy()
    return torch.from_numpy(dequantize(quantize(a, qt), qt).reshape(a.shape))


def _nf4_roundtrip(t: torch.Tensor) -> torch.Tensor:
    import bitsandbytes.functional as F

    x = t.detach().to("cuda", torch.float16)
    q, state = F.quantize_4bit(x, blocksize=64, quant_type="nf4")
    return F.dequantize_4bit(q, state).float().cpu()


def roundtrip(t: torch.Tensor, prec: str) -> torch.Tensor:
    """fp32 CPU tensor holding the values the weights take when stored at `prec`."""
    if prec in ("fp32", "fp16"):
        return t.detach().float().cpu()
    if prec == "bf16":
        return t.detach().to(torch.bfloat16).float().cpu()
    if prec.lower() == "nf4":
        return _nf4_roundtrip(t)
    if prec in GGUF_SIMPLE:
        return _gguf_roundtrip(t, prec)
    raise ValueError(f"{prec}: use relapse.gguf for llama.cpp k-quants")


def is_quant_target(name: str, p: torch.Tensor) -> bool:
    """The decoder's linear weights (what llama.cpp and bitsandbytes quantize to low bits)."""
    return p.dim() == 2 and ".layers." in name and name.endswith("_proj.weight")


# --- fake quantizers for quantization-aware unlearning (bit-exact with llama.cpp Q4_0 / Q4_1) ---
_NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                     -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                     0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                     0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])


def fq_q4_0(w: torch.Tensor, bs: int = 32) -> torch.Tensor:
    x = w.reshape(-1, bs)
    amax = torch.gather(x, 1, x.abs().argmax(dim=1, keepdim=True))
    d = amax / -8.0
    inv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    q = torch.clamp(torch.floor(x * inv + 8.5), 0, 15)
    return ((q - 8.0) * d.half().float()).reshape(w.shape)


def fq_q4_1(w: torch.Tensor, bs: int = 32) -> torch.Tensor:
    x = w.reshape(-1, bs)
    mn, mx = x.min(dim=1, keepdim=True).values, x.max(dim=1, keepdim=True).values
    d = (mx - mn) / 15.0
    inv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    q = torch.clamp(torch.floor((x - mn) * inv + 0.5), 0, 15)
    return (q * d.half().float() + mn.half().float()).reshape(w.shape)


def fq_nf4(w: torch.Tensor, bs: int = 64) -> torch.Tensor:
    x = w.reshape(-1, bs)
    am = x.abs().max(dim=1, keepdim=True).values.clamp_min(1e-12)
    code = _NF4.to(w.device, w.dtype)
    q = (x.div(am).unsqueeze(-1) - code).abs().argmin(dim=-1)
    return (code[q] * am).reshape(w.shape)


FAKE_QUANT = {"q4_0": fq_q4_0, "q4_1": fq_q4_1, "nf4": fq_nf4}


class FakeQuantSTE(torch.nn.Module):
    """Weight parametrization: with probability p the forward pass sees the weight after a random 4-bit
    round trip, with a straight-through gradient. Training through it makes an edit survive 4-bit storage."""

    def __init__(self, p: float = 0.5, kinds=("q4_0", "q4_1", "nf4")):
        super().__init__()
        self.p, self.kinds = p, list(kinds)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return w
        with torch.no_grad():
            wq = FAKE_QUANT[random.choice(self.kinds)](w.detach())
        return w + (wq - w).detach()
