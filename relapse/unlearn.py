"""Unlearning runs on Tiny Aya through Lexsi SafeTune.

rmu(): SafeTune's RMU (Li et al., 2024) on a truncated copy of the model. RMU only reads activations at
`layer`, so layers above it are dropped while training (fits a 16 GB T4 in fp32); the trained MLP
down-projections are returned and written back into the full model.
quant_aware=True trains the same objective through FakeQuantSTE, so the forgetting has to hold for the
4-bit version of the weights too (the Relapse fix).
"""
from __future__ import annotations

import random

import torch
import torch.nn.utils.parametrize as parametrize

from .precision import FakeQuantSTE


def _truncated(model_path: str, layer: int, device="cuda:0"):
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_path)
    cfg.num_hidden_layers = layer + 1
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = cfg.layer_types[: layer + 1]
    m = AutoModelForCausalLM.from_pretrained(model_path, config=cfg, dtype=torch.float32, device_map={"": device})
    return m.model.eval()


def activation_norm(inner, batches, layer: int, n: int = 8) -> float:
    norms = []
    h = inner.layers[layer].register_forward_hook(
        lambda m, i, o: norms.append((o[0] if isinstance(o, tuple) else o).norm(dim=-1).mean().item()))
    with torch.no_grad():
        for b in batches[:n]:
            inner(**b)
    h.remove()
    return sum(norms) / len(norms)


def rmu(model_path: str, forget_batches, retain_batches, *, layer: int = 7, alpha: float = 100.0,
        lr: float = 4e-5, mult: float = 3.0, steps: int = 250, quant_aware: bool = False,
        p_quant: float = 0.5, seed: int = 0) -> dict[str, torch.Tensor]:
    """Returns {full-model param name: trained fp32 tensor} for the updated down_proj weights."""
    from safetune.interventions.unlearn.rmu import RMUConfig, rmu_unlearn

    upd = [layer - 2, layer - 1, layer]
    frozen = _truncated(model_path, layer)
    for p in frozen.parameters():
        p.requires_grad_(False)
    coeff = mult * activation_norm(frozen, forget_batches, layer)
    torch.manual_seed(seed)
    random.seed(seed)
    model = _truncated(model_path, layer)
    if quant_aware:
        for l in upd:
            parametrize.register_parametrization(model.layers[l].mlp.down_proj, "weight", FakeQuantSTE(p_quant))
    cfg = RMUConfig(layer_id=layer, update_layer_ids=upd, steering_coeff=coeff, alpha=alpha, lr=lr,
                    max_num_batches=steps)
    rmu_unlearn(model, retain_batches=retain_batches, forget_batches=forget_batches, frozen_model=frozen, config=cfg)
    if quant_aware:
        for l in upd:
            parametrize.remove_parametrizations(model.layers[l].mlp.down_proj, "weight", leave_parametrized=False)
    out = {"model." + n: p.detach().float().cpu().clone() for n, p in model.named_parameters()
           if any(f"layers.{l}.mlp.down_proj" in n for l in upd)}
    del model, frozen
    torch.cuda.empty_cache()
    return out
