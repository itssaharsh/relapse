"""The storage-format round trips the runs used match llama.cpp's own reference code bit for bit."""
import numpy as np
import pytest
import torch

from common import FQ, is_quant_target, roundtrip

pytest.importorskip("gguf")


@pytest.mark.parametrize("name,qname", [("q4_0", "Q4_0"), ("q4_1", "Q4_1")])
def test_fake_quant_is_bit_exact_with_llama_cpp(name, qname):
    from gguf.constants import GGMLQuantizationType as Q
    from gguf.quants import dequantize, quantize

    x = (np.random.default_rng(0).standard_normal((64, 256)) * 0.02).astype(np.float32)
    ref = dequantize(quantize(x, getattr(Q, qname)), getattr(Q, qname)).reshape(x.shape)
    assert np.array_equal(ref, FQ[name](torch.from_numpy(x)).numpy())


@pytest.mark.parametrize("prec", ["Q8_0", "Q4_0"])
def test_roundtrip_matches_gguf(prec):
    from gguf.constants import GGMLQuantizationType as Q
    from gguf.quants import dequantize, quantize

    x = (np.random.default_rng(1).standard_normal((16, 256)) * 0.02).astype(np.float32)
    ref = dequantize(quantize(x, getattr(Q, prec)), getattr(Q, prec)).reshape(x.shape)
    assert np.array_equal(roundtrip(torch.from_numpy(x), prec).numpy(), ref)


def test_fp16_is_identity_and_bf16_rounds():
    t = torch.randn(8, 32) * 0.02
    assert torch.equal(roundtrip(t, "fp16"), t)
    assert torch.equal(roundtrip(t, "bf16"), t.to(torch.bfloat16).float())
    assert not torch.equal(roundtrip(t, "bf16"), t)


def test_nf4_fake_quant_picks_the_nearest_code():
    from common import _NF4, fq_nf4

    x = torch.randn(512, 64) * 0.02
    am = x.abs().max(dim=1, keepdim=True).values
    ref = _NF4[((x / am).unsqueeze(-1) - _NF4).abs().argmin(-1)] * am
    assert torch.equal(fq_nf4(x.reshape(-1)).reshape(x.shape), ref)


def test_quant_targets_are_the_decoder_linear_weights():
    assert is_quant_target("model.layers.3.mlp.down_proj.weight", torch.zeros(2, 2))
    assert not is_quant_target("model.embed_tokens.weight", torch.zeros(2, 2))
    assert not is_quant_target("model.layers.3.input_layernorm.weight", torch.zeros(2))
